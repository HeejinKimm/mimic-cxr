# local_gating/train_local_kd.py
# -*- coding: utf-8 -*-
import os
import json
import argparse
import numpy as np
import torch
from torch.optim import AdamW

# ---- 우리 프로젝트 내부 모듈 ----
from global_train.config import cfg
from global_train.prep_clients import build_dataloaders   # 로더 재사용
## from ..local import utils_fusion as uf                # 백본 인코더
from local import train_local as tl


from .kd_utils import compute_pos_weight, kd_repr_loss
from .model_head import LocalClassifierWithAugment


# ---------------------------
# 페이로드(Z) 로더 (JSON → .npy)
# ---------------------------
def load_payload_for_client(cid: int):
    """
    global_outputs/client_{cid}/global_payload.json을 읽어
    Z 경로를 찾고 .npy를 로드해 1D torch.Tensor로 반환.
    우선순위: Z_proxy_text_path → Z_proxy_image_path → Z_path
    """
    p_json = os.path.join(cfg.OUT_GLOBAL_DIR, f"client_{cid}", "global_payload.json")
    if not os.path.exists(p_json):
        raise FileNotFoundError(f"payload json not found: {p_json}")

    payload = json.loads(open(p_json, "r", encoding="utf-8").read())

    z_path = (payload.get("Z_proxy_text_path")
              or payload.get("Z_proxy_image_path")
              or payload.get("Z_path"))
    if not z_path or not os.path.exists(z_path):
        raise FileNotFoundError(f"Z path not found in payload or file does not exist: {z_path}")

    Z = np.load(z_path)
    if Z.ndim == 2:   # (K, d)면 평균으로 집계 (원하면 median/max로 바꿔도 됨)
        Z = Z.mean(axis=0)
    if Z.ndim != 1:
        raise ValueError(f"unexpected Z shape: {Z.shape} (expect [d] or [K,d])")

    Z = torch.tensor(Z, dtype=torch.float32)  # [d_model]
    T = float(payload.get("kd_temperature", cfg.KD_TEMP))
    W = float(payload.get("kd_rep_weight", cfg.KD_REP_WEIGHT))
    role = payload.get("role", "unknown")

    return Z, T, W, role, payload


# ---------------------------
# 백본 인코더(고정)
# ---------------------------
def build_backbone(device: torch.device):
    """
    utils_fusion.FusionClassifier의 인코더(img_enc/txt_enc)만 사용.
    필요하면 여기서 client별 체크포인트를 불러오도록 확장 가능.
    """
    model = tl.MultiModalLateFusion(num_classes=cfg.NUM_CLASSES).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def extract_reps(backbone, batch, device):
    """
    배치에서 이미지/텍스트 임베딩 추출 (존재하는 것만).
    반환: (zi or None, zt or None)
    """
    pv  = batch.get("pixel_values")
    ids = batch.get("input_ids")
    am  = batch.get("attention_mask")

    to_dev = (lambda x: x.to(device, non_blocking=True) if x is not None else None)
    pv, ids, am = to_dev(pv), to_dev(ids), to_dev(am)

    zi = backbone.img_enc(pv)           if (pv  is not None and hasattr(backbone, "img_enc")) else None
    zt = backbone.txt_enc(ids, am)      if (ids is not None and hasattr(backbone, "txt_enc")) else None
    return zi, zt


# ---------------------------
# 학습/평가 루프
# ---------------------------
def train_one_epoch(backbone, head, loader, Z, device, bce, optimizer, kd_weight: float):
    head.train()
    Z = Z.to(device)
    total_bce = 0.0
    total_kd  = 0.0

    for b in loader:
        y = b["labels"].to(device).float()

        with torch.no_grad():
            zi, zt = extract_reps(backbone, b, device)

        logits, R = head(img_rep=zi, txt_rep=zt, Z_global=Z)
        loss_bce  = bce(logits, y)
        loss_kd   = kd_repr_loss(R, Z, kd_weight)
        loss = loss_bce + loss_kd

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()

        total_bce += float(loss_bce.detach().cpu())
        total_kd  += float(loss_kd.detach().cpu())

    n = max(1, len(loader))
    return {"bce": total_bce / n, "kd": total_kd / n, "total": (total_bce + total_kd) / n}


@torch.no_grad()
def evaluate(backbone, head, loader, Z, device):
    from sklearn.metrics import f1_score, roc_auc_score
    head.eval(); Z = Z.to(device)

    Ls, Ys = [], []
    for b in loader:
        y = b["labels"].to(device).float()
        zi, zt = extract_reps(backbone, b, device)
        logits, _ = head(img_rep=zi, txt_rep=zt, Z_global=Z)
        Ls.append(logits.detach().cpu()); Ys.append(y.detach().cpu())

    L = torch.cat(Ls, 0)
    Y = torch.cat(Ys, 0)

    P = torch.sigmoid(L).numpy()
    Y = Y.numpy().astype("int32")
    yhat = (P >= 0.5).astype("int32")

    f1_micro = f1_score(Y, yhat, average="micro", zero_division=0)
    f1_macro = f1_score(Y, yhat, average="macro", zero_division=0)

    aucs = []
    for c in range(Y.shape[1]):
        col = Y[:, c]
        if len(set(col.tolist())) < 2:
            continue
        try:
            aucs.append(roc_auc_score(col, P[:, c]))
        except Exception:
            pass
    auc_macro = float(sum(aucs) / len(aucs)) if aucs else float("nan")
    return {"f1_micro": float(f1_micro), "f1_macro": float(f1_macro), "auc_macro": auc_macro}


# ---------------------------
# 엔트리
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cid", type=int, required=True, help="클라이언트 ID")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--use_hallucinate", action="store_true",
                    help="결손 모달을 Z로 보완(hallucinate). 미설정 시 0벡터로 대체.")
    args = ap.parse_args()

    device = torch.device(args.device)
    

    # 0) 전역 Z 로드 + KD 하이퍼
    Z, T, W, role, payload = load_payload_for_client(args.cid)
    print(f"[client_{args.cid}] role={role}  kd_repr_weight={W}  kd_temperature={T}")

    # 1) 데이터 로더 (prep_clients의 규약 재사용)
    train_loader, val_loader = build_dataloaders(args.cid, args.batch)

    # 2) pos_weight 기반 BCE
    pw  = compute_pos_weight(train_loader, cfg.NUM_CLASSES).to(device)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=pw)

    # 3) 백본 인코더(고정) + 헤드(학습)
    backbone = build_backbone(device)
    head = LocalClassifierWithAugment(
        d_img=cfg.IMG_DIM,
        d_txt=cfg.TXT_DIM,
        d_local_fused=cfg.FUSED_DIM,
        d_global=cfg.D_MODEL,
        n_labels=cfg.NUM_CLASSES,
        use_hallucinate=args.use_hallucinate,
        z_dropout=0.0,
        feat_dropout=0.0,
    ).to(device)
    optimizer = AdamW(head.parameters(), lr=args.lr)

    best_metric = None
    best_path = os.path.join(cfg.BASE_DIR, f"client_{args.cid}", f"client_{args.cid}_local_gated_best.pt")

    for ep in range(1, args.epochs + 1):
        tr = train_one_epoch(backbone, head, train_loader, Z, device, bce, optimizer, kd_weight=W)
        m  = evaluate(backbone, head, val_loader, Z, device)
        print(f"[client_{args.cid}] ep{ep}  "
              f"train: total={tr['total']:.4f} (bce={tr['bce']:.4f}, kd={tr['kd']:.4f})  "
              f"val: f1_micro={m['f1_micro']:.4f}  f1_macro={m['f1_macro']:.4f}  auc={m['auc_macro']:.4f}")

        # 베스트 갱신 기준: 검증 macro F1 우선, 동률이면 AUC
        is_better = (best_metric is None or
                     (m["f1_macro"], m["auc_macro"]) > (best_metric["f1_macro"], best_metric["auc_macro"]))
        if is_better:
            best_metric = m
            torch.save(
                {
                    "head": head.state_dict(),
                    "meta": {
                        "cid": args.cid,
                        "use_hallucinate": args.use_hallucinate,
                        "d_global": payload.get("d_global", cfg.D_MODEL),
                        "kd_repr_weight": W,
                        "kd_temperature": T,
                        "role": role,
                    },
                },
                best_path,
            )
            print(f"[client_{args.cid}] saved best -> {best_path}")

    # 최종 메트릭 기록
    outj = os.path.join(cfg.BASE_DIR, f"client_{args.cid}", f"client_{args.cid}_local_gated_metrics.json")
    with open(outj, "w", encoding="utf-8") as f:
        json.dump(best_metric, f, indent=2, ensure_ascii=False)
    print(f"[client_{args.cid}] metrics saved -> {outj}")


if __name__ == "__main__":
    main()
