# local_gating/train_local_kd.py
# -*- coding: utf-8 -*-
# python -m local_gating.train_local_kd --cid 1 --epochs 5 --batch 128
import os
import json
import argparse
import numpy as np
import torch
from torch.optim import AdamW

# ---- 우리 프로젝트 내부 모듈 ----
from global_train.config import cfg
## from ..local import utils_fusion as uf                # 백본 인코더
from local_train.data import ClientDataset, build_image_picker_from_metadata, load_label_table, img_transform
from local_train.config import Cfg as LocalCfg
from local_train.models import MultiModalLateFusion as LocalFusion


from .kd_utils import compute_pos_weight, kd_repr_loss
from .model_head import LocalClassifierWithAugment

# train_local_kd.py 맨 위 근처에 임시로 추가해서 써보기
import time, torch

def benchmark(loader, backbone, head, Z, device, n_iters=50):
    backbone.eval(); head.train()
    Z = Z.to(device)
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    times = []
    it = 0
    for b in loader:
        if it >= n_iters: break
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0.record() if torch.cuda.is_available() else None

        with torch.no_grad():
            zi, zt = extract_reps(backbone, b, device)
        logits, R = head(img_rep=zi, txt_rep=zt, Z_global=Z)
        _ = logits  # 역전파 제외(순전파만)

        t1.record() if torch.cuda.is_available() else None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            ms = t0.elapsed_time(t1)  # 밀리초
        else:
            ms = 0.0  # CPU면 time.perf_counter로 감싸도 됨
        times.append(ms/1000.0)  # 초
        it += 1
    return sum(times)/max(1,len(times))  # 배치당 평균 초


def build_dataloaders(cid: int, batch_size: int):
    from torch.utils.data import DataLoader, random_split
    from pathlib import Path

    lcfg = LocalCfg()

    # CSV 경로
    csv_dir = Path(lcfg.CLIENT_CSV_DIR)
    csv_path = csv_dir / f"client_{cid:02d}.csv"
    if not csv_path.exists():
        csv_path = csv_dir / f"client_{cid}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Client CSV not found for client_{cid}: {csv_path}")

    # 라벨 테이블 + 메타
    label_csv = lcfg.LABEL_CSV_NEGBIO if lcfg.USE_LABEL == "negbio" else lcfg.LABEL_CSV_CHEXPERT
    label_table = load_label_table(str(label_csv), lcfg.LABEL_COLUMNS)
    meta_picker = build_image_picker_from_metadata(str(lcfg.METADATA_CSV), str(lcfg.IMG_ROOT))

    # 모드 결정
    mode = (
        "multimodal" if 1 <= cid <= 16 else
        "image_only" if cid in (17, 18) else
        "text_only"  if cid in (19, 20) else
        "unknown"
    )

    # 전체 Dataset 로드
    ds_all = ClientDataset(
        str(csv_path),
        label_table,
        mode,
        meta_picker,
        lcfg.TEXT_MODEL_NAME,
        lcfg.MAX_LEN,
        img_transform()
    )

    # 9:1 random split (seed 고정)
    n = len(ds_all)
    n_tr = int(n * 0.9)
    n_val = n - n_tr
    tr_set, val_set = random_split(
        ds_all, [n_tr, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(tr_set, batch_size=batch_size, shuffle=True,  num_workers=lcfg.NUM_WORKERS)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=lcfg.NUM_WORKERS)
    return train_loader, val_loader

# ---------------------------
# 페이로드(Z) 로더 (JSON → .npy)
# ---------------------------
def load_payload_for_client(cid: int):
    """
    global_output/client_{cid:02d or cid}/global_payload.(json|pt) 를 읽어
    Z (np.ndarray or torch.Tensor)를 1D torch.Tensor 로 반환.
    우선순위: Z_proxy_text → Z_proxy_image → Z
    """
    base = cfg.OUT_GLOBAL_DIR
    d_pad = os.path.join(base, f"client_{cid:02d}")
    d_raw = os.path.join(base, f"client_{cid}")
    cdir = d_pad if os.path.isdir(d_pad) else (d_raw if os.path.isdir(d_raw) else None)
    if cdir is None:
        raise FileNotFoundError(f"client dir not found: {d_pad} or {d_raw}")

    # 파일 후보들(.json 우선, 없으면 .pt)
    pj = os.path.join(cdir, "global_payload.json")
    pp = os.path.join(cdir, "global_payload.pt")

    payload = None
    if os.path.exists(pj):
        import json
        with open(pj, "r", encoding="utf-8") as f:
            payload = json.load(f)
        # JSON은 보통 경로를 들고 있음 → 경로로부터 Z 로드
        z_path = (payload.get("Z_proxy_text_path")
                  or payload.get("Z_proxy_image_path")
                  or payload.get("Z_path"))
        if not z_path or not os.path.exists(z_path):
            raise FileNotFoundError(f"Z path missing or not found in JSON: {z_path}")
        Z = np.load(z_path)
    elif os.path.exists(pp):
        # ✅ payload는 혼합형(dict)이므로 weights_only 쓰지 말 것
        obj = torch.load(pp, map_location="cpu")

        if not isinstance(obj, dict):
            raise TypeError(f"payload .pt must be a dict, got {type(obj)}")

        # 우선순위대로 키 선택
        Z = obj.get("Z_proxy_text") or obj.get("Z_proxy_image") or obj.get("Z")
        if Z is None:
            raise KeyError(f"'Z' not found in payload: keys={list(obj.keys())}")

        payload = obj

        # numpy면 그대로, torch면 numpy로
        if isinstance(Z, torch.Tensor):
            Z = Z.detach().cpu().numpy()
    else:
        raise FileNotFoundError(f"payload file not found: {pj} or {pp}")

    # Z shape 정리: (K,d) → 평균, (d,) 그대로
    if Z.ndim == 2:
        Z = Z.mean(axis=0)
    if Z.ndim != 1:
        raise ValueError(f"unexpected Z shape: {Z.shape} (expect [d] or [K,d])")

    Z = torch.tensor(Z, dtype=torch.float32)  # [d_model]

    # 하이퍼 파싱(없으면 cfg 기본값)
    T = float(payload.get("kd_temperature", cfg.KD_TEMP))
    W = float(payload.get("kd_rep_weight", cfg.KD_REP_WEIGHT))
    role = payload.get("role", "unknown")

    return Z, T, W, role, payload



# ---------------------------
# 백본 인코더(고정)
# ---------------------------
def build_backbone(device: torch.device):
    import torch
    from torch.nn.parameter import UninitializedParameter

    model = LocalFusion(num_classes=cfg.NUM_CLASSES).to(device)
    model.eval()

    # 1) Lazy 모듈 materialize: 멀티모달 더미 입력으로 1회 forward
    #    (LocalFusion의 forward 시그니처가 (img, input_ids, attn_mask)라고 가정)
    try:
        with torch.no_grad():
            x_img = torch.zeros(1, 3, 224, 224, device=device)     # 이미지 더미
            txt_len = 64
            input_ids = torch.ones(1, txt_len, dtype=torch.long, device=device)
            attn_mask = torch.ones(1, txt_len, dtype=torch.long, device=device)
            _ = model(x_img, input_ids, attn_mask)
    except Exception as e:
        # 텍스트/이미지 중 하나만 받는 구조면 필요한 모달만 넣어 재시도
        tried = False
        try:
            with torch.no_grad():
                x_img = torch.zeros(1, 3, 224, 224, device=device)
                _ = model(x_img, None, None)
                tried = True
        except Exception:
            pass
        if not tried:
            try:
                with torch.no_grad():
                    txt_len = 64
                    input_ids = torch.ones(1, txt_len, dtype=torch.long, device=device)
                    attn_mask = torch.ones(1, txt_len, dtype=torch.long, device=device)
                    _ = model(None, input_ids, attn_mask)
            except Exception as e2:
                print(f"[build_backbone] Lazy 초기화 더미 패스 실패: {e!r} / fallback: {e2!r}")

    # 2) 이제 materialize된 뒤 freeze
    for p in model.parameters():
        if isinstance(p, UninitializedParameter):
            # 혹시 남아있다면 안전하게 스킵
            continue
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