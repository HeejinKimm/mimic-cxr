# -*- coding: utf-8 -*-
"""
local_update.py
- prep_clients.py가 만든 임베딩(repr_img.npy, repr_txt.npy 등)을 이용해
  각 클라이언트의 분류 헤드(Linear)를 Z 기반으로 소규모 업데이트합니다.
- best.pt의 head(fuse.3.weight/bias)를 가능한 한 자동으로 불러와 초기화합니다.
- 임베딩 파일 우선순위: repr_* > train_*  (index도 client_splits로 fallback)
"""

import os, json, argparse, csv
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, roc_auc_score

from local_train.config import Cfg, BASE  # BASE는 모듈 레벨 상수

# ---------------------------
# 전역 상수/설정
# ---------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

LABEL_COLUMNS = Cfg.LABEL_COLUMNS
N_CLASSES = len(LABEL_COLUMNS)

# 기본 라벨 CSV 자동 선택
DEFAULT_LABEL_CSV = str(Cfg.LABEL_CSV_NEGBIO if getattr(Cfg, "USE_LABEL", "negbio").lower() == "negbio"
                        else Cfg.LABEL_CSV_CHEXPERT)

# ---------------------------
# 경로/도움 함수
# ---------------------------
def cdir(cid: int) -> Path:
    """client 디렉토리 (BASE/local_train_outputs/client_XX)"""
    d = BASE / "outputs" / f"client_{cid:02d}"
    return d

def _first_existing(base: Path, names: List[str]) -> Optional[Path]:
    for n in names:
        p = base / n
        if p.exists():
            return p
    return None

def fmt4(x: float) -> str:
    return "nan" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x:.4f}"

# ---------------------------
# 데이터(임베딩 + 인덱스)
# ---------------------------
def _normalize_index_df(df: pd.DataFrame) -> pd.DataFrame:
    """split/index 컬럼을 내부 표준으로 정규화: has_image/has_text -> has_img/has_txt"""
    # 필수 ID 컬럼 확인
    need_cols = {"subject_id", "study_id"}
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(f"index에 필수 컬럼 누락: {missing}")

    # has_* 컬럼 통일
    if "has_img" not in df.columns and "has_image" in df.columns:
        df["has_img"] = df["has_image"]
    if "has_txt" not in df.columns and "has_text" in df.columns:
        df["has_txt"] = df["has_text"]
    if "has_img" not in df.columns:
        df["has_img"] = 0
    if "has_txt" not in df.columns:
        df["has_txt"] = 0

    # True/False/문자열 -> 0/1
    def _to01(x):
        if isinstance(x, str):
            s = x.strip().lower()
            if s in ("true", "1", "y", "yes"): return 1
            if s in ("false", "0", "n", "no"): return 0
        if isinstance(x, (bool, np.bool_)):
            return int(bool(x))
        try:
            return int(x)
        except Exception:
            return 0

    df["has_img"] = df["has_img"].map(_to01)
    df["has_txt"] = df["has_txt"].map(_to01)
    return df

def load_embeddings_and_index(cid: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[Dict[str, str]]]:
    """
    임베딩 파일 우선순위:
      IMG: repr_img.npy > train_img_reps.npy
      TXT: repr_txt.npy > train_txt_reps.npy
    인덱스(index) 우선순위:
      client_XX 폴더의 index.csv > train_index.csv > index_train.csv
      없으면 BASE/client_splits/client_XX.csv fallback
    """
    base = cdir(cid)
    Xi = Xt = None

    # 임베딩
    p_img = _first_existing(base, ["repr_img.npy", "train_img_reps.npy"])
    p_txt = _first_existing(base, ["repr_txt.npy", "train_txt_reps.npy"])
    if p_img is not None:
        Xi = np.load(p_img)
        print(f"[client_{cid:02d}] using IMG reps: {p_img.name} shape={Xi.shape}")
    if p_txt is not None:
        Xt = np.load(p_txt)
        print(f"[client_{cid:02d}] using TXT reps: {p_txt.name} shape={Xt.shape}")

    # 인덱스
    p_idx = _first_existing(base, ["index.csv", "train_index.csv", "index_train.csv"])
    if p_idx is None:
        alt = Cfg.CLIENT_CSV_DIR / f"client_{cid:02d}.csv"
        if alt.exists():
            p_idx = alt
    if p_idx is None:
        raise FileNotFoundError(f"[client_{cid:02d}] index 파일을 찾을 수 없음")

    df = pd.read_csv(p_idx)
    df = _normalize_index_df(df)
    rows = df.to_dict("records")
    print(f"[client_{cid:02d}] using INDEX: {p_idx.name} rows={len(rows)}")

    if (Xi is None) and (Xt is None):
        raise FileNotFoundError(f"[client_{cid:02d}] 임베딩 파일이 없음")

    return Xi, Xt, rows

# ---------------------------
# 라벨 테이블
# ---------------------------
def load_label_table(label_csv: str) -> Dict[Tuple[int, int], List[int]]:
    if not os.path.exists(label_csv):
        raise FileNotFoundError(f"라벨 CSV가 없습니다: {label_csv}")
    df = pd.read_csv(label_csv)
    required = {"subject_id", "study_id", *LABEL_COLUMNS}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"라벨 CSV 칼럼 누락: {missing}")

    for c in LABEL_COLUMNS:
        df[c] = df[c].fillna(0)
        df[c] = (df[c] >= 1).astype(int)

    table = {}
    for _, r in df.iterrows():
        key = (int(r["subject_id"]), int(r["study_id"]))
        table[key] = [int(r[c]) for c in LABEL_COLUMNS]
    return table

def idstr_to_ints(p: str, s: str) -> Tuple[int, int]:
    # p like 'p10000032', s like 's53189527'
    return int(str(p)[1:]), int(str(s)[1:])

def build_embed_dataset(cid: int, label_csv: str):
    Xi, Xt, rows = load_embeddings_and_index(cid)

    i_ptr = 0; t_ptr = 0
    Ximg_list, Xtxt_list, Y_list, mask_img, mask_txt = [], [], [], [], []

    label_table = load_label_table(label_csv)

    for r in rows:
        sid, stid = idstr_to_ints(str(r["subject_id"]), str(r["study_id"]))
        y = label_table.get((sid, stid), [0] * len(LABEL_COLUMNS))
        has_img = int(r.get("has_img", 0)) == 1 and Xi is not None
        has_txt = int(r.get("has_txt", 0)) == 1 and Xt is not None

        if has_img and i_ptr < (0 if Xi is None else Xi.shape[0]):
            Ximg_list.append(Xi[i_ptr]); i_ptr += 1; mask_img.append(1)
        else:
            Ximg_list.append(np.zeros((0,), dtype=np.float32)); mask_img.append(0)

        if has_txt and t_ptr < (0 if Xt is None else Xt.shape[0]):
            Xtxt_list.append(Xt[t_ptr]); t_ptr += 1; mask_txt.append(1)
        else:
            Xtxt_list.append(np.zeros((0,), dtype=np.float32)); mask_txt.append(0)

        Y_list.append(y)

    Y = torch.tensor(np.array(Y_list, dtype=np.float32), device=DEVICE)   # [N, C]
    N = len(rows)
    Di = 0 if Xi is None else Xi.shape[1]
    Dt = 0 if Xt is None else Xt.shape[1]

    def pad_stack(lst, D):
        if D == 0: return torch.zeros((N, 0), device=DEVICE)
        X = np.zeros((N, D), dtype=np.float32)
        for idx in range(N):
            v = lst[idx]
            if v.shape == (0,): continue
            X[idx] = v
        return torch.tensor(X, device=DEVICE)

    Ximg = pad_stack(Ximg_list, Di)   # (N, Di)
    Xtxt = pad_stack(Xtxt_list, Dt)   # (N, Dt)
    m_img = torch.tensor(np.array(mask_img, dtype=np.float32), device=DEVICE)  # (N,)
    m_txt = torch.tensor(np.array(mask_txt, dtype=np.float32), device=DEVICE)  # (N,)

    has_any_img = (Di > 0) and (m_img.sum().item() > 0)
    has_any_txt = (Dt > 0) and (m_txt.sum().item() > 0)
    if has_any_img and has_any_txt:
        mode = "multimodal"
    elif has_any_img:
        mode = "image_only"
    elif has_any_txt:
        mode = "text_only"
    else:
        raise RuntimeError(f"[client_{cid:02d}] 사용 가능한 임베딩이 없습니다.")

    return Ximg, Xtxt, Y, m_img, m_txt, mode

# ---------------------------
# Z 로드 & 정렬 손실(게이팅)
# ---------------------------
def load_Z_for_client(cid: int) -> Optional[torch.Tensor]:
    # 1) global_payload.json에서 경로 추출
    p_payload = cdir(cid) / "global_payload.json"
    z_path = None
    if p_payload.exists():
        payload = json.loads(p_payload.read_text(encoding="utf-8"))
        z_path = payload.get("Z_path") or payload.get("Z_proxy_text_path") or payload.get("Z_proxy_image_path")
    # 2) 로컬 폴더에 Z.npy 류가 있으면 fallback
    if z_path is None:
        for name in ["Z.npy", "Z_img0_txt0.npy"]:
            alt = cdir(cid) / name
            if alt.exists():
                z_path = str(alt); break

    if not z_path or not os.path.exists(z_path):
        return None

    Z = torch.from_numpy(np.load(z_path)).float().to(DEVICE)
    Z = torch.nn.functional.normalize(Z, dim=1)
    print(f"[client_{cid:02d}] Z loaded: shape={tuple(Z.shape)}")
    return Z

def _pad_to_dim(X: torch.Tensor, D: int) -> torch.Tensor:
    Dx = X.shape[1]
    if Dx == D: return X
    pad = torch.zeros((X.size(0), D - Dx), device=X.device, dtype=X.dtype)
    return torch.cat([X, pad], dim=1)

def _fused_feature(Fi: Optional[torch.Tensor],
                   Ft: Optional[torch.Tensor],
                   mi: torch.Tensor, mt: torch.Tensor) -> Optional[torch.Tensor]:
    if Fi is None and Ft is None: return None
    if Fi is None: return Ft
    if Ft is None: return Fi
    D = max(Fi.shape[1], Ft.shape[1])
    Fi_p = torch.nn.functional.normalize(_pad_to_dim(Fi, D), dim=1)
    Ft_p = torch.nn.functional.normalize(_pad_to_dim(Ft, D), dim=1)
    denom = (mi + mt).clamp(min=1.0)
    return (Fi_p * mi + Ft_p * mt) / denom

def alignment_loss_with_gating(F: torch.Tensor, Z: torch.Tensor,
                               base: float = 0.1, tau: float = 0.1) -> torch.Tensor:
    if (Z is None) or (F is None) or (F.numel() == 0):
        return torch.tensor(0.0, device=F.device if isinstance(F, torch.Tensor) else DEVICE)
    D = max(F.shape[1], Z.shape[1])
    Fp = torch.nn.functional.normalize(_pad_to_dim(F, D), dim=1)
    Zp = torch.nn.functional.normalize(_pad_to_dim(Z, D), dim=1)
    sim = Fp @ Zp.t()                 # (B, K)
    sim_max, _ = sim.max(dim=1)       # (B,)
    gate = torch.sigmoid((1.0 - sim_max) / max(tau, 1e-6))
    loss_align = (1.0 - sim_max) * gate
    return base * loss_align.mean()

# ---------------------------
# best.pt에서 헤드 로딩
# ---------------------------
def load_baseline_head_from_bestpt(cid: int, in_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    best.pt에서 분류 헤드 가중치 로드.
    1) fuse.3.weight/bias 우선
    2) fallback: shape == (N_CLASSES, in_dim)
    없으면 scratch로 초기화
    """
    ckpt_path = cdir(cid) / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"[client_{cid:02d}] best.pt 없음: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model", ckpt)

    W = b = None
    # 1) fuse.3 우선
    for k, v in sd.items():
        if k.endswith("weight") and isinstance(v, torch.Tensor) and v.ndim == 2 and v.shape[0] == N_CLASSES:
            if "fuse.3" in k:
                W = v; b = sd.get(k.replace("weight", "bias"))
                print(f"[client_{cid:02d}] found head: {k} shape={tuple(v.shape)}")
                break
    # 2) shape fallback
    if W is None:
        for k, v in sd.items():
            if isinstance(v, torch.Tensor) and v.ndim == 2 and v.shape == (N_CLASSES, in_dim):
                W = v; b = sd.get(k.replace("weight", "bias"))
                print(f"[client_{cid:02d}] fallback head: {k} shape={tuple(v.shape)}")
                break

    if W is None or b is None:
        print(f"[WARN] client_{cid:02d} head 불일치 → scratch 초기화 (in_dim={in_dim})")
        W = torch.randn((N_CLASSES, in_dim))
        b = torch.zeros(N_CLASSES)

    return W.to(DEVICE).float(), b.to(DEVICE).float()

# ---------------------------
# 학습/평가 유틸
# ---------------------------
class LinearHead(nn.Module):
    def __init__(self, d_in: int, n_out: int):
        super().__init__()
        self.fc = nn.Linear(d_in, n_out)
    def forward(self, X):
        return self.fc(X)

def evaluate_logits_vs_labels(logits: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    crit = nn.BCEWithLogitsLoss()
    loss = crit(logits, y).item()
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    y_true = y.detach().cpu().numpy().astype(np.int32)
    y_pred = (probs >= 0.5).astype(np.int32)
    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    aucs = []
    for c in range(y_true.shape[1]):
        col = y_true[:, c]
        if len(np.unique(col)) < 2: continue
        try:
            aucs.append(roc_auc_score(col, probs[:, c]))
        except Exception:
            pass
    auc_macro = float(np.mean(aucs)) if len(aucs) > 0 else float("nan")
    return {"loss": loss, "f1_micro": float(f1_micro), "f1_macro": float(f1_macro), "auc_macro": auc_macro}

# ---------------------------
# 메인
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--client_id", type=str, default="1-20", help='예: "1-20" 또는 "1,3,5"')
    ap.add_argument("--label_csv", type=str, default=DEFAULT_LABEL_CSV)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--z_weight", type=float, default=0.1)
    ap.add_argument("--tau", type=float, default=0.1)
    args = ap.parse_args()

    # client id 파싱
    text = args.client_id.strip().lower()
    ids: List[int] = []
    if text in ("all", ""):
        ids = list(range(1, 21))
    else:
        for part in text.split(","):
            part = part.strip()
            if "-" in part:
                a, b = part.split("-", 1)
                ids.extend(list(range(int(a), int(b) + 1)))
            else:
                ids.append(int(part))
        ids = sorted(list(dict.fromkeys(ids)))

    out_csv = BASE / "outputs" / "update_embed_summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows_out: List[Dict[str, object]] = []

    for cid in ids:
        print("\n" + "=" * 70)
        print(f"▶ Client {cid:02d} | build dataset from embeddings")
        print("=" * 70)

        try:
            Ximg, Xtxt, Y, m_img, m_txt, mode_now = build_embed_dataset(cid, args.label_csv)
        except Exception as e:
            print(f"[SKIP client {cid:02d}] {e}")
            continue

        # late-fusion 입력 차원
        Di = Ximg.shape[1] if Ximg is not None else 0
        Dt = Xtxt.shape[1] if Xtxt is not None else 0

        # best.pt에서 head 로딩 (fuse.3 우선)
        # 본 스크립트는 하나의 fused head를 업데이트하므로, 입력은 '사용되는 모달의 융합 특징'
        D_in_fused = max(Di, Dt) if (Di > 0 or Dt > 0) else 0
        if D_in_fused == 0:
            print(f"[SKIP client {cid:02d}] no usable embeddings")
            continue
        W0, b0 = load_baseline_head_from_bestpt(cid, D_in_fused)

        # 학습용 입력 특징 준비: 간단한 late-fusion 평균 (모달 마스크 고려)
        def fuse_feats(Xi, Xt, mi, mt):
            Fi = Xi if Di > 0 else None
            Ft = Xt if Dt > 0 else None
            mi = mi.unsqueeze(1)
            mt = mt.unsqueeze(1)
            F = _fused_feature(Fi, Ft, mi, mt)
            if F is None:  # 안전장치
                F = torch.zeros((Y.size(0), D_in_fused), device=DEVICE)
            return F

        F_all = fuse_feats(Ximg, Xtxt, m_img, m_txt)

        # baseline 평가
        with torch.no_grad():
            logits_base = F_all @ W0.t() + b0
        baseline_val = evaluate_logits_vs_labels(logits_base, Y)
        print("[Baseline on val] ", baseline_val)

        # 모델/옵티마이저
        head = nn.Linear(D_in_fused, N_CLASSES).to(DEVICE)
        with torch.no_grad():
            head.weight.copy_(W0); head.bias.copy_(b0)
        opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
        crit = nn.BCEWithLogitsLoss()

        # Z 로드
        Z = load_Z_for_client(cid)

        # train/val split (10% val 동일 분할)
        N = Y.shape[0]
        idx = np.arange(N); rng = np.random.RandomState(SEED); rng.shuffle(idx)
        n_val = max(1, int(N * 0.1))
        va_idx = torch.tensor(idx[:n_val], device=DEVICE)
        tr_idx = torch.tensor(idx[n_val:], device=DEVICE)

        # 학습 루프
        for ep in range(1, args.epochs + 1):
            head.train()
            opt.zero_grad()

            F_tr = F_all[tr_idx]
            y_tr = Y[tr_idx]
            logits_tr = head(F_tr)
            loss_main = crit(logits_tr, y_tr)
            loss_z = alignment_loss_with_gating(F_tr, Z, base=args.z_weight, tau=args.tau) if Z is not None else 0.0
            loss = loss_main + (loss_z if isinstance(loss_z, torch.Tensor) else 0.0)

            loss.backward()
            opt.step()

            head.eval()
            with torch.no_grad():
                logits_va = head(F_all[va_idx])
                val_metrics = evaluate_logits_vs_labels(logits_va, Y[va_idx])
            print(f"[Ep {ep}/{args.epochs}] train_loss={loss.item():.4f} | val {val_metrics}")

        # 최종 updated 성능(같은 val 분할로)
        head.eval()
        with torch.no_grad():
            logits_upd = head(F_all[va_idx])
            updated_val = evaluate_logits_vs_labels(logits_upd, Y[va_idx])
        print("[Updated on val]", updated_val)

        # 저장
        save_path = cdir(cid) / "updated_heads.npz"
        np.savez(save_path,
                 weight=head.weight.detach().cpu().numpy(),
                 bias=head.bias.detach().cpu().numpy())
        print(f"[client {cid:02d}] updated heads saved → {save_path}")

        rows_out.append({
            "client_id": cid,
            "mode_now": mode_now,
            "baseline_loss": baseline_val["loss"], "updated_loss": updated_val["loss"],
            "baseline_f1_micro": baseline_val["f1_micro"], "updated_f1_micro": updated_val["f1_micro"],
            "baseline_f1_macro": baseline_val["f1_macro"], "updated_f1_macro": updated_val["f1_macro"],
            "baseline_auc_macro": baseline_val["auc_macro"], "updated_auc_macro": updated_val["auc_macro"],
        })

    if rows_out:
        with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            w.writeheader(); w.writerows(rows_out)
        print(f"\n[DONE] Summary saved → {out_csv} (rows={len(rows_out)})")
    else:
        print("\n[WARN] no clients processed; nothing saved.")

if __name__ == "__main__":
    main()
