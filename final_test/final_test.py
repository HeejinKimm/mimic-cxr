# -*- coding: utf-8 -*-
"""
final_test.py (shared test set version)
- 모든 클라이언트가 동일한 test.csv와 공유 테스트 임베딩을 사용해 분류 수행
- 각 클라이언트는 자기 소속 (img_group, txt_group)에서 나온 최종 Z를 사용
- 파이프라인: late-fusion R(α,β) → LSTM gating(Forget=Z, Input=R) → 분류
- 산출물:
    final_test/
      ├─ client_{cid:02d}_preds.csv
      └─ client_{cid:02d}_metrics.json
"""

import os
import json
import csv
import argparse
from typing import Optional, Dict, Tuple, List

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None

# -------------------------
# 경로
# -------------------------
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GLOBAL_OUT = os.path.join(PROJ_ROOT, "global_output")
CLUSTER_DIR = os.path.join(GLOBAL_OUT, "clustering")
FINAL_Z_DIR = os.path.join(GLOBAL_OUT, "final_output_z")
LOCAL_OUT = os.path.join(PROJ_ROOT, "local_train_outputs")
FINAL_TEST_DIR = os.path.join(PROJ_ROOT, "final_test")
CLIENT_SPLITS_DIR = os.path.join(PROJ_ROOT, "client_splits")

# -------------------------
# 유틸
# -------------------------
def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def l2n(x: np.ndarray, eps: float = 1e-12):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n = np.maximum(n, eps)
    return x / n

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def try_load(path: str) -> Optional[np.ndarray]:
    return np.load(path).astype(np.float32) if os.path.exists(path) else None

def seed_proj(in_dim: int, out_dim: int, seed: int = 2025) -> np.ndarray:
    rng = np.random.default_rng(seed + in_dim * 1000 + out_dim)
    W = rng.standard_normal((in_dim, out_dim)).astype(np.float32)
    W /= max(np.linalg.norm(W, ord=2), 1e-6)
    return W

# -------------------------
# 클러스터→Z 매핑
# -------------------------
def load_cluster_assignments() -> Tuple[Dict[int,int], Dict[int,int]]:
    img_asn, txt_asn = {}, {}
    p_img = os.path.join(CLUSTER_DIR, "img_client_assignments.json")
    p_txt = os.path.join(CLUSTER_DIR, "txt_client_assignments.json")
    if os.path.exists(p_img):
        data = load_json(p_img)
        for a in data.get("assignments", []):
            img_asn[int(a["client_id"])] = int(a["assigned_group"])
    if os.path.exists(p_txt):
        data = load_json(p_txt)
        for a in data.get("assignments", []):
            txt_asn[int(a["client_id"])] = int(a["assigned_group"])
    return img_asn, txt_asn

def load_final_z_index():
    idx_path = os.path.join(FINAL_Z_DIR, "final_output_z_index.json")
    if not os.path.exists(idx_path):
        raise FileNotFoundError(f"final_output_z_index.json not found in {FINAL_Z_DIR}")
    idx = load_json(idx_path)
    pairs = idx.get("pairs", [])
    img2txt, txt2img, pair2z = {}, {}, {}
    for p in pairs:
        gi = int(p["img_group"]); tj = int(p["txt_group"])
        pair2z[(gi, tj)] = p["z_file"]
        img2txt[gi] = tj
        txt2img[tj] = gi
    d_model = int(idx.get("d_model", 256))
    return img2txt, txt2img, pair2z, d_model

def resolve_z_for_client(cid: int,
                         img_asn: Dict[int,int],
                         txt_asn: Dict[int,int],
                         img2txt: Dict[int,int],
                         txt2img: Dict[int,int],
                         pair2z: Dict[Tuple[int,int], str]) -> Tuple[np.ndarray, str, Tuple[int,int]]:
    if cid in img_asn:
        gi = img_asn[cid]
        tj = img2txt.get(gi, 0)
    elif cid in txt_asn:
        tj = txt_asn[cid]
        gi = txt2img.get(tj, 0)
    else:
        gi, tj = 0, img2txt.get(0, 0)

    z_file = pair2z.get((gi, tj))
    if z_file is None:
        raise KeyError(f"No Z for pair (img_group={gi}, txt_group={tj})")
    z_path = os.path.join(FINAL_Z_DIR, z_file)
    if not os.path.exists(z_path):
        raise FileNotFoundError(f"Z file not found: {z_path}")
    Z = np.load(z_path).astype(np.float32)
    return Z, z_path, (gi, tj)

# -------------------------
# 공유 테스트 임베딩 로드
# -------------------------
def _candidate_dirs_for_shared() -> List[str]:
    return [
        os.path.join(GLOBAL_OUT, "shared_test"),
        os.path.join(GLOBAL_OUT),                 # 바로 아래
        os.path.join(FINAL_TEST_DIR),
        os.path.join(LOCAL_OUT, "common"),
        os.path.join(LOCAL_OUT, "client_01"),     # 최후 폴백
    ]

def load_shared_test_arrays(test_csv_path: str):
    """
    - test.csv 전 행을 사용하여 (Xi, Xt, Y)를 구성
    - 공유 풀(test_img_reps.npy, test_txt_reps.npy, test_labels.npy)을 여러 경로에서 탐색
    - label은 test_labels.npy가 없으면 test.csv의 'label' 컬럼 사용
    """
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Shared test.csv not found at {test_csv_path}")

    # 공유 풀 찾기
    I_pool = T_pool = Y_pool = None
    pool_root = None
    for d in _candidate_dirs_for_shared():
        i_p = os.path.join(d, "test_img_reps.npy")
        t_p = os.path.join(d, "test_txt_reps.npy")
        y_p = os.path.join(d, "test_labels.npy")
        i_arr = try_load(i_p)
        t_arr = try_load(t_p)
        y_arr = try_load(y_p)
        if (i_arr is not None) or (t_arr is not None) or (y_arr is not None):
            I_pool, T_pool, Y_pool = i_arr, t_arr, y_arr
            pool_root = d
            break
    if (I_pool is None) and (T_pool is None):
        raise FileNotFoundError(
            "Cannot find shared test embeddings. Expected test_img_reps.npy or test_txt_reps.npy "
            f"in one of: {', '.join(_candidate_dirs_for_shared())}"
        )

    # test.csv 읽기 (전 행 사용)
    rows = []
    with open(test_csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    if not rows:
        raise RuntimeError("Shared test.csv has no rows.")

    # 행별로 인덱스 파싱
    Ximg, Xtxt, Y = [], [], []
    for row in rows:
        # 인덱스 컬럼: img_idx/txt_idx 우선, 없으면 index/idx 공용 사용
        img_idx = row.get("img_idx")
        txt_idx = row.get("txt_idx")
        if img_idx is None and txt_idx is None:
            k = row.get("index", row.get("idx"))
            img_idx = k; txt_idx = k

        if I_pool is not None and (img_idx is not None):
            Ximg.append(I_pool[int(img_idx)])
        else:
            Ximg.append(None)

        if T_pool is not None and (txt_idx is not None):
            Xtxt.append(T_pool[int(txt_idx)])
        else:
            Xtxt.append(None)

        # 라벨은 test_labels.npy가 우선, 없으면 CSV의 label
        if Y_pool is not None:
            # test.csv와 길이 일치한다고 가정 (인덱스 무시하고 순서 사용)
            pass
        else:
            Y.append(int(row.get("label", row.get("y", 0))))

    # None 채우기 및 스택
    def infer_dim(arr):
        if arr is not None and arr.size > 0 and arr.ndim == 2:
            return int(arr.shape[1])
        return 256

    d_img = infer_dim(I_pool)
    d_txt = infer_dim(T_pool)
    Xi = None if all(v is None for v in Ximg) else np.stack(
        [(np.zeros((d_img,), np.float32) if v is None else v.astype(np.float32)) for v in Ximg], axis=0)
    Xt = None if all(v is None for v in Xtxt) else np.stack(
        [(np.zeros((d_txt,), np.float32) if v is None else v.astype(np.float32)) for v in Xtxt], axis=0)

    if Y_pool is not None:
        Y = Y_pool.astype(np.int64)
        # 길이가 csv 행 수와 다르면 최소 길이에 맞춰 자름
        n = len(rows)
        if len(Y) != n:
            m = min(len(Y), n)
            Y = Y[:m]
            if Xi is not None: Xi = Xi[:m]
            if Xt is not None: Xt = Xt[:m]
    else:
        Y = np.array(Y, dtype=np.int64)

    return Xi, Xt, Y, pool_root

# -------------------------
# Late-fusion (R) + LSTM Gating
# -------------------------
def late_fusion_R(Xi: Optional[np.ndarray], Xt: Optional[np.ndarray],
                  d_model: int, alpha: float = 0.5, beta: float = 0.5, seed: int = 2025) -> np.ndarray:
    comps = []
    if Xi is not None:
        di = Xi.shape[1]
        Xi_ = Xi @ seed_proj(di, d_model, seed=seed + 1) if di != d_model else Xi
        comps.append(alpha * l2n(Xi_))
    if Xt is not None:
        dt = Xt.shape[1]
        Xt_ = Xt @ seed_proj(dt, d_model, seed=seed + 2) if dt != d_model else Xt
        comps.append(beta * l2n(Xt_))
    if not comps:
        return np.zeros((1, d_model), np.float32)
    R = np.zeros_like(comps[0])
    for c in comps:
        R += c
    return R

def lstm_gating_fuse(R: np.ndarray, Z: np.ndarray, tz: float = 1.0) -> np.ndarray:
    d = Z.shape[0]
    Zb = Z.reshape(1, d)
    f = sigmoid(tz * Zb)
    i = sigmoid(R)
    g = np.tanh(R)
    c = f * Zb + i * g
    h = np.tanh(c)
    return h.astype(np.float32)

# -------------------------
# 분류기
# -------------------------
class LateFusionMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def predict_param_free(h: np.ndarray, Z: np.ndarray) -> np.ndarray:
    hhat = l2n(h); Zhat = l2n(Z.reshape(1, -1)).reshape(-1)
    s = np.sum(hhat * Zhat[None, :], axis=1)
    return sigmoid(s.astype(np.float32))

def predict_with_local_mlp(model_path: str, h: np.ndarray, Z: np.ndarray, device: str) -> Optional[np.ndarray]:
    if torch is None or not os.path.exists(model_path):
        return None
    in_dim = h.shape[1] + Z.shape[0]
    model = LateFusionMLP(in_dim)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()
    with torch.no_grad():
        N = h.shape[0]
        Zi = np.repeat(Z.reshape(1, -1), N, axis=0).astype(np.float32)
        x = np.concatenate([h.astype(np.float32), Zi], axis=1)
        logits = model(torch.from_numpy(x).to(device)).cpu().numpy().astype(np.float32)
        return sigmoid(logits)

# -------------------------
# 메트릭
# -------------------------
def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_true = y_true.astype(np.int64)
    y_pred = (y_prob >= 0.5).astype(np.int64)
    acc = float((y_pred == y_true).mean())
    try:
        thr = np.unique(y_prob); thr = np.concatenate([[0.0], thr, [1.0]])
        TPR, FPR = [], []
        P, N = (y_true == 1).sum(), (y_true == 0).sum()
        for t in thr:
            yp = (y_prob >= t).astype(np.int64)
            TP = ((yp == 1) & (y_true == 1)).sum()
            FP = ((yp == 1) & (y_true == 0)).sum()
            TPR.append(TP / P if P > 0 else 0.0)
            FPR.append(FP / N if N > 0 else 0.0)
        order = np.argsort(FPR)
        auc = float(np.trapz(np.array(TPR)[order], np.array(FPR)[order]))
    except Exception:
        auc = None
    return {"accuracy": acc, "roc_auc": auc}

# -------------------------
# 실행
# -------------------------
def run(alpha: float = 0.5, beta: float = 0.5, tz: float = 1.0,
        clients: Optional[str] = None, device: str = "cpu"):
    ensure_dir(FINAL_TEST_DIR)

    # 0) 공유 테스트 셋 로드 (한 번만)
    test_csv_path = os.path.join(CLIENT_SPLITS_DIR, "test.csv")
    Xi, Xt, Y, pool_root = load_shared_test_arrays(test_csv_path)
    n_samples = Xi.shape[0] if Xi is not None else (Xt.shape[0] if Xt is not None else len(Y))
    print(f"[FINAL-TEST] Shared test loaded from: {pool_root}  (N={n_samples})")

    # 1) 클러스터/매핑/Z 준비
    img_asn, txt_asn = load_cluster_assignments()
    img2txt, txt2img, pair2z, d_model = load_final_z_index()

    # 2) 테스트 R(α,β) 선계산 (모든 클라 공통)
    R = late_fusion_R(Xi, Xt, d_model, alpha=alpha, beta=beta)

    # 3) 평가 대상 클라이언트
    if clients:
        client_list = [int(x.strip()) for x in clients.split(",") if x.strip()]
    else:
        client_list = sorted(set(list(img_asn.keys()) + list(txt_asn.keys())))
        # 혹시 비어있으면 1..20 가정
        if not client_list:
            client_list = list(range(1, 21))
    print(f"[FINAL-TEST] targets: {client_list}")

    # 4) 각 클라이언트별로 Z만 달리 적용
    for cid in client_list:
        # 4-1) Z 결정
        Z, z_path, (gi, tj) = resolve_z_for_client(cid, img_asn, txt_asn, img2txt, txt2img, pair2z)
        # 4-2) LSTM gating → h
        h = lstm_gating_fuse(R, Z, tz=tz)  # (N,d_model)
        # 4-3) 분류기 (있으면 사용)
        clf_path = os.path.join(LOCAL_OUT, f"client_{cid:02d}", "late_fusion_classifier.pt")
        y_prob = predict_with_local_mlp(clf_path, h, Z, device=device)
        used_local = y_prob is not None
        if y_prob is None:
            y_prob = predict_param_free(h, Z)

        # 4-4) 저장
        pred_csv = os.path.join(FINAL_TEST_DIR, f"client_{cid:02d}_preds.csv")
        with open(pred_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["index", "prob", "pred_label", "true_label"])
            for i, p in enumerate(y_prob):
                w.writerow([i, float(p), int(p >= 0.5), int(Y[i])])

        metrics = compute_metrics(Y[:len(y_prob)], y_prob)
        meta = {
            "client_id": cid,
            "img_group": gi, "txt_group": tj,
            "z_file": z_path,
            "n_samples": int(len(Y)),
            "alpha": alpha, "beta": beta, "tz": tz,
            "used_local_classifier": used_local,
            "test_pool_root": pool_root,
            "metrics": metrics
        }
        with open(os.path.join(FINAL_TEST_DIR, f"client_{cid:02d}_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"[FINAL-TEST] client {cid:02d} → acc={metrics['accuracy']:.4f}, auc={metrics['roc_auc']}")

    print(f"[FINAL-TEST] done -> {FINAL_TEST_DIR}")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.5, help="late fusion IMG 가중치")
    ap.add_argument("--beta",  type=float, default=0.5, help="late fusion TXT 가중치")
    ap.add_argument("--tz",    type=float, default=1.0, help="forget gate에서 Z 온도(scale)")
    ap.add_argument("--clients", type=str, default=None, help="쉼표구분 리스트(예: 1,2,19). 생략시 전체.")
    ap.add_argument("--device", type=str, default="cuda" if (torch is not None and torch.cuda.is_available()) else "cpu")
    ap.add_argument("--img_pool_path", type=str, default=None,
                help="공유 이미지 임베딩 .npy 절대경로 (test_img_reps.npy / repr_img_kd.npy / train_img_reps.npy 중 하나)")
    ap.add_argument("--txt_pool_path", type=str, default=None,
                help="공유 텍스트 임베딩 .npy 절대경로 (test_txt_reps.npy / repr_txt_kd.npy / train_txt_reps.npy 중 하나)")
    ap.add_argument("--labels_path", type=str, default=None,
                help="공유 레이블 .npy 절대경로 (없으면 test.csv의 label 컬럼 사용)")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(alpha=args.alpha, beta=args.beta, tz=args.tz, clients=args.clients, device=args.device)
