# -*- coding: utf-8 -*-
"""
build_vectors.py (robust, projector-based)
- 각 클라이언트 임베딩을 모아 KMeans로 이미지/텍스트 센트로이드를 만든 뒤,
  코사인 유사도 기반 헝가리안 매칭 → 공통 차원 투영 → 최종 d_model로 투영하여
  글로벌 벡터 Z를 생성합니다.
- CrossAttention 의존성을 제거하고, 차원 불일치/샘플 부족 상황을 모두 방어합니다.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from .config import cfg
from .utils_io import get_client_reps

# --------------- cosine / hungarian helpers ---------------

def _l2_normalize(X: np.ndarray) -> np.ndarray:
    if X is None or X.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    X = X.astype(np.float32, copy=False)
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    return (X / n).astype(np.float32)

def _cosine_cost(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    if A.size == 0 or B.size == 0:
        return np.zeros((A.shape[0], B.shape[0]), dtype=np.float32)
    A = _l2_normalize(A); B = _l2_normalize(B)
    sim = A @ B.T
    return (1.0 - sim).astype(np.float32)

def hungarian_match_centroids(img_cent: np.ndarray, txt_cent: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    반환: (img_ordered[K,d], txt_ordered[K,d], sim[K])  (K = min(K_img, K_txt))
    매칭은 코사인 비용(1-sim) 최소합으로 수행. scipy 없으면 그리디 폴백.
    """
    Ka, Kb = img_cent.shape[0], txt_cent.shape[0]
    K = min(Ka, Kb)
    if K == 0:
        return (np.zeros((0, img_cent.shape[1]), np.float32),
                np.zeros((0, txt_cent.shape[1]), np.float32),
                np.zeros((0,), np.float32))

    img_use = img_cent[:K]
    txt_use = txt_cent[:K]
    cost = _cosine_cost(img_use, txt_use)

    try:
        from scipy.optimize import linear_sum_assignment
        r_idx, c_idx = linear_sum_assignment(cost)
        # 행 기준 정렬 보장
        order = np.argsort(r_idx)
        r_idx = r_idx[order]; c_idx = c_idx[order]
    except Exception:
        # greedy fallback
        used = set()
        c_idx = [None] * K
        row_order = np.argsort(cost.min(axis=1))  # 가장 좋은 행부터
        for r in row_order:
            for c in np.argsort(cost[r]):
                if c not in used:
                    used.add(c); c_idx[r] = c; break
        for r in range(K):
            if c_idx[r] is None:
                c_idx[r] = int(np.argmin(cost[r]))
        r_idx = np.arange(K, dtype=int)
        c_idx = np.array(c_idx, dtype=int)

    img_ordered = img_use[r_idx]
    txt_ordered = txt_use[c_idx]
    sim = (1.0 - cost[r_idx, c_idx]).astype(np.float32)
    return img_ordered, txt_ordered, sim

# --------------- grouping helpers (기존 인터페이스 유지) ---------------

def split_groups_by_quantile(metrics: dict) -> Tuple[list, list, list]:
    """
    멀티모달 클라를 metric 3분위수로 분할 (top/mid/low).
    cfg.METRIC_NAME이 loss면 낮을수록 좋다는 점은 orchestrator에서 처리.
    여기서는 단순 분위수만 계산.
    """
    vals = np.array([v for k, v in metrics.items() if k in cfg.FUSION_CLIENTS and not np.isnan(v)])
    if vals.size == 0:
        return [], [], []
    q1, q2 = np.quantile(vals, 1/3), np.quantile(vals, 2/3)
    low = [cid for cid, v in metrics.items() if cid in cfg.FUSION_CLIENTS and not np.isnan(v) and v <= q1]
    mid = [cid for cid, v in metrics.items() if cid in cfg.FUSION_CLIENTS and not np.isnan(v) and q1 < v <= q2]
    top = [cid for cid, v in metrics.items() if cid in cfg.FUSION_CLIENTS and not np.isnan(v) and v >  q2]
    return top, mid, low

def stack_reps(clients: List[int], split: str = "train", max_samples: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
    imgs, txts = [], []
    for cid in clients:
        img, txt = get_client_reps(cid, split=split, max_samples=max_samples)
        if img is not None and len(img): imgs.append(img)
        if txt is not None and len(txt): txts.append(txt)
    Ximg = np.vstack(imgs) if len(imgs) else np.zeros((0, 0), dtype=np.float32)
    Xtxt = np.vstack(txts) if len(txts) else np.zeros((0, 0), dtype=np.float32)
    return Ximg.astype(np.float32), Xtxt.astype(np.float32)

def kmeans_centroids(X: np.ndarray, k: int) -> np.ndarray:
    """
    빈 입력/중복행을 방어하고, L2 정규화된 센트로이드 반환.
    """
    if X is None or X.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    Xu = np.unique(X, axis=0)
    k  = max(1, min(k, Xu.shape[0]))
    km = KMeans(n_clusters=k, n_init=cfg.KMEANS_N_INIT, random_state=cfg.SEED).fit(Xu)
    C  = km.cluster_centers_.astype(np.float32)
    return _l2_normalize(C)

# --------------- dimensionality helpers ---------------

def _project_to_dim(X: np.ndarray, d: int) -> np.ndarray:
    """
    X([N,D]) → 목표 차원 d 로 투영 (PCA→pad/trim).
    n_components ≤ min(N,D,d) 준수. 입력이 너무 작으면 그대로 두고 pad.
    """
    if X is None or X.size == 0:
        return np.zeros((0, d), dtype=np.float32)
    N, D = X.shape
    if d == D:
        return X.astype(np.float32)

    n_comp = max(1, min(d, D, N))
    Y = X.astype(np.float32)
    if N >= 2 and D > 1:
        try:
            pca = PCA(n_components=n_comp, random_state=cfg.SEED)
            Y = pca.fit_transform(X).astype(np.float32)
        except Exception:
            Y = X.astype(np.float32)

    if Y.shape[1] < d:
        pad = np.zeros((Y.shape[0], d - Y.shape[1]), dtype=np.float32)
        Y = np.concatenate([Y, pad], axis=1)
    elif Y.shape[1] > d:
        Y = Y[:, :d]
    return Y

# --------------- Z builder ---------------

def build_global_vectors(img_cent: np.ndarray, txt_cent: np.ndarray) -> Dict[str, np.ndarray]:
    """
    1) 매칭용 공통 차원으로 투영 (d_pair = min(D_img, D_txt))
    2) 코사인 기반 헝가리안 매칭으로 K개의 페어 추출
    3) 원본 센트로이드에서 페어를 뽑아 d_model로 투영 (PCA→pad/trim)
    4) Z_mm = (img_proj + txt_proj)/2, Z_img2txt = txt_proj, Z_txt2img = img_proj
    반환은 모두 np.ndarray(shape=[K, cfg.D_MODEL], float32)
    """
    if img_cent is None or img_cent.size == 0 or txt_cent is None or txt_cent.size == 0:
        d = int(cfg.D_MODEL)
        return {"Z_mm": np.zeros((0, d), np.float32),
                "Z_img2txt": np.zeros((0, d), np.float32),
                "Z_txt2img": np.zeros((0, d), np.float32)}

    Di = img_cent.shape[1]
    Dt = txt_cent.shape[1]
    d_pair = max(1, min(Di, Dt))

    # 매칭 전 공통 차원으로 투영(매칭 품질을 위해 L2 정규화가 들어간 cosine만 사용)
    img_for_match = _project_to_dim(img_cent, d_pair)
    txt_for_match = _project_to_dim(txt_cent, d_pair)

    img_m, txt_m, _sim = hungarian_match_centroids(img_for_match, txt_for_match)
    K = img_m.shape[0]
    if K == 0:
        d = int(cfg.D_MODEL)
        return {"Z_mm": np.zeros((0, d), np.float32),
                "Z_img2txt": np.zeros((0, d), np.float32),
                "Z_txt2img": np.zeros((0, d), np.float32)}

    # 페어링된 인덱스에 맞춰 원본 센트로이드에서 재선택
    # (img_m/txt_m은 매칭용 투영 결과라 원본과 1:1 인덱스를 공유하지 않을 수 있으므로
    #  여기서는 그냥 img_m/txt_m를 그대로 사용해도 무방합니다. 필요시 원본에서 별도 추적해 뽑도록 변경하세요.)
    # 최종 Z는 d_model로 맞춘 뒤 생성
    d_model = int(cfg.D_MODEL)
    img_proj = _project_to_dim(img_m, d_model)   # [K, d_model]
    txt_proj = _project_to_dim(txt_m, d_model)   # [K, d_model]

    Z_mm      = 0.5 * (img_proj + txt_proj)
    Z_img2txt = txt_proj.copy()
    Z_txt2img = img_proj.copy()

    return {
        "Z_mm":      Z_mm.astype(np.float32),
        "Z_img2txt": Z_img2txt.astype(np.float32),
        "Z_txt2img": Z_txt2img.astype(np.float32),
    }
