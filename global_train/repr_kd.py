# -*- coding: utf-8 -*-
import os
import numpy as np
from sklearn.cluster import KMeans

from global_train.config import cfg
from global_train.utils_io import (
    ensure_dir, client_dir, load_client_metric, get_client_reps
)

ALPHA = 0.3         # 학생 벡터에 교사 프로토타입을 섞는 비율
K_TEACH = 32        # 교사 프로토타입(클러스터 수)

def _higher_is_better(metric_name: str) -> bool:
    # "loss" 들어가면 낮을수록 좋다고 판단, 그 외는 높을수록 좋다고 가정
    name = (metric_name or "").lower()
    return ("loss" not in name)

def _group_clients_by_metric():
    """
    멀티모달(1~16): metric 기준 상위 8 → s1(교사 pool), 하위 8 → s2(학생)
    이미지/텍스트 전용(17~18, 19~20): 각 페어에서 더 좋은 쪽이 교사
    """
    metric_name = cfg.METRIC_NAME
    better_high = _higher_is_better(metric_name)

    # --- 멀티모달 ---
    vals = {cid: load_client_metric(cid) for cid in cfg.FUSION_CLIENTS}
    # NaN은 맨 뒤로 가도록 정렬 키 구성
    def _key(c):
        v = vals[c]
        if np.isnan(v):
            # NaN이면 항상 최후순위
            return (True, 0.0, c)
        # better_high면 내림차순(큰 값 좋음) → -v, 아니면 오름차순 → v
        return (False, (-v if better_high else v), c)
    c_sorted = sorted(cfg.FUSION_CLIENTS, key=_key)
    s1, s2 = c_sorted[:8], c_sorted[8:]

    # --- 이미지 전용 ---
    img1, img2 = cfg.IMAGE_ONLY
    v1, v2 = load_client_metric(img1), load_client_metric(img2)
    if np.isnan(v1) and np.isnan(v2):
        # 둘 다 NaN이면 ID 작은 쪽을 교사로(임시 규칙)
        img_teach, img_stud = (min(img1, img2), max(img1, img2))
    else:
        if better_high:
            img_teach, img_stud = ((img1, img2) if (v1 >= v2 or np.isnan(v2)) else (img2, img1))
        else:
            img_teach, img_stud = ((img1, img2) if (v1 <= v2 or np.isnan(v2)) else (img2, img1))

    # --- 텍스트 전용 ---
    txt1, txt2 = cfg.TEXT_ONLY
    v1, v2 = load_client_metric(txt1), load_client_metric(txt2)
    if np.isnan(v1) and np.isnan(v2):
        txt_teach, txt_stud = (min(txt1, txt2), max(txt1, txt2))
    else:
        if better_high:
            txt_teach, txt_stud = ((txt1, txt2) if (v1 >= v2 or np.isnan(v2)) else (txt2, txt1))
        else:
            txt_teach, txt_stud = ((txt1, txt2) if (v1 <= v2 or np.isnan(v2)) else (txt2, txt1))

    print(f"[KD] metric='{metric_name}' (higher_is_better={better_high})")
    print(f"[KD] MM teacher pool (s1): {s1}")
    print(f"[KD] MM students (s2):     {s2}")
    print(f"[KD] IMG teacher→student:  {img_teach} → {img_stud}")
    print(f"[KD] TXT teacher→student:  {txt_teach} → {txt_stud}")

    return {
        "mm":  {"s1": s1, "s2": s2},
        "img": {"teach": img_teach, "stud": img_stud},
        "txt": {"teach": txt_teach, "stud": txt_stud},
    }

def _kmeans_centers(X, k):
    if X is None or len(X) == 0:
        return np.zeros((0, 1), dtype=np.float32)
    # 중복만 있는 경우 k=1
    k = min(k, max(1, len(np.unique(X, axis=0))))
    return KMeans(n_clusters=k, n_init="auto", random_state=cfg.SEED)\
        .fit(X).cluster_centers_.astype(np.float32)

def _blend_to_prototypes(X, P, alpha=ALPHA):
    """학생 벡터 X를 최근접 프로토타입 P로 살짝 당겨주는 단순 KD(표현 정합)"""
    if X is None or len(X) == 0 or P is None or len(P) == 0:
        return X
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)
    idx = np.argmax(Xn @ Pn.T, axis=1)
    return (1 - alpha) * X + alpha * P[idx]

def _safe_vstack(parts, dim):
    if len(parts) == 0:
        return np.zeros((0, dim), dtype=np.float32)
    return np.vstack(parts)

def main():
    g = _group_clients_by_metric()
    print(g)

    # ===== 멀티모달: s1(교사 pool) 프로토타입 만들기 =====
    X_img_T, X_txt_T = [], []
    for c in g["mm"]["s1"]:
        xi, xt = get_client_reps(c, split="train", max_samples=10_000)
        if xi is not None and len(xi): X_img_T.append(xi)
        if xt is not None and len(xt): X_txt_T.append(xt)

    P_img = _kmeans_centers(_safe_vstack(X_img_T, cfg.IMG_DIM), K_TEACH) if len(X_img_T) else np.zeros((0, cfg.IMG_DIM), np.float32)
    P_txt = _kmeans_centers(_safe_vstack(X_txt_T, cfg.TXT_DIM), K_TEACH) if len(X_txt_T) else np.zeros((0, cfg.TXT_DIM), np.float32)
    # print(f"[DEBUG] P_img={P_img.shape}, P_txt={P_txt.shape}")


    # ===== 멀티모달: s2 학생에 KD 적용 =====
    for c in g["mm"]["s2"]:
        # print(f"[DEBUG] client_{c:02d}: xi={None if xi is None else xi.shape}, xt={None if xt is None else xt.shape}")
        # print(f"[DEBUG] P_img={P_img.shape}, P_txt={P_txt.shape}")
        base = client_dir(c); ensure_dir(base)
        xi, xt = get_client_reps(c, split="train", max_samples=10_000)
        print(xi.shape, xt.shape)

        if xi is not None and len(xi) and len(P_img):
            np.save(os.path.join(base, "repr_img_kd.npy"), _blend_to_prototypes(xi, P_img))
            print(f"[SAVE] client_{c:02d} -> {os.path.join(base, 'repr_img_kd.npy') if xi is not None else 'skip_img'}")

        if xt is not None and len(xt) and len(P_txt):
            np.save(os.path.join(base, "repr_txt_kd.npy"), _blend_to_prototypes(xt, P_txt))
            print(f"[SAVE] client_{c:02d} -> {os.path.join(base, 'repr_txt_kd.npy') if xt is not None else 'skip_txt'}")


    # ===== 이미지 전용: teach → stud =====
    cT, cS = g["img"]["teach"], g["img"]["stud"]
    XiT, _ = get_client_reps(cT, split="train", max_samples=10_000)
    P_i = _kmeans_centers(XiT, K_TEACH) if XiT is not None and len(XiT) else np.zeros((0, cfg.IMG_DIM), np.float32)
    XiS, _ = get_client_reps(cS, split="train", max_samples=10_000)
    if XiS is not None and len(XiS) and len(P_i):
        np.save(os.path.join(client_dir(cS), "repr_img_kd.npy"), _blend_to_prototypes(XiS, P_i))
        print(f"[SAVE] client_{c:02d} -> {os.path.join(base, 'repr_img_kd.npy') if xi is not None else 'skip_img'}")


    # ===== 텍스트 전용: teach → stud =====
    cT, cS = g["txt"]["teach"], g["txt"]["stud"]
    _, XtT = get_client_reps(cT, split="train", max_samples=10_000)
    P_t = _kmeans_centers(XtT, K_TEACH) if XtT is not None and len(XtT) else np.zeros((0, cfg.TXT_DIM), np.float32)
    _, XtS = get_client_reps(cS, split="train", max_samples=10_000)
    if XtS is not None and len(XtS) and len(P_t):
        np.save(os.path.join(client_dir(cS), "repr_txt_kd.npy"), _blend_to_prototypes(XtS, P_t))
        print(f"[SAVE] client_{c:02d} -> {os.path.join(base, 'repr_img_kd.npy') if xi is not None else 'skip_img'}")
    
    print("[KD] representation KD done (repr_*_kd.npy written where applicable)")

if __name__ == "__main__":
    main()
