# repr_kd.py
import os, numpy as np
from sklearn.cluster import KMeans
from global_train.config import cfg
from global_train.utils_io import ensure_dir, client_dir, load_client_metric, save_json, get_client_reps

ALPHA = 0.5
K_TEACH = 32

def _group_clients_by_metric():
    # 멀티모달: f1_macro 기준으로 s1(상위5)/s2(다음5)
    vals = {cid: load_client_metric(cid) for cid in cfg.FUSION_CLIENTS}
    c_sorted = sorted(vals, key=lambda c: (-(vals[c] if not np.isnan(vals[c]) else -1e9), c))
    s1, s2 = c_sorted[:8], c_sorted[8:]  # 8명씩 
    # 이미지/텍스트 전용: 높은 f1=교사, 낮은 f1=학생
    img1, img2 = cfg.IMAGE_ONLY
    txt1, txt2 = cfg.TEXT_ONLY
    img_teach, img_stud = (img1, img2) if (load_client_metric(img1) >= load_client_metric(img2)) else (img2, img1)
    txt_teach, txt_stud = (txt1, txt2) if (load_client_metric(txt1) >= load_client_metric(txt2)) else (txt2, txt1)
    return {"mm": {"s1": s1, "s2": s2},
            "img": {"teach": img_teach, "stud": img_stud},
            "txt": {"teach": txt_teach, "stud": txt_stud}}

def _kmeans_centers(X, k):
    if len(X) == 0: return np.zeros((0, X.shape[1]), dtype=np.float32)
    k = min(k, max(1, len(np.unique(X, axis=0))))
    return KMeans(n_clusters=k, n_init="auto", random_state=cfg.SEED).fit(X).cluster_centers_.astype(np.float32)

def _blend_to_prototypes(X, P, alpha=ALPHA):
    if len(X) == 0 or len(P) == 0: return X
    # 최근접 프로토타입
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True)+1e-8)
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True)+1e-8)
    idx = np.argmax(Xn @ Pn.T, axis=1)
    return (1-alpha)*X + alpha*P[idx]

def main():
    g = _group_clients_by_metric()

    # ----- 멀티모달: s1을 교사 풀로, s2를 학생으로 KD -----
    # 교사 풀 프로토타입(이미지/텍스트)
    X_img_T, X_txt_T = [], []
    for c in g["mm"]["s1"]:
        xi, xt = get_client_reps(c, split="train", max_samples=10_000)
        if len(xi): X_img_T.append(xi)
        if len(xt): X_txt_T.append(xt)
    P_img = _kmeans_centers(np.vstack(X_img_T) if X_img_T else np.zeros((0, cfg.IMG_DIM), np.float32), K_TEACH)
    P_txt = _kmeans_centers(np.vstack(X_txt_T) if X_txt_T else np.zeros((0, cfg.TXT_DIM), np.float32), K_TEACH)

    # 학생(s3)과 s2(옵션: 유지) 처리
    for c in g["mm"]["s2"]:
        base = client_dir(c); os.makedirs(base, exist_ok=True)
        xi, xt = get_client_reps(c, split="train", max_samples=10_000)
        if len(xi): np.save(os.path.join(base, "repr_img_kd.npy"), _blend_to_prototypes(xi, P_img))
        if len(xt): np.save(os.path.join(base, "repr_txt_kd.npy"), _blend_to_prototypes(xt, P_txt))

    # ----- 이미지 전용: teach→stud -----
    cT, cS = g["img"]["teach"], g["img"]["stud"]
    XiT, _ = get_client_reps(cT, split="train", max_samples=10_000)
    P_i = _kmeans_centers(XiT, K_TEACH)
    XiS, _ = get_client_reps(cS, split="train", max_samples=10_000)
    if len(XiS): np.save(os.path.join(client_dir(cS), "repr_img_kd.npy"), _blend_to_prototypes(XiS, P_i))

    # ----- 텍스트 전용: teach→stud -----
    cT, cS = g["txt"]["teach"], g["txt"]["stud"]
    _, XtT = get_client_reps(cT, split="train", max_samples=10_000)
    P_t = _kmeans_centers(XtT, K_TEACH)
    _, XtS = get_client_reps(cS, split="train", max_samples=10_000)
    if len(XtS): np.save(os.path.join(client_dir(cS), "repr_txt_kd.npy"), _blend_to_prototypes(XtS, P_t))

    print("[KD] representation KD done (repr_*_kd.npy written where applicable)")

if __name__ == "__main__":
    main()