# -*- coding: utf-8 -*-
"""
cluster_and_group_z.py (balanced client-level grouping + centroid dumps)
- IMG: 클라이언트 {1..16, 17,18} 포함 (텍스트만 보유한 19,20은 제외)
- TXT: 클라이언트 {1..16, 19,20} 포함 (이미지만 보유한 17,18은 제외)
- 각 모달리티에서 '클라이언트 평균 임베딩'으로 K=4 균형 그룹(예: 18명 -> 4,4,5,5)으로 배정
- 그룹별 Z 산출 시 사용한 '서브센트로이드'를 .npy로 저장하고, JSON 인덱스에 정리
- 산출물: global_output/clustering/
    - global_img_centroids.npy / global_txt_centroids.npy
    - global_centroids.json
    - img_client_assignments.csv/json, txt_client_assignments.csv/json
    - img_clientgroup_{g}_centroids.npy, txt_clientgroup_{g}_centroids.npy
    - img_clientgroup_{g}_Z.npz, txt_clientgroup_{g}_Z.npz
    - img_clientgroup_centroids.json, txt_clientgroup_centroids.json
    - groups_overview.csv
"""

import os, json, csv
import argparse
import numpy as np

from .config import cfg
from .utils_io import set_seed, ensure_dir
from .orchestrator import get_client_reps_kd_first
from .build_vectors import kmeans_centroids, build_global_vectors


# ---------------------------
# 유틸
# ---------------------------
def _save_json(path: str, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _append_csv(path: str, rows: list, header: list):
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerows(rows)

def _balance_sizes(n: int, k: int):
    """n명을 k그룹으로 균등 배분 (뒤 그룹부터 +1)  예: 18,4 -> [4,4,5,5]"""
    base = n // k
    r = n % k
    sizes = [base] * k
    for i in range(r):
        sizes[k - r + i] += 1
    return sizes

def _kmeans_plus_plus_init(X: np.ndarray, k: int, rng: np.random.Generator):
    N, D = X.shape
    centroids = np.empty((k, D), dtype=np.float32)
    idx0 = rng.integers(0, N)
    centroids[0] = X[idx0]
    dist2 = np.sum((X - centroids[0])**2, axis=1)
    for c in range(1, k):
        probs = dist2 / (dist2.sum() + 1e-12)
        idx = rng.choice(N, p=probs)
        centroids[c] = X[idx]
        dnew = np.sum((X - centroids[c])**2, axis=1)
        dist2 = np.minimum(dist2, dnew)
    return centroids

def _balanced_assign(X: np.ndarray, k: int, sizes: list, rng: np.random.Generator, iters: int = 2):
    """용량 제약 k-means 배정(소규모 N에 충분): k-means++ init → 용량 고려 그리디 배정 → 중심 갱신 반복"""
    N, D = X.shape
    sizes = list(map(int, sizes))
    assert sum(sizes) == N, f"sum(sizes) must be N (got {sum(sizes)} vs {N})"
    C = _kmeans_plus_plus_init(X, k, rng)

    def greedy_assign(X, C, sizes):
        x2 = np.sum(X * X, axis=1, keepdims=True)
        c2 = np.sum(C * C, axis=1, keepdims=True).T
        dist2 = x2 + c2 - 2.0 * (X @ C.T)  # (N,K)
        pairs = np.dstack(np.meshgrid(np.arange(N), np.arange(k), indexing="ij")).reshape(-1, 2)
        dvals = dist2[pairs[:,0], pairs[:,1]]
        order = np.argsort(dvals)
        cap = sizes[:]
        assigned = -np.ones((N,), dtype=np.int64)
        taken = np.zeros((k,), dtype=np.int64)
        for idx in order:
            i, j = pairs[idx]
            if assigned[i] != -1:
                continue
            if taken[j] < cap[j]:
                assigned[i] = j
                taken[j] += 1
                if taken.sum() == N:
                    break
        assert (assigned != -1).all(), "balanced assign failed (some unassigned)"
        return assigned

    labels = greedy_assign(X, C, sizes)
    for _ in range(iters):
        for j in range(k):
            idx = np.where(labels == j)[0]
            if len(idx) > 0:
                C[j] = X[idx].mean(axis=0, dtype=np.float64).astype(np.float32)
        labels = greedy_assign(X, C, sizes)
    return labels, C


# ---------------------------
# 데이터 적재(클라이언트 평균/샘플)
# ---------------------------
def _collect_clients_for_img():
    """ IMG 모달리티를 가진 클라이언트: 1..16 + 17,18 """
    base = list(cfg.FUSION_CLIENTS)
    extra = list(getattr(cfg, "IMAGE_ONLY", []))
    return base + extra

def _collect_clients_for_txt():
    """ TXT 모달리티를 가진 클라이언트: 1..16 + 19,20 """
    base = list(cfg.FUSION_CLIENTS)
    extra = list(getattr(cfg, "TEXT_ONLY", []))
    return base + extra

def _load_client_means_and_samples(cids: list, sample_per_client: int, use_img: bool, use_txt: bool):
    img_means, txt_means = [], []
    img_by_c, txt_by_c = {}, {}
    for cid in cids:
        xi, xt = get_client_reps_kd_first(int(cid), split="train", max_samples=sample_per_client)
        if use_img:
            if len(xi):
                xi = xi.astype(np.float32)
                img_by_c[cid] = xi
                img_means.append(xi.mean(axis=0, dtype=np.float64).astype(np.float32))
            else:
                img_by_c[cid] = np.zeros((0, cfg.IMG_DIM), np.float32)
                img_means.append(np.zeros((cfg.IMG_DIM,), np.float32))
        if use_txt:
            if len(xt):
                xt = xt.astype(np.float32)
                txt_by_c[cid] = xt
                txt_means.append(xt.mean(axis=0, dtype=np.float64).astype(np.float32))
            else:
                txt_by_c[cid] = np.zeros((0, cfg.TXT_DIM), np.float32)
                txt_means.append(np.zeros((cfg.TXT_DIM,), np.float32))
    img_means = np.stack(img_means, axis=0) if use_img else np.zeros((0, cfg.IMG_DIM), np.float32)
    txt_means = np.stack(txt_means, axis=0) if use_txt else np.zeros((0, cfg.TXT_DIM), np.float32)
    return img_means, txt_means, img_by_c, txt_by_c


# ---------------------------
# 메인 로직
# ---------------------------
def run(k_groups: int, k_img_per_group: int, k_txt_per_group: int):
    set_seed(cfg.SEED)
    ensure_dir(cfg.OUT_GLOBAL_DIR)
    out_dir = os.path.join(cfg.OUT_GLOBAL_DIR, "clustering")
    ensure_dir(out_dir)
    rng = np.random.default_rng(cfg.SEED)

    # 1) 글로벌 콘텍스트(해당 모달을 가진 모든 클라의 '모든 샘플')
    img_cids_all = _collect_clients_for_img()   # 18명: 1..16,17,18
    txt_cids_all = _collect_clients_for_txt()   # 18명: 1..16,19,20

    img_samples_all = []
    for cid in img_cids_all:
        xi, _ = get_client_reps_kd_first(cid, split="train", max_samples=cfg.SAMPLE_PER_CLIENT)
        if len(xi): img_samples_all.append(xi.astype(np.float32))
    img_all = np.vstack(img_samples_all) if img_samples_all else np.zeros((0, cfg.IMG_DIM), np.float32)

    txt_samples_all = []
    for cid in txt_cids_all:
        _, xt = get_client_reps_kd_first(cid, split="train", max_samples=cfg.SAMPLE_PER_CLIENT)
        if len(xt): txt_samples_all.append(xt.astype(np.float32))
    txt_all = np.vstack(txt_samples_all) if txt_samples_all else np.zeros((0, cfg.TXT_DIM), np.float32)

    img_cent_global = kmeans_centroids(img_all, cfg.K_IMG) if img_all.size else np.zeros((0, cfg.IMG_DIM), np.float32)
    txt_cent_global = kmeans_centroids(txt_all, cfg.K_TXT) if txt_all.size else np.zeros((0, cfg.TXT_DIM), np.float32)

    # 글로벌 센트로이드 저장(+ 인덱스 JSON)
    np.save(os.path.join(out_dir, "global_img_centroids.npy"), img_cent_global)
    np.save(os.path.join(out_dir, "global_txt_centroids.npy"), txt_cent_global)
    _save_json(os.path.join(out_dir, "global_centroids.json"), {
        "img": {"file": "global_img_centroids.npy", "shape": list(map(int, img_cent_global.shape))},
        "txt": {"file": "global_txt_centroids.npy", "shape": list(map(int, txt_cent_global.shape))}
    })

    print(f"[GLOBAL] IMG clients: {img_cids_all} (N={len(img_cids_all)}), TXT clients: {txt_cids_all} (N={len(txt_cids_all)})")
    print(f"[GLOBAL] global centroids → IMG {img_cent_global.shape}, TXT {txt_cent_global.shape}")

    # 2) IMG 모달리티: 클라이언트 평균 임베딩 → 균형 4그룹
    img_means, _, img_by_c, _ = _load_client_means_and_samples(img_cids_all, cfg.SAMPLE_PER_CLIENT, use_img=True, use_txt=False)
    sizes_img = _balance_sizes(len(img_cids_all), k_groups)   # e.g., [4,4,5,5]
    labels_img, cents_img = _balanced_assign(img_means, k_groups, sizes_img, rng, iters=2)

    img_assign_json = {"k_groups": int(k_groups), "sizes": sizes_img, "assignments": []}
    img_assign_rows = []
    for idx, cid in enumerate(img_cids_all):
        g = int(labels_img[idx])
        img_assign_json["assignments"].append({"client_id": int(cid), "assigned_group": g})
        img_assign_rows.append([int(cid), g])
    _save_json(os.path.join(out_dir, "img_client_assignments.json"), img_assign_json)
    _append_csv(os.path.join(out_dir, "img_client_assignments.csv"), img_assign_rows, header=["client_id", "assigned_group"])

    # 3) TXT 모달리티: 클라이언트 평균 임베딩 → 균형 4그룹
    _, txt_means, _, txt_by_c = _load_client_means_and_samples(txt_cids_all, cfg.SAMPLE_PER_CLIENT, use_img=False, use_txt=True)
    sizes_txt = _balance_sizes(len(txt_cids_all), k_groups)   # e.g., [4,4,5,5]
    labels_txt, cents_txt = _balanced_assign(txt_means, k_groups, sizes_txt, rng, iters=2)

    txt_assign_json = {"k_groups": int(k_groups), "sizes": sizes_txt, "assignments": []}
    txt_assign_rows = []
    for idx, cid in enumerate(txt_cids_all):
        g = int(labels_txt[idx])
        txt_assign_json["assignments"].append({"client_id": int(cid), "assigned_group": g})
        txt_assign_rows.append([int(cid), g])
    _save_json(os.path.join(out_dir, "txt_client_assignments.json"), txt_assign_json)
    _append_csv(os.path.join(out_dir, "txt_client_assignments.csv"), txt_assign_rows, header=["client_id", "assigned_group"])

    # 4) 그룹별 Z + 센트로이드 덤프 인덱스
    rows_overview = []
    img_centroids_index = {"k_groups": int(k_groups), "groups": []}
    txt_centroids_index = {"k_groups": int(k_groups), "groups": []}

    # IMG 쪽
    for g in range(k_groups):
        member_cids = [cid for i, cid in enumerate(img_cids_all) if int(labels_img[i]) == g]
        Xg_list = [img_by_c[cid] for cid in member_cids if img_by_c[cid].size]
        Xg = np.vstack(Xg_list) if Xg_list else np.zeros((0, cfg.IMG_DIM), np.float32)

        if Xg.size:
            kg = min(max(1, k_img_per_group), Xg.shape[0])
            img_cents_g = kmeans_centroids(Xg, kg)
            Zs = build_global_vectors(img_cents_g, txt_cent_global)
        else:
            img_cents_g = np.zeros((0, cfg.IMG_DIM), np.float32)
            Zs = {"Z_mm": np.zeros((cfg.D_MODEL,), np.float32),
                  "Z_img2txt": np.zeros((cfg.D_MODEL,), np.float32),
                  "Z_txt2img": np.zeros((cfg.D_MODEL,), np.float32)}

        # 파일 저장
        cent_file = f"img_clientgroup_{g}_centroids.npy"
        np.save(os.path.join(out_dir, cent_file), img_cents_g)
        zfile = f"img_clientgroup_{g}_Z.npz"
        np.savez(os.path.join(out_dir, zfile), **Zs)

        # 인덱스 JSON에 기록
        img_centroids_index["groups"].append({
            "group_id": g,
            "n_clients": len(member_cids),
            "members": list(map(int, member_cids)),
            "n_samples": int(Xg.shape[0]),
            "subset_centroids_file": cent_file,
            "subset_centroids_shape": list(map(int, img_cents_g.shape)),
            "context_centroids_file": "global_txt_centroids.npy",
            "context_centroids_shape": list(map(int, txt_cent_global.shape)),
            "z_file": zfile
        })

        rows_overview.append(["img_clientgroup", g, len(member_cids), zfile])

    # TXT 쪽
    for g in range(k_groups):
        member_cids = [cid for i, cid in enumerate(txt_cids_all) if int(labels_txt[i]) == g]
        Xg_list = [txt_by_c[cid] for cid in member_cids if txt_by_c[cid].size]
        Xg = np.vstack(Xg_list) if Xg_list else np.zeros((0, cfg.TXT_DIM), np.float32)

        if Xg.size:
            kg = min(max(1, k_txt_per_group), Xg.shape[0])
            txt_cents_g = kmeans_centroids(Xg, kg)
            Zs = build_global_vectors(img_cent_global, txt_cents_g)
        else:
            txt_cents_g = np.zeros((0, cfg.TXT_DIM), np.float32)
            Zs = {"Z_mm": np.zeros((cfg.D_MODEL,), np.float32),
                  "Z_img2txt": np.zeros((cfg.D_MODEL,), np.float32),
                  "Z_txt2img": np.zeros((cfg.D_MODEL,), np.float32)}

        cent_file = f"txt_clientgroup_{g}_centroids.npy"
        np.save(os.path.join(out_dir, cent_file), txt_cents_g)
        zfile = f"txt_clientgroup_{g}_Z.npz"
        np.savez(os.path.join(out_dir, zfile), **Zs)

        txt_centroids_index["groups"].append({
            "group_id": g,
            "n_clients": len(member_cids),
            "members": list(map(int, member_cids)),
            "n_samples": int(Xg.shape[0]),
            "subset_centroids_file": cent_file,
            "subset_centroids_shape": list(map(int, txt_cents_g.shape)),
            "context_centroids_file": "global_img_centroids.npy",
            "context_centroids_shape": list(map(int, img_cent_global.shape)),
            "z_file": zfile
        })

        rows_overview.append(["txt_clientgroup", g, len(member_cids), zfile])

    # 인덱스 JSON 저장
    _save_json(os.path.join(out_dir, "img_clientgroup_centroids.json"), img_centroids_index)
    _save_json(os.path.join(out_dir, "txt_clientgroup_centroids.json"), txt_centroids_index)

    # 요약 CSV
    _append_csv(os.path.join(out_dir, "groups_overview.csv"), rows_overview,
                header=["unit", "group_id", "n_clients", "z_file"])

    print(f"[CLUST] done -> {out_dir}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", type=int, default=4, help="그룹 개수(K). 4로 고정 권장")
    ap.add_argument("--k_img_per_group", type=int, default=None,
                    help="IMG 그룹 내부 서브센트로이드 개수 (기본=cfg.K_IMG)")
    ap.add_argument("--k_txt_per_group", type=int, default=None,
                    help="TXT 그룹 내부 서브센트로이드 개수 (기본=cfg.K_TXT)")
    return ap.parse_args()

def main():
    args = parse_args()
    k_img = args.k_img_per_group if args.k_img_per_group is not None else int(cfg.K_IMG)
    k_txt = args.k_txt_per_group if args.k_txt_per_group is not None else int(cfg.K_TXT)
    run(k_groups=int(args.groups), k_img_per_group=int(k_img), k_txt_per_group=int(k_txt))

if __name__ == "__main__":
    main()
