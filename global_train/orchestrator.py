# -*- coding: utf-8 -*-
"""
orchestrator.py (KD-free version)
- repr_kd.py 실행 이후 생성된 KD 임베딩(repr_*_kd.npy)을 우선 사용.
- 없으면 기존 train_*_reps.npy fallback.
- 모든 fusion clients의 임베딩을 모아 글로벌 Z 계산 및 저장.
"""

import os
import numpy as np
from .config import cfg
from .utils_io import (
    set_seed, ensure_dir, save_json, save_payload_for_client, client_dir
)
from .build_vectors import (
    kmeans_centroids, build_global_vectors
)

# ======================================================
# KD-aware 임베딩 로더
# ======================================================
def get_client_reps_kd_first(cid: int, split="train", max_samples=10_000):
    base = client_dir(cid)
    img_path_kd = os.path.join(base, "repr_img_kd.npy")
    txt_path_kd = os.path.join(base, "repr_txt_kd.npy")
    img_path_raw = os.path.join(base, f"{split}_img_reps.npy")
    txt_path_raw = os.path.join(base, f"{split}_txt_reps.npy")

    def _safe_load(p):
        if os.path.exists(p):
            arr = np.load(p)
            if len(arr) > max_samples:
                idx = np.random.default_rng(cfg.SEED).choice(len(arr), max_samples, replace=False)
                arr = arr[idx]
            return arr.astype(np.float32)
        return np.zeros((0, 0), dtype=np.float32)

    img = _safe_load(img_path_kd if os.path.exists(img_path_kd) else img_path_raw)
    txt = _safe_load(txt_path_kd if os.path.exists(txt_path_kd) else txt_path_raw)

    if os.path.exists(img_path_kd) or os.path.exists(txt_path_kd):
        print(f"[LOAD] client_{cid:02d}: using KD reprs "
              f"({'img' if os.path.exists(img_path_kd) else ''} "
              f"{'txt' if os.path.exists(txt_path_kd) else ''})")
    else:
        print(f"[LOAD] client_{cid:02d}: using raw train_*_reps")

    return img, txt


# ======================================================
# MAIN
# ======================================================
def main():
    set_seed(cfg.SEED)
    ensure_dir(cfg.OUT_GLOBAL_DIR)

    print(f"[GLOBAL] stacking KD (or raw) reps from fusion clients: {list(cfg.FUSION_CLIENTS)}")

    imgs, txts = [], []
    for cid in cfg.FUSION_CLIENTS:
        xi, xt = get_client_reps_kd_first(cid, split="train", max_samples=cfg.SAMPLE_PER_CLIENT)
        if len(xi): imgs.append(xi)
        if len(xt): txts.append(xt)

    img_all = np.vstack(imgs) if imgs else np.zeros((0, cfg.IMG_DIM), np.float32)
    txt_all = np.vstack(txts) if txts else np.zeros((0, cfg.TXT_DIM), np.float32)
    print(f"[GLOBAL] stacked img reps: {img_all.shape}, txt reps: {txt_all.shape}")

    if img_all.size == 0 or txt_all.size == 0:
        print("[WARN] empty reps detected. Distributing zero global vectors.")
        Zs = {
            "Z_mm": np.zeros((cfg.D_MODEL,), np.float32),
            "Z_img2txt": np.zeros((cfg.D_MODEL,), np.float32),
            "Z_txt2img": np.zeros((cfg.D_MODEL,), np.float32),
        }
    else:
        img_cent = kmeans_centroids(img_all, cfg.K_IMG)
        txt_cent = kmeans_centroids(txt_all, cfg.K_TXT)
        print(f"[GLOBAL] centroids -> img: {img_cent.shape}, txt: {txt_cent.shape}")
        Zs = build_global_vectors(img_cent, txt_cent)

    # --- Z 결과 저장 ---
    z_path = os.path.join(cfg.OUT_GLOBAL_DIR, "global_Z_vectors.npz")
    np.savez(z_path, **Zs)
    print(f"[SAVE] Global Z vectors saved -> {z_path}")

    # --- 각 클라이언트별 payload 저장 (Z만 전달) ---
    for cid in cfg.FUSION_CLIENTS:
        payload = {
            "client_id": cid,
            "type": "fusion",
            "Z": Zs["Z_mm"],
            "d_global": cfg.D_MODEL,
            "note": "KD-based global representation",
        }
        save_payload_for_client(cid, payload)

    print("[GLOBAL] Done ->", cfg.OUT_GLOBAL_DIR)

if __name__ == "__main__":
    main()
