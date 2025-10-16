# -*- coding: utf-8 -*-
"""
visualize_global_Z.py
- global_output/global_Z_vectors.npz 파일을 불러와
  Z_mm, Z_img2txt, Z_txt2img 임베딩을 2D로 시각화합니다.
- PCA → t-SNE 순으로 차원 축소 후, matplotlib으로 산점도 출력.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# -------------------------------------------------------------
# 기본 경로 (필요시 수정)
# -------------------------------------------------------------
BASE_DIR = r"C:\HJHJ0808\김희진\연구\졸업프로젝트\mimic-cxr\mimic-cxr"
Z_PATH = os.path.join(BASE_DIR, "global_output", "global_Z_vectors.npz")

# -------------------------------------------------------------
# Z 로드
# -------------------------------------------------------------
if not os.path.exists(Z_PATH):
    raise FileNotFoundError(f"파일을 찾을 수 없습니다: {Z_PATH}")

data = np.load(Z_PATH)
Z_mm = data["Z_mm"]
Z_img2txt = data["Z_img2txt"]
Z_txt2img = data["Z_txt2img"]

# ✅ shape 보정 (1D → 2D)
for name, Z in [("Z_mm", Z_mm), ("Z_img2txt", Z_img2txt), ("Z_txt2img", Z_txt2img)]:
    if Z.ndim == 1:
        Z = Z.reshape(1, -1)
        locals()[name] = Z  # 동적 변수 교체
        print(f"[FIX] {name} reshaped to {Z.shape}")

print(f"[LOAD] Z_mm={Z_mm.shape}, Z_img2txt={Z_img2txt.shape}, Z_txt2img={Z_txt2img.shape}")

# -------------------------------------------------------------
# 차원 축소 (PCA → t-SNE)
# -------------------------------------------------------------
def reduce_dim(Z, n_pca=30, n_tsne=2, seed=42):
    if Z.shape[0] > 5000:
        idx = np.random.default_rng(seed).choice(Z.shape[0], 5000, replace=False)
        Z = Z[idx]
    # 1차: PCA
    n_comp = min(n_pca, Z.shape[1], Z.shape[0])
    Z_pca = PCA(n_components=n_comp, random_state=seed).fit_transform(Z)
    # 2차: t-SNE
    Z_2d = TSNE(n_components=n_tsne, perplexity=30, random_state=seed, n_iter=1000).fit_transform(Z_pca)
    return Z_2d

print("[REDUCE] Dimensionality reduction (PCA → t-SNE) 진행 중...")
Z_mm_2d = reduce_dim(Z_mm)
Z_i2t_2d = reduce_dim(Z_img2txt)
Z_t2i_2d = reduce_dim(Z_txt2img)

# -------------------------------------------------------------
# 시각화
# -------------------------------------------------------------
plt.figure(figsize=(10, 8))
plt.scatter(Z_mm_2d[:, 0], Z_mm_2d[:, 1], c="blue", s=15, alpha=0.6, label="Z_mm (mean fused)")
plt.scatter(Z_i2t_2d[:, 0], Z_i2t_2d[:, 1], c="green", s=15, alpha=0.6, label="Z_img2txt")
plt.scatter(Z_t2i_2d[:, 0], Z_t2i_2d[:, 1], c="red", s=15, alpha=0.6, label="Z_txt2img")
plt.title("Global Z Embeddings (PCA + t-SNE)", fontsize=14)
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

SAVE_PATH = os.path.join(BASE_DIR, "global_output", "Z_embedding_tsne.png")
plt.savefig(SAVE_PATH, dpi=300)
plt.show()

print(f"[SAVE] 시각화 완료 → {SAVE_PATH}")
