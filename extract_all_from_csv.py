# -*- coding: utf-8 -*-
"""
extract_all_from_csv.py
- test.csv에 적힌 image_dir의 모든 폴더에 대해
  extract_img_embed.py를 순차 실행해서 *_embed.npy 생성
"""

import os
import pandas as pd
from glob import glob
import subprocess

CSV_PATH = r"C:\HJHJ0808\김희진\연구\졸업프로젝트\mimic-cxr\mimic-cxr\client_splits\test.csv"
IMG_SCRIPT = r"C:\HJHJ0808\김희진\연구\졸업프로젝트\mimic-cxr\mimic-cxr\extract_img_embed.py"

# CSV 읽기
df = pd.read_csv(CSV_PATH)
print(f"[INFO] Loaded {len(df)} rows from {CSV_PATH}")

# GPU 사용 여부 (옵션)
DEVICE = "cpu"  # or "cpu"

for i, row in df.iterrows():
    img_dir = row["image_dir"]
    if not os.path.isdir(img_dir):
        print(f"[WARN] skip row {i} (no dir): {img_dir}")
        continue

    # 폴더 안의 이미지 파일들 찾기 (jpg만)
    jpgs = glob(os.path.join(img_dir, "*.jpg"))
    if not jpgs:
        print(f"[WARN] skip row {i} (no jpg): {img_dir}")
        continue

    # 대표 이미지 선택 (첫 번째)
    img_path = jpgs[0]

    # extract_img_embed.py 실행
    cmd = [
        "python",
        IMG_SCRIPT,
        "--img", img_path,
        "--device", DEVICE
    ]
    print(f"[RUN] {img_path}")
    subprocess.run(cmd)
