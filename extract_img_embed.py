# -*- coding: utf-8 -*-
"""
extract_img_embed.py
- 단일 X-ray 이미지 파일을 MobileNetV3(Small) 기반으로 임베딩 추출
- local_train의 ImageHead와 동일한 구조(출력 576차원)
- 결과: 원본 경로에 '_embed.npy'로 저장
"""

import os
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import argparse

# -----------------------------
# MobileNetV3 기반 Feature Extractor
# -----------------------------
class ImageFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # 576차원으로 flatten
    def forward(self, x):
        h = self.features(x)
        h = self.pool(h).flatten(1)  # [B, 576]
        return h

# -----------------------------
# 이미지 전처리 (local_train.data.img_transform()과 동일)
# -----------------------------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -----------------------------
# 메인 함수
# -----------------------------
def extract_and_save(img_path: str, out_path: str = None, device="cpu"):
    model = ImageFeatureExtractor().to(device)
    model.eval()

    # 이미지 불러오기
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    # 특징 추출
    with torch.no_grad():
        feat = model(x).cpu().numpy().squeeze()  # shape (576,)

    if out_path is None:
        out_path = os.path.splitext(img_path)[0] + "_embed.npy"

    np.save(out_path, feat)
    print(f"[OK] saved embedding → {out_path} | shape={feat.shape}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", type=str, required=True, help="이미지 경로 (.jpg)")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    extract_and_save(args.img, device=args.device)
