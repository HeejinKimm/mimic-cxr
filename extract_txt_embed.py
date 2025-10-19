# -*- coding: utf-8 -*-
"""
extract_txt_embed.py
- 단일 리포트(.txt)를 BERT-mini 기반 256차 임베딩으로 변환
- local_train.config의 설정(Cfg.MAX_LEN=256, TEXT_MODEL_NAME=prajjwal1/bert-mini)에 맞춤
- 결과: 원본 경로에 '_embed.npy'로 저장 (예: ...\s53807891.txt -> ...\s53807891_embed.npy)
"""

import os
import argparse
import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel

def mean_pool(last_hidden_state, attention_mask):
    # last_hidden_state: [B, T, H], attention_mask: [B, T]
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # [B, T, 1]
    summed = (last_hidden_state * mask).sum(dim=1)                  # [B, H]
    counts = mask.sum(dim=1).clamp(min=1e-6)                        # [B, 1]
    return summed / counts

class TextFeatureExtractor(nn.Module):
    def __init__(self, model_name="prajjwal1/bert-mini"):
        super().__init__()
        self.base = AutoModel.from_pretrained(model_name)  # hidden_size=256
        # local 모델과 동일 차원 유지(256) → 투영층은 아이덴티티에 해당
        self.proj = nn.Identity()

    @torch.no_grad()
    def forward(self, input_ids, attention_mask):
        out = self.base(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled = mean_pool(out.last_hidden_state, attention_mask)   # [B, 256]
        z = self.proj(pooled)                                       # [B, 256]
        return z

def read_text(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        t = f.read()
    # 너무 긴 리포트는 토크나이저에서 truncation 처리
    return t

def extract_and_save(txt_path: str, out_path: str = None, model_name="prajjwal1/bert-mini",
                     max_len: int = 256, device: str = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TextFeatureExtractor(model_name=model_name).to(device).eval()

    text = read_text(txt_path)
    enc = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        feat = model(input_ids=input_ids, attention_mask=attention_mask).cpu().numpy().squeeze()  # (256,)

    if out_path is None:
        root, _ = os.path.splitext(txt_path)
        out_path = root + "_embed.npy"

    np.save(out_path, feat)
    print(f"[OK] saved embedding → {out_path} | shape={feat.shape}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--txt", type=str, required=True, help="리포트 파일 경로 (.txt)")
    ap.add_argument("--model_name", type=str, default="prajjwal1/bert-mini")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--device", type=str, default=None, help="cuda / cpu (기본 자동)")
    args = ap.parse_args()

    extract_and_save(
        txt_path=args.txt,
        model_name=args.model_name,
        max_len=args.max_len,
        device=args.device
    )
