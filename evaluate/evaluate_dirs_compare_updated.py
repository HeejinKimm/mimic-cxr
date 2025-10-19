# -*- coding: utf-8 -*-
"""
evaluate_testcsv_compare_updated.py
- test.csv를 직접 불러서 테스트셋을 구성
- 각 클라이언트의 baseline(best.pt) vs updated_heads(npz)를 비교 평가
- MultiModalLateFusion 구조 기반, late fusion + 모달 마스크 사용
"""

import os, csv, json, argparse
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from PIL import Image, ImageFile

# --------------------------
# 설정
# --------------------------
ImageFile.LOAD_TRUNCATED_IMAGES = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
TEXT_MODEL_NAME = "prajjwal1/bert-mini"

LABEL_COLUMNS = [
    "Atelectasis","Cardiomegaly","Consolidation","Edema","Pleural Effusion",
    "Enlarged Cardiomediastinum","Fracture","Lung Lesion","Lung Opacity",
    "Pleural Other","Pneumonia","Pneumothorax","Support Devices"
]

OUT_DIR = Path("./eval_results"); OUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_CSV = OUT_DIR / "summary_testcsv_compare_updated.csv"

IMG_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

# --------------------------
# 유틸
# --------------------------
def cdir(cid: int) -> Path:
    p = Path(f"./outputs/client_{cid:02d}")
    p.mkdir(parents=True, exist_ok=True)
    return p

def safe_per_class_auc(y_true: np.ndarray, y_prob: np.ndarray):
    C = y_true.shape[1]
    aurocs = []
    for j in range(C):
        col = y_true[:, j]
        if len(np.unique(col)) < 2:
            aurocs.append(np.nan)
            continue
        aurocs.append(roc_auc_score(col, y_prob[:, j]))
    return np.array(aurocs, dtype=float)

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    y_pred = (y_prob >= threshold).astype(int)
    f1_micro = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    per_class_auc = safe_per_class_auc(y_true, y_prob)
    macro_auc_valid = float(np.nanmean(per_class_auc))
    micro_auc = float(roc_auc_score(y_true.ravel(), y_prob.ravel())) if len(np.unique(y_true))>1 else float("nan")
    ap_micro = float(average_precision_score(y_true.ravel(), y_prob.ravel())) if len(np.unique(y_true))>1 else float("nan")
    return {
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "macro_auroc_valid": macro_auc_valid,
        "micro_auroc": micro_auc,
        "micro_ap": ap_micro,
        "per_class_auroc": per_class_auc.tolist()
    }

# --------------------------
# 데이터셋 (test.csv 기반)
# --------------------------
class TestCSVDataset(Dataset):
    def __init__(self, csv_path, label_csv):
        df = pd.read_csv(csv_path)
        self.rows = df.to_dict("records")

        # 라벨 테이블 불러오기
        ldf = pd.read_csv(label_csv)
        for c in LABEL_COLUMNS:
            ldf[c] = ldf[c].fillna(0)
            ldf[c] = (ldf[c] >= 1).astype(int)
        self.label_table = {(int(r["subject_id"]), int(r["study_id"])): [int(r[c]) for c in LABEL_COLUMNS]
                            for _, r in ldf.iterrows()}

        self.tok = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        sid, stid = int(str(r["subject_id"])[1:]), int(str(r["study_id"])[1:])
        y = self.label_table.get((sid, stid), [0]*len(LABEL_COLUMNS))
        y = torch.tensor(y, dtype=torch.float32)

        # 이미지 로드
        imgs = list(Path(r["image_dir"]).glob("*.jpg"))
        if imgs:
            try:
                img = IMG_TRANSFORM(Image.open(imgs[0]).convert("RGB"))
                has_img = 1.0
            except:
                img = torch.zeros(3,224,224)
                has_img = 0.0
        else:
            img = torch.zeros(3,224,224)
            has_img = 0.0

        # 텍스트 로드
        txt_path = Path(r["text_path"])
        if txt_path.exists():
            try:
                text = txt_path.read_text(encoding="utf-8")
            except:
                text = ""
        else:
            text = ""
        enc = self.tok(text, truncation=True, padding="max_length", max_length=256, return_tensors="pt")
        has_txt = 0.0 if text == "" else 1.0

        return {
            "labels": y,
            "image": img,
            "text": {k: v.squeeze(0) for k, v in enc.items()},
            "has_img": torch.tensor(has_img, dtype=torch.float32),
            "has_txt": torch.tensor(has_txt, dtype=torch.float32),
        }

# --------------------------
# 모델 구조
# --------------------------
class ImageHead(nn.Module):
    def __init__(self, n_out):
        super().__init__()
        base = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.head = nn.Linear(576, n_out)
    def forward(self, x):
        h = self.pool(self.features(x)).flatten(1)
        return self.head(h)

class TextHead(nn.Module):
    def __init__(self, n_out):
        super().__init__()
        self.enc = AutoModel.from_pretrained(TEXT_MODEL_NAME)
        hid = self.enc.config.hidden_size
        self.cls_head = nn.Linear(hid, n_out)
    def forward(self, **x):
        out = self.enc(**x)
        pooled = out.last_hidden_state[:,0]
        return self.cls_head(pooled)

class MultiModalLateFusion(nn.Module):
    def __init__(self, n_out):
        super().__init__()
        self.img = ImageHead(n_out)
        self.txt = TextHead(n_out)

# --------------------------
# 로딩 / 헤드 교체
# --------------------------
def load_ckpt_as_model(ckpt_path: str, n_out: int):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    mode = ckpt.get("mode", "multimodal")
    model = MultiModalLateFusion(n_out)
    model.load_state_dict(ckpt["model"], strict=False)
    return model.to(DEVICE), mode

def maybe_apply_updated_heads(model, cid: int, n_out: int) -> bool:
    """
    outputs/client_xx/updated_heads.npz가 있으면 가능한 헤드에 가중치를 주입.
    - 지원 키:
        1) 'W_img','b_img','W_txt','b_txt'  (모듈별 저장 케이스)
        2) 'weight','bias'                   (단일 헤드 저장 케이스; shape으로 img/txt를 자동 매핑)
    반환: 하나라도 적용되면 True
    """
    npz_path = cdir(cid) / "updated_heads.npz"
    if not npz_path.exists():
        return False

    try:
        data = np.load(npz_path)
        keys = set(data.files)
    except Exception:
        return False

    applied = False

    def _as_tensor(arr, like_weight):
        return torch.tensor(arr, dtype=like_weight.dtype, device=like_weight.device)

    with torch.no_grad():
        # --- Case 1: 명시적 키들(W_img/W_txt/b_img/b_txt)
        if {"W_img","b_img"}.issubset(keys) and hasattr(model, "img"):
            Wi = _as_tensor(data["W_img"], model.img.head.weight)
            bi = _as_tensor(data["b_img"], model.img.head.weight)  # dtype/device 맞춤
            if tuple(Wi.shape) == tuple(model.img.head.weight.shape) and tuple(bi.shape) == tuple(model.img.head.bias.shape):
                model.img.head.weight.copy_(Wi); model.img.head.bias.copy_(bi); applied = True

        if {"W_txt","b_txt"}.issubset(keys) and hasattr(model, "txt"):
            Wt = _as_tensor(data["W_txt"], model.txt.cls_head.weight)
            bt = _as_tensor(data["b_txt"], model.txt.cls_head.weight)
            if tuple(Wt.shape) == tuple(model.txt.cls_head.weight.shape) and tuple(bt.shape) == tuple(model.txt.cls_head.bias.shape):
                model.txt.cls_head.weight.copy_(Wt); model.txt.cls_head.bias.copy_(bt); applied = True

        # --- Case 2: 단일 키(weight/bias) → shape으로 매핑
        if {"weight","bias"}.issubset(keys):
            W = torch.tensor(data["weight"])
            b = torch.tensor(data["bias"])

            # img head (in_features=576?)
            if hasattr(model, "img"):
                if tuple(W.shape) == tuple(model.img.head.weight.shape) and tuple(b.shape) == tuple(model.img.head.bias.shape):
                    model.img.head.weight.copy_(_as_tensor(data["weight"], model.img.head.weight))
                    model.img.head.bias.copy_(_as_tensor(data["bias"], model.img.head.weight))
                    applied = True

            # txt head (in_features=256?)
            if not applied and hasattr(model, "txt"):
                if tuple(W.shape) == tuple(model.txt.cls_head.weight.shape) and tuple(b.shape) == tuple(model.txt.cls_head.bias.shape):
                    model.txt.cls_head.weight.copy_(_as_tensor(data["weight"], model.txt.cls_head.weight))
                    model.txt.cls_head.bias.copy_(_as_tensor(data["bias"], model.txt.cls_head.weight))
                    applied = True

    return applied


# --------------------------
# 평가 함수
# --------------------------
@torch.no_grad()
def evaluate_mmasked(model, loader):
    model.eval()
    crit = nn.BCEWithLogitsLoss()
    n_samples, tot_loss = 0, 0.0
    y_true, y_prob = [], []

    for batch in loader:
        y = batch["labels"].to(DEVICE)
        if isinstance(model, MultiModalLateFusion):
            mi = batch["has_img"].to(DEVICE).unsqueeze(1)
            mt = batch["has_txt"].to(DEVICE).unsqueeze(1)
            li = model.img(batch["image"].to(DEVICE))
            lt = model.txt(**{k:v.to(DEVICE) for k,v in batch["text"].items()})
            logits = (li*mi + lt*mt) / (mi+mt).clamp(min=1.0)
        elif isinstance(model, ImageHead):
            logits = model(batch["image"].to(DEVICE))
        else:
            logits = model(**{k:v.to(DEVICE) for k,v in batch["text"].items()})

        loss = crit(logits, y)
        bs = y.size(0)
        tot_loss += loss.item() * bs
        n_samples += bs

        y_true.append(y.cpu().numpy())
        y_prob.append(torch.sigmoid(logits).cpu().numpy())

    y_true = np.vstack(y_true)
    y_prob = np.vstack(y_prob)
    metrics = compute_metrics(y_true, y_prob)
    avg_loss = tot_loss / max(1, n_samples)
    return avg_loss, metrics["f1_micro"], metrics["f1_macro"], metrics["macro_auroc_valid"], np.array(metrics["per_class_auroc"])

# --------------------------
# 메인
# --------------------------
def parse_clients(s: str) -> List[int]:
    s = s.strip().lower()
    if s in ("all",""): return list(range(1,21))
    out=[]
    for part in s.split(","):
        part=part.strip()
        if "-" in part:
            a,b = part.split("-",1)
            out.extend(list(range(int(a), int(b)+1)))
        else:
            out.append(int(part))
    return sorted(list(dict.fromkeys(out)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", type=str, default="1-20")
    ap.add_argument("--test_csv", type=str, required=True)
    ap.add_argument("--label_csv", type=str, required=True)
    ap.add_argument("--batch", type=int, default=BATCH_SIZE)
    args = ap.parse_args()

    ds = TestCSVDataset(args.test_csv, args.label_csv)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False)
    n_out = len(LABEL_COLUMNS)

    clients = parse_clients(args.clients)
    rows = []
    for cid in clients:
        ckpt_path = cdir(cid) / "best.pt"
        if not ckpt_path.exists():
            print(f"[WARN] skip client {cid:02d} (no best.pt)")
            continue

        # 1️⃣ Baseline
        base_model, mode = load_ckpt_as_model(str(ckpt_path), n_out)
        b_loss, b_f1_micro, b_f1_macro, b_macro_auc, b_aucs = evaluate_mmasked(base_model, dl)

        # 2️⃣ Updated
        upd_model, _ = load_ckpt_as_model(str(ckpt_path), n_out)
        applied = maybe_apply_updated_heads(upd_model, cid, n_out)
        if not applied:
            print(f"[INFO] client_{cid:02d}: updated_heads.npz 없음 또는 shape 불일치 → baseline과 동일로 간주")
        u_loss, u_f1_micro, u_f1_macro, u_macro_auc, u_aucs = evaluate_mmasked(upd_model, dl)

        print(f"[Client {cid:02d} | {mode}] BASE mAUC={b_macro_auc:.4f} → UPD mAUC={u_macro_auc:.4f}  Δ={u_macro_auc - b_macro_auc:+.4f}")

        row = {
            "client_id": cid, "mode": mode,
            "baseline_loss": b_loss, "updated_loss": u_loss,
            "baseline_f1_micro": b_f1_micro, "updated_f1_micro": u_f1_micro,
            "baseline_f1_macro": b_f1_macro, "updated_f1_macro": u_f1_macro,
            "baseline_macro_auroc": b_macro_auc, "updated_macro_auroc": u_macro_auc,
            "delta_macro_auroc": u_macro_auc - b_macro_auc
        }
        rows.append(row)

    if rows:
        headers = list(rows[0].keys())
        with open(SUMMARY_CSV, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=headers)
            w.writeheader()
            w.writerows(rows)
        print(f"\n[INFO] Saved summary → {SUMMARY_CSV} ({len(rows)} clients)")
    else:
        print("[WARN] no evaluated clients.")

if __name__ == "__main__":
    main()
