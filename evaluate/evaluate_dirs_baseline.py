# -*- coding: utf-8 -*-
"""
evaluate_dirs_baseline.py
- 훈련/업데이트/임베딩 없이, 디렉터리 트리를 재귀 스캔해 baseline(best.pt)로만 평가
- 폴더명에서 p#######/s########를 파싱하여 라벨 CSV에서 라벨 매칭
- 모달 마스크 기반 late fusion: 텍스트 없으면 이미지 로짓만 사용

예)
  python -u evaluate_dirs_baseline.py --clients 1-20 \
    --label_csv .\mimic-cxr-2.1.0-test-set-labeled.csv \
    --roots ".\data\p10,.\data\p11"

  # 기본값은 --roots "./data/p10,./data/p11"
"""

import os, re, csv, json, argparse, glob
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score, roc_auc_score
from PIL import Image, ImageFile

# --------------------------
# 설정
# --------------------------
ImageFile.LOAD_TRUNCATED_IMAGES = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_WORKERS = 0
TEXT_MODEL_NAME = "prajjwal1/bert-mini"

LABEL_COLUMNS = [
    "Atelectasis","Cardiomegaly","Consolidation","Edema","Pleural Effusion",
    "Enlarged Cardiomediastinum","Fracture","Lung Lesion","Lung Opacity",
    "Pleural Other","Pneumonia","Pneumothorax","Support Devices"
]

OUT_DIR = Path("./eval_results"); OUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_CSV = OUT_DIR / "summary_dirs_baseline.csv"

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

def parse_roots(roots_arg: str) -> List[str]:
    if not roots_arg:
        return ["./data/p10", "./data/p11"]
    return [r.strip() for r in roots_arg.split(",") if r.strip()]

def find_study_dirs(roots: List[str]) -> List[str]:
    study_dirs: List[str] = []
    for root in roots:
        if not os.path.exists(root): continue
        # p*/s* 형태의 스터디 폴더를 재귀 검색
        study_dirs.extend(glob.glob(os.path.join(root, "**", "s*"), recursive=True))
    study_dirs = [d for d in study_dirs if os.path.isdir(d)]
    # p#######/s######## 패턴만 유지
    kept = []
    for d in study_dirs:
        if re.search(r"[\\/]p\d+[\\/]", d) and re.search(r"[\\/]s\d+[\\/]?$", d):
            kept.append(d)
    # 중복 제거
    return sorted(list(dict.fromkeys(kept)))

def parse_ids_from_path(study_dir: str) -> Tuple[int,int]:
    # .../p10/p100032/s50414267 -> (100032, 50414267)
    m_p = re.search(r"[\\/]p(\d+)[\\/]", study_dir)
    m_s = re.search(r"[\\/]s(\d+)[\\/]?$", study_dir)
    if not m_p or not m_s:
        raise ValueError(f"경로에서 subject/study를 파싱할 수 없습니다: {study_dir}")
    return int(m_p.group(1)), int(m_s.group(1))

def load_label_table(label_csv: str) -> Dict[Tuple[int,int], List[int]]:
    import pandas as pd
    if not os.path.exists(label_csv):
        raise FileNotFoundError(f"라벨 CSV가 없습니다: {label_csv}")
    df = pd.read_csv(label_csv)
    required = {"subject_id","study_id", *LABEL_COLUMNS}
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"라벨 CSV에 필요한 칼럼이 없습니다: {miss}")
    for c in LABEL_COLUMNS:
        df[c] = df[c].fillna(0)
        df[c] = (df[c] >= 1).astype(int)
    table = {}
    for _, r in df.iterrows():
        key = (int(r["subject_id"]), int(r["study_id"]))
        table[key] = [int(r[c]) for c in LABEL_COLUMNS]
    return table

# --------------------------
# 데이터셋 (원본 파일 로드, 텍스트 없으면 마스크=0)
# --------------------------
class TestDirsDataset(Dataset):
    def __init__(self, study_dirs: List[str], label_table: Dict[Tuple[int,int], List[int]]):
        self.items = []
        for d in study_dirs:
            try:
                sid, stid = parse_ids_from_path(d)
            except Exception:
                continue
            y = label_table.get((sid, stid), [0]*len(LABEL_COLUMNS))
            self.items.append({"dir": d, "y": y})
        if not self.items:
            raise RuntimeError("유효한 스터디 폴더가 없습니다(라벨 매칭 실패 가능).")
        self.tok = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)

    def __len__(self): return len(self.items)

    def _load_image(self, d: str):
        imgs = sorted([*Path(d).glob("*.jpg"), *Path(d).glob("*.jpeg"), *Path(d).glob("*.png")])
        if not imgs:
            return torch.zeros(3,224,224), 0.0
        try:
            return IMG_TRANSFORM(Image.open(imgs[0]).convert("RGB")), 1.0
        except Exception:
            return torch.zeros(3,224,224), 0.0

    def _find_text(self, d: str) -> Optional[Path]:
        cands = list(Path(d).glob("*.txt"))
        return cands[0] if cands else None

    def _load_text(self, p: Optional[Path]):
        if p is None:
            text = ""
        else:
            try:
                text = Path(p).read_text(encoding="utf-8")
            except Exception:
                text = ""
        enc = self.tok(text, truncation=True, padding="max_length", max_length=256, return_tensors="pt")
        return {k: v.squeeze(0) for k,v in enc.items()}, (0.0 if text=="" else 1.0)

    def __getitem__(self, idx):
        it = self.items[idx]
        y = torch.tensor(np.array(it["y"], dtype=np.float32))
        img, has_img = self._load_image(it["dir"])
        txt_path = self._find_text(it["dir"])
        txt, has_txt = self._load_text(txt_path)
        return {
            "labels": y,
            "image": img,
            "text": txt,
            "has_img": torch.tensor([has_img], dtype=torch.float32).squeeze(0),
            "has_txt": torch.tensor([has_txt], dtype=torch.float32).squeeze(0),
        }

# --------------------------
# 모델 (train_local.py와 동형)
# --------------------------
class ImageHead(nn.Module):
    def __init__(self, n_out):
        super().__init__()
        base = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        in_f = base.classifier[0].in_features
        base.classifier = nn.Identity()
        self.backbone = base
        self.head = nn.Linear(in_f, n_out)
    def forward(self, x): return self.head(self.backbone(x))

class TextHead(nn.Module):
    def __init__(self, n_out):
        super().__init__()
        self.enc = AutoModel.from_pretrained(TEXT_MODEL_NAME)
        hid = self.enc.config.hidden_size
        self.cls_head = nn.Linear(hid, n_out)
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.enc(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled = out.last_hidden_state[:,0]
        return self.cls_head(pooled)

class MultiModalLateFusion(nn.Module):
    def __init__(self, n_out):
        super().__init__()
        self.img = ImageHead(n_out)
        self.txt = TextHead(n_out)

def load_ckpt_as_model(ckpt_path: str, n_out: int):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    mode = ckpt.get("mode", "multimodal")
    if mode == "multimodal":
        model = MultiModalLateFusion(n_out)
    elif mode == "image_only":
        model = ImageHead(n_out)
    elif mode == "text_only":
        model = TextHead(n_out)
    else:
        model = MultiModalLateFusion(n_out)
    model.load_state_dict(ckpt["model"])
    return model.to(DEVICE), mode

# --------------------------
# 평가 (모달 마스크 기반 late fusion)
# --------------------------
@torch.no_grad()
def evaluate_mmasked(model, loader):
    model.eval()
    crit = nn.BCEWithLogitsLoss()
    tot = 0.0; y_true=[]; y_prob=[]
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
        tot += loss.item()*y.size(0)
        y_true.append(y.cpu().numpy())
        y_prob.append(torch.sigmoid(logits).cpu().numpy())
    y_true = np.vstack(y_true); y_prob = np.vstack(y_prob)
    f1_micro = f1_score(y_true, (y_prob>=0.5).astype(int), average="micro", zero_division=0)
    f1_macro = f1_score(y_true, (y_prob>=0.5).astype(int), average="macro", zero_division=0)
    aucs=[]
    for j in range(y_true.shape[1]):
        col = y_true[:,j]
        if len(np.unique(col))<2: aucs.append(float("nan"))
        else:
            try: aucs.append(roc_auc_score(col, y_prob[:,j]))
            except: aucs.append(float("nan"))
    macro_auc = float(np.nanmean(aucs))
    return tot/len(loader.dataset), f1_micro, f1_macro, macro_auc, aucs

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
    ap.add_argument("--label_csv", type=str, required=True)
    ap.add_argument("--roots", type=str, default="./data/p10,./data/p11",
                    help="쉼표로 여러 루트. 각 루트에서 p*/s* 디렉터리를 재귀 검색")
    ap.add_argument("--batch", type=int, default=BATCH_SIZE)
    args = ap.parse_args()

    # 테스트 세트 구성 (test.csv 필요 없음)
    roots = parse_roots(args.roots)
    study_dirs = find_study_dirs(roots)
    if not study_dirs:
        raise RuntimeError(f"스터디 폴더를 찾지 못했습니다. roots={roots}")
    label_table = load_label_table(args.label_csv)
    ds = TestDirsDataset(study_dirs, label_table)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=NUM_WORKERS)
    n_out = len(LABEL_COLUMNS)

    clients = parse_clients(args.clients)
    rows=[]
    for cid in clients:
        ckpt_path = cdir(cid)/"best.pt"
        if not ckpt_path.exists():
            print(f"[WARN] skip client {cid:02d} (no best.pt)")
            continue
        model, mode = load_ckpt_as_model(str(ckpt_path), n_out)
        loss, f1_micro, f1_macro, macro_auc, aurocs = evaluate_mmasked(model, dl)
        print(f"[Client {cid:02d} | {mode:11s}] "
              f"loss={loss:.4f}  f1_micro={f1_micro:.4f}  f1_macro={f1_macro:.4f}  macro_auc={macro_auc:.4f}")

        # per-client JSON 저장
        outj = {
            "client_id": cid, "mode": mode,
            "loss": loss, "f1_micro": f1_micro, "f1_macro": f1_macro,
            "macro_auroc": macro_auc,
            "per_class_auroc": {c: (None if (a!=a) else float(a)) for c,a in zip(LABEL_COLUMNS, aurocs)}
        }
        with open(cdir(cid)/"test_metrics_dirs.json", "w", encoding="utf-8") as f:
            json.dump(outj, f, indent=2, ensure_ascii=False)

        row = {"client_id": cid, "mode": mode, "loss": loss,
               "f1_micro": f1_micro, "f1_macro": f1_macro, "macro_auroc": macro_auc}
        for c,a in zip(LABEL_COLUMNS, aurocs): row[f"AUROC_{c}"] = a
        rows.append(row)

    # 요약 CSV
    if rows:
        rows.sort(key=lambda r: (r["macro_auroc"] if r["macro_auroc"]==r["macro_auroc"] else -1), reverse=True)
        headers = ["client_id","mode","loss","f1_micro","f1_macro","macro_auroc"] + [f"AUROC_{c}" for c in LABEL_COLUMNS]
        with open(SUMMARY_CSV, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=headers); w.writeheader(); w.writerows(rows)
        print(f"\n[INFO] Saved summary → {SUMMARY_CSV} ({len(rows)} rows)")
        print("Top-5 by macro_auroc:")
        for r in rows[:5]:
            print(f"  client_{r['client_id']:02d} [{r['mode']}]: {r['macro_auroc']:.4f}")
    else:
        print("[WARN] no evaluated clients; check paths/label_csv")

if __name__ == "__main__":
    main()