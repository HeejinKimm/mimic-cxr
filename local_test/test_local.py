# evaluate_all_clients_on_test.py
import os, csv
from pathlib import Path
import sys
from typing import Dict, Tuple, List, Optional

# === 프로젝트 루트 경로 등록 (local_train import 전에!) ===
PROJ_ROOT = Path(__file__).resolve().parents[1]  # .../mimic-cxr/mimic-cxr
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer

# 학습과 동일한 구성 재사용
from local_train.config import Cfg
from local_train.data import img_transform, build_image_picker_from_metadata, load_label_table
from local_train.models import MultiModalLateFusion

# =========================
# 경로/설정
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = (PROJ_ROOT / "local_test_results"); OUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_CSV = OUT_DIR / "summary.csv"
BATCH_SIZE = 32
NUM_WORKERS = 0  # Windows에서는 0 권장

def _default_split_dir(cfg: Cfg) -> Path:
    """client_splits(test.csv)을 가장 그럴듯한 위치에서 탐색"""
    candidates: List[Path] = []
    try:
        if getattr(cfg, "CLIENT_CSV_DIR", None):
            candidates.append(Path(cfg.CLIENT_CSV_DIR))
    except Exception:
        pass
    here = Path(__file__).resolve()
    candidates.append(here.parents[1] / "client_splits")  # .../mimic-cxr/mimic-cxr/client_splits
    candidates.append(Path.cwd() / "client_splits")
    for p in candidates:
        if p.exists():
            return p
    return candidates[0] if candidates else Path("./client_splits")

def _resolve_test_csv(split_dir: Path) -> Path:
    cand = [split_dir / "test.csv", split_dir / "client_00.csv"]
    for p in cand:
        if p.exists():
            return p
    have = sorted([p.name for p in split_dir.glob("*.csv")])
    raise FileNotFoundError(
        f"[client_splits] test csv를 찾을 수 없습니다. 찾은 경로 후보: {cand}\n"
        f"split_dir='{split_dir}' 목록: {have}"
    )

# =========================
# 데이터셋
# =========================
class TestDataset(Dataset):
    """
    학습 DataLoader의 배치 키와 최대한 호환:
      pixel_values, input_ids, attention_mask, img_mask, txt_mask, labels
    """
    def __init__(self, csv_path: str, label_table, meta_picker=None, cfg: Cfg = Cfg()):
        import csv as _csv
        self.rows = []
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            for row in _csv.DictReader(f):
                self.rows.append(row)

        self.label_table = label_table
        self.meta_picker = meta_picker
        self.cfg = cfg
        self.tok = AutoTokenizer.from_pretrained(cfg.TEXT_MODEL_NAME)
        self.img_tf = img_transform()

    def __len__(self):
        return len(self.rows)

    def _id_to_int(self, subject_id: str, study_id: str) -> Tuple[int, int]:
        return int(subject_id[1:]), int(study_id[1:])

    def _load_image_with_flag(self, image_dir: str, subject_id: str=None, study_id: str=None) -> Tuple[torch.Tensor, int]:
        """메타데이터가 있으면 거기서 대표 이미지. 없으면 디렉토리 내 첫 이미지. 못 찾으면 zero 텐서 + mask=0"""
        from PIL import Image
        p_img: Optional[str] = None
        if self.meta_picker and subject_id and study_id:
            key = (int(subject_id[1:]), int(study_id[1:]))
            p_img = self.meta_picker.get(key)
        try:
            if p_img and os.path.exists(p_img):
                img = self.img_tf(Image.open(p_img).convert("RGB"))
                return img, 1
        except Exception:
            pass

        p = Path(image_dir)
        imgs = sorted([*p.glob("*.jpg"), *p.glob("*.jpeg"), *p.glob("*.png")]) if p.exists() else []
        if imgs:
            try:
                img = self.img_tf(Image.open(imgs[0]).convert("RGB"))
                return img, 1
            except Exception:
                pass
        # fallback
        return torch.zeros(3, 224, 224), 0

    def _load_text_with_flag(self, text_path: str):
        # 파일이 있고 내용이 비어있지 않으면 mask=1, 아니면 mask=0
        text = ""
        present = 0
        try:
            if os.path.exists(text_path):
                with open(text_path, "r", encoding="utf-8") as f:
                    text = f.read()
                if text.strip():
                    present = 1
        except Exception:
            pass
        enc = self.tok(text, truncation=True, padding="max_length", max_length=self.cfg.MAX_LEN, return_tensors="pt")
        enc = {k: v.squeeze(0) for k, v in enc.items()}  # [L] -> ok
        return enc, present

    def __getitem__(self, idx):
        r = self.rows[idx]
        sid_int, stid_int = self._id_to_int(r["subject_id"], r["study_id"])
        y = torch.tensor(self.label_table.get((sid_int, stid_int), [0]*len(self.cfg.LABEL_COLUMNS)), dtype=torch.float)

        # image/text + presence mask
        pixel_values, img_present = self._load_image_with_flag(r["image_dir"], r["subject_id"], r["study_id"])
        text_tok,   txt_present   = self._load_text_with_flag(r["text_path"])

        sample = {
            "labels": y,
            "pixel_values": pixel_values,                  # [3,224,224]
            "input_ids": text_tok["input_ids"],           # [L]
            "attention_mask": text_tok["attention_mask"], # [L]
            # 모델이 [B,1] 마스크를 기대하므로 여기서 1D로 두고 collate되면 [B]가 됨 → 모델에서 unsqueeze(1)
            "img_mask": torch.tensor(img_present, dtype=torch.float),
            "txt_mask": torch.tensor(txt_present, dtype=torch.float),
        }
        if "token_type_ids" in text_tok:
            sample["token_type_ids"] = text_tok["token_type_ids"]
        return sample

# =========================
# 평가 루프
# =========================
@torch.no_grad()
def evaluate_model_on_test(model: MultiModalLateFusion, loader, criterion, device: str,
                           has_token_type_ids: bool, force_mode: str,
                           require_both: bool = False):
    """
    force_mode: 'as_trained'에서 넘어온 최종 모드 값(text_only/image_only/fusion 또는 ckpt의 mode)
    require_both: True면 이미지+텍스트 둘 다 있는 샘플만 집계 (fusion 성능 순수 평가용)
    """
    model.eval()
    total_loss = 0.0
    total_n = 0
    y_true, y_prob = [], []

    for batch in loader:
        pix = batch["pixel_values"].to(device)
        ids = batch["input_ids"].to(device)
        att = batch["attention_mask"].to(device)
        img_m = batch["img_mask"].to(device).unsqueeze(1)     # [B,1]
        txt_m = batch["txt_mask"].to(device).unsqueeze(1)     # [B,1]

        # 샘플 필터링 (둘 다 있어야만 평가)
        if require_both:
            keep = ((img_m > 0.5) & (txt_m > 0.5)).squeeze(1)  # [B]
            if keep.sum().item() == 0:
                continue
            pix   = pix[keep]
            ids   = ids[keep]
            att   = att[keep]
            img_m = img_m[keep]
            txt_m = txt_m[keep]
            y     = batch["labels"][keep].to(device)
        else:
            y = batch["labels"].to(device)

        # 모드 강제
        if force_mode == "text_only":
            img_m = torch.zeros_like(img_m); pix = torch.zeros_like(pix)
        elif force_mode == "image_only":
            txt_m = torch.zeros_like(txt_m)
            # att = torch.zeros_like(att)  # 선택

        logits = model(pixel_values=pix, input_ids=ids, attention_mask=att,
                       img_mask=img_m, txt_mask=txt_m)

        loss = criterion(logits, y)
        bs = y.size(0)
        total_loss += loss.item() * bs
        total_n += bs
        y_true.append(y.detach().cpu().numpy())
        y_prob.append(torch.sigmoid(logits).detach().cpu().numpy())

    if total_n == 0:
        # 평가할 샘플이 하나도 없을 때
        out_dim = model.fuse[-1].out_features if hasattr(model.fuse[-1], "out_features") else 0
        return float("nan"), [float("nan")] * out_dim

    y_true = np.vstack(y_true); y_prob = np.vstack(y_prob)
    aurocs = []
    for j in range(y_true.shape[1]):
        try:
            aurocs.append(roc_auc_score(y_true[:, j], y_prob[:, j]))
        except ValueError:
            aurocs.append(float("nan"))
    return total_loss/total_n, aurocs

def load_ckpt_as_model(ckpt_path: str, n_out: int, device: str, cfg: Cfg):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = MultiModalLateFusion(n_out, text_model_name=cfg.TEXT_MODEL_NAME)
    model.load_state_dict(ckpt["model"])
    return model.to(device), ckpt.get("mode", "multimodal")

# =========================
# 메인
# =========================
def main(split_dir: Optional[str] = None,
         eval_mode: str = "as_trained",
         global_ckpt: Optional[str] = None,
         clients: Optional[List[int]] = None,
         require_both: bool = False):
    cfg = Cfg()

    # test.csv & 라벨/메타 로드
    base_dir = Path(split_dir) if split_dir else _default_split_dir(cfg)
    test_csv = _resolve_test_csv(base_dir)

    label_csv = cfg.LABEL_CSV_NEGBIO if cfg.USE_LABEL == "negbio" else cfg.LABEL_CSV_CHEXPERT
    label_table = load_label_table(str(label_csv), cfg.LABEL_COLUMNS)
    meta_picker = build_image_picker_from_metadata(str(cfg.METADATA_CSV), str(cfg.IMG_ROOT))

    dataset = TestDataset(str(test_csv), label_table, meta_picker=meta_picker, cfg=cfg)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    criterion = nn.BCEWithLogitsLoss()
    n_out = len(cfg.LABEL_COLUMNS)

    rows = []

    def _eval_one(ckpt_path: Path, tag: str):
        model, mode_ckpt = load_ckpt_as_model(str(ckpt_path), n_out, DEVICE, cfg)
        has_token_type_ids = ("token_type_ids" in dataset[0]) if len(dataset) > 0 else False
        # 최종 평가 모드 결정
        force = mode_ckpt if eval_mode == "as_trained" else eval_mode

        loss, aurocs = evaluate_model_on_test(
            model, loader, criterion, DEVICE, has_token_type_ids,
            force_mode=force, require_both=require_both
        )
        macro = float(np.nanmean(aurocs))
        print(f"[{tag:<12s} | ckpt={mode_ckpt:11s} | eval={force:7s}] Loss={loss:.4f}  Macro-AUROC={macro:.4f}")

        row = {"tag": tag, "mode_ckpt": mode_ckpt, "mode_eval": force, "loss": loss, "macro_auroc": macro}
        for c, a in zip(cfg.LABEL_COLUMNS, aurocs):
            row[f"AUROC_{c}"] = a
        rows.append(row)

    if global_ckpt:
        # 단일 글로벌 모델 평가
        _eval_one(Path(global_ckpt), tag="GLOBAL")
    else:
        # 클라이언트 목록
        id_list = clients if clients else range(1, 21)
        for cid in id_list:
            ckpt_path = PROJ_ROOT / f"local_outputs/client_{cid:02d}/best.pt"
            if not ckpt_path.exists():
                print(f"[WARN] skip client {cid:02d} (checkpoint not found: {ckpt_path})")
                continue
            _eval_one(ckpt_path, tag=f"client_{cid:02d}")

    # 저장
    if rows:
        rows.sort(key=lambda r: (r["macro_auroc"] if r["macro_auroc"] == r["macro_auroc"] else -1), reverse=True)
        headers = ["tag","mode_ckpt","mode_eval","loss","macro_auroc"] + [f"AUROC_{c}" for c in cfg.LABEL_COLUMNS]
        with open(SUMMARY_CSV, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=headers)
            w.writeheader(); w.writerows(rows)
        print(f"\n[INFO] Saved summary → {SUMMARY_CSV} ({len(rows)} rows)")
        print("\nTop-5 by Macro-AUROC:")
        for r in rows[:5]:
            print(f"  {r['tag']:>12s} | eval={r['mode_eval']}: {r['macro_auroc']:.4f}")
    else:
        print("[WARN] No results.")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_dir", type=str, default=None, help="client_splits 디렉토리 경로 (옵션)")
    ap.add_argument("--eval_mode", choices=["as_trained","text_only","image_only","fusion"],
                    default="as_trained", help="평가 모드 선택")
    ap.add_argument("--global_ckpt", type=str, default=None,
                    help="단일 글로벌 체크포인트 경로(.pt). 지정되면 이것만 평가")
    ap.add_argument("--clients", type=int, nargs="+", default=None,
                    help="평가할 client id 목록 (예: --clients 19 20). 지정 없으면 1..20")
    ap.add_argument("--require_both", action="store_true",
                    help="fusion 평가 시 이미지+텍스트 모두 있는 샘플만 집계")
    args = ap.parse_args()

    main(split_dir=args.split_dir,
         eval_mode=args.eval_mode,
         global_ckpt=args.global_ckpt,
         clients=args.clients,
         require_both=args.require_both)
