# global_test/test.py
# ------------------------------------------------------------
# Global payload(Z) 기반 테스트 스크립트 (final_output/summary.csv 저장)
# - PROJ_ROOT/global_output/client_XX/global_payload.pt 자동 탐색
# - 각 client의 local best.pt 로 모델 구성 후, 전역 Z를 주입하여 평가
#   * Z가 2D (out,in | in,out | flat numel 일치) -> weight에 주입
#   * Z가 1D 이고 len==out_features -> bias에 주입 (weight는 로컬 유지)
# - 결과: PROJ_ROOT/final_output/summary.csv
# ------------------------------------------------------------

import os, csv, re
from pathlib import Path
import sys
from typing import Dict, Tuple, List, Optional

# === 프로젝트 루트 경로 등록 (local_train import 전에!) ===
PROJ_ROOT = Path(__file__).resolve().parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer
from PIL import Image

# 학습과 동일한 구성 재사용
from local_train.config import Cfg
from local_train.data import img_transform, build_image_picker_from_metadata, load_label_table
from local_train.models import MultiModalLateFusion

# =========================
# 경로/설정
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = (PROJ_ROOT / "final_output"); OUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_CSV = OUT_DIR / "summary.csv"
BATCH_SIZE = 32
NUM_WORKERS = 0  # Windows에서는 0 권장

def _default_split_dir(cfg: Cfg) -> Path:
    candidates: List[Path] = []
    try:
        if getattr(cfg, "CLIENT_CSV_DIR", None):
            candidates.append(Path(cfg.CLIENT_CSV_DIR))
    except Exception:
        pass
    here = Path(__file__).resolve()
    candidates.append(here.parents[1] / "client_splits")
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
        f"[client_splits] test csv를 찾을 수 없습니다. 후보: {cand}\n"
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
        return torch.zeros(3, 224, 224), 0

    def _load_text_with_flag(self, text_path: str):
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
        enc = self.tok(text, truncation=True, padding="max_length",
                       max_length=self.cfg.MAX_LEN, return_tensors="pt")
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        return enc, present

    def __getitem__(self, idx):
        r = self.rows[idx]
        sid_int, stid_int = self._id_to_int(r["subject_id"], r["study_id"])
        y = torch.tensor(self.label_table.get((sid_int, stid_int), [0]*len(self.cfg.LABEL_COLUMNS)), dtype=torch.float)

        pixel_values, img_present = self._load_image_with_flag(r["image_dir"], r["subject_id"], r["study_id"])
        text_tok,   txt_present   = self._load_text_with_flag(r["text_path"])

        sample = {
            "labels": y,
            "pixel_values": pixel_values,                  # [3,224,224]
            "input_ids": text_tok["input_ids"],           # [L]
            "attention_mask": text_tok["attention_mask"], # [L]
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
    model.eval()
    total_loss = 0.0
    total_n = 0
    y_true, y_prob = [], []

    for batch in loader:
        pix = batch["pixel_values"].to(device)
        ids = batch["input_ids"].to(device)
        att = batch["attention_mask"].to(device)
        img_m = batch["img_mask"].to(device).unsqueeze(1)
        txt_m = batch["txt_mask"].to(device).unsqueeze(1)

        if require_both:
            keep = ((img_m > 0.5) & (txt_m > 0.5)).squeeze(1)
            if keep.sum().item() == 0:
                continue
            pix   = pix[keep]; ids = ids[keep]; att = att[keep]
            img_m = img_m[keep]; txt_m = txt_m[keep]
            y     = batch["labels"][keep].to(device)
        else:
            y = batch["labels"].to(device)

        if force_mode == "text_only":
            img_m = torch.zeros_like(img_m); pix = torch.zeros_like(pix)
        elif force_mode == "image_only":
            txt_m = torch.zeros_like(txt_m)

        logits = model(pixel_values=pix, input_ids=ids, attention_mask=att,
                       img_mask=img_m, txt_mask=txt_m)

        loss = criterion(logits, y)
        bs = y.size(0)
        total_loss += loss.item() * bs
        total_n += bs
        y_true.append(y.detach().cpu().numpy())
        y_prob.append(torch.sigmoid(logits).detach().cpu().numpy())

    if total_n == 0:
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

# =========================
# 글로벌 페이로드(Z) -> 모델 주입 유틸
# =========================
def _find_fusion_linear(model: nn.Module, fused_dim: int, d_model: int) -> Optional[nn.Linear]:
    for m in model.modules():
        if isinstance(m, nn.Linear) and getattr(m, "in_features", None) == fused_dim and getattr(m, "out_features", None) == d_model:
            return m
    return None

def _find_linear_by_out(model: nn.Module, out_features: int) -> Optional[nn.Linear]:
    for m in model.modules():
        if isinstance(m, nn.Linear) and getattr(m, "out_features", None) == out_features:
            return m
    return None

def _find_linear_by_numel(model: nn.Module, numel: int) -> Optional[nn.Linear]:
    for m in model.modules():
        if isinstance(m, nn.Linear) and m.weight.numel() == numel:
            return m
    return None

def _load_local_best_for_client(cid: int, n_out: int, device: str, cfg: Cfg):
    base = Path(getattr(cfg, "BASE_DIR", PROJ_ROOT / "local_train_outputs"))
    ckpt_name = getattr(cfg, "CKPT_NAME", "best.pt")
    local_ckpt = base / f"client_{cid:02d}" / ckpt_name
    if not local_ckpt.exists():
        raise FileNotFoundError(f"[ERR] Local ckpt not found: {local_ckpt}")

    try:
        obj = torch.load(local_ckpt, map_location="cpu", weights_only=True)  # torch>=2.4
    except TypeError:
        obj = torch.load(local_ckpt, map_location="cpu")

    state = obj["model"] if isinstance(obj, dict) and "model" in obj else (obj.get("state_dict", obj) if isinstance(obj, dict) else obj)
    mode = obj.get("mode", obj.get("mode_ckpt", "fusion")) if isinstance(obj, dict) else "fusion"

    model = MultiModalLateFusion(n_out, text_model_name=cfg.TEXT_MODEL_NAME)
    model.load_state_dict(state, strict=False)
    return model.to(device), mode

def _normalize_Z_to_weight_shape(Z: torch.Tensor, target_shape: Tuple[int, int]) -> torch.Tensor:
    """Z를 target (out,in) shape으로 맞춤: (out,in), (in,out), (flat numel) 모두 지원. 1D는 여기서 처리하지 않음."""
    out, in_ = target_shape
    Z = Z.detach().float().cpu()
    if Z.ndim == 2:
        if Z.shape == (out, in_):
            return Z
        if Z.shape == (in_, out):
            return Z.t()
    if Z.numel() == out * in_:
        return Z.reshape(out, in_)
    raise RuntimeError(f"[ERR] Z를 target shape {target_shape}로 변환할 수 없습니다. Z.shape={tuple(Z.shape)}, numel={Z.numel()} vs {out*in_}")

def load_ckpt_as_model_from_payload(ckpt_path: str, n_out: int, device: str, cfg: Cfg, cid: int):
    """
    1) 로컬 best.pt로 모델 로드
    2) global_payload.pt 안 Z를 타깃 선형층에 주입
       - 2D Z -> weight
       - 1D Z (len==out_features) -> bias
    """
    model, mode_local = _load_local_best_for_client(cid, n_out, device, cfg)

    try:
        obj = torch.load(ckpt_path, map_location="cpu", weights_only=True)  # torch>=2.4
    except TypeError:
        obj = torch.load(ckpt_path, map_location="cpu")

    if not isinstance(obj, dict) or "Z" not in obj:
        raise RuntimeError(f"[ERR] '{ckpt_path}'에 'Z' 키가 없습니다. keys={list(obj.keys()) if isinstance(obj, dict) else type(obj)}")

    Z = obj["Z"]
    if isinstance(Z, np.ndarray):
        Z = torch.from_numpy(Z)
    if not isinstance(Z, torch.Tensor):
        raise RuntimeError(f"[ERR] Z 타입이 torch.Tensor/np.ndarray가 아닙니다: {type(Z)}")
    Z = Z.float()

    # ---- 타깃 선형층 탐색 ----
    d_global_hint = None
    if "d_global" in obj and isinstance(obj["d_global"], (int, np.integer)):
        d_global_hint = int(obj["d_global"])
    lin: Optional[nn.Linear] = None

    if d_global_hint is not None:
        lin = _find_linear_by_out(model, d_global_hint)

    if lin is None:
        fused_dim = getattr(cfg, "FUSED_DIM", None)
        d_model   = getattr(cfg, "D_MODEL", None)
        if fused_dim and d_model:
            lin = _find_fusion_linear(model, fused_dim=fused_dim, d_model=d_model)

    if lin is None:
        # 2D일 가능성 고려하여 numel 매칭
        lin = _find_linear_by_numel(model, Z.numel())

    if lin is None:
        # 휴리스틱: in>out인 가장 큰 Linear
        cand = []
        for m in model.modules():
            if isinstance(m, nn.Linear) and m.in_features > m.out_features:
                cand.append(m)
        lin = cand[0] if cand else None

    if lin is None:
        raise RuntimeError("[ERR] 전역 Z를 주입할 타깃 Linear 층을 찾지 못했습니다.")

    # ---- Z 주입 ----
    if Z.ndim == 1:
        # 1D 벡터 → bias 주입 (len==out_features 필요)
        if Z.shape[0] != lin.out_features:
            raise RuntimeError(f"[ERR] 1D Z 길이({Z.shape[0]})가 타깃 Linear out_features({lin.out_features})와 다릅니다. "
                               f"bias 주입 불가. 파일: {ckpt_path}")
        if lin.bias is None:
            raise RuntimeError("[ERR] 타깃 Linear에 bias가 없습니다. 1D Z를 주입할 수 없습니다.")
        with torch.no_grad():
            lin.bias.copy_(Z.to(lin.bias.dtype).to(lin.bias.device))
        print(f"[INFO] Injected 1D Z into bias of Linear(out={lin.out_features}, in={lin.in_features})")
    else:
        # 2D/flat → weight 주입
        Z2 = _normalize_Z_to_weight_shape(Z, (lin.out_features, lin.in_features))
        with torch.no_grad():
            lin.weight.copy_(Z2.to(lin.weight.dtype).to(lin.weight.device))
        print(f"[INFO] Injected 2D Z into weight of Linear(out={lin.out_features}, in={lin.in_features})")

    mode_payload = obj.get("mode", obj.get("mode_ckpt", None))
    final_mode = mode_payload if mode_payload else mode_local
    return model.to(device), final_mode

def _iter_global_payloads(base_dir: Path, clients: Optional[List[int]] = None):
    if not base_dir.exists():
        return
    for sub in sorted(base_dir.glob("client_*")):
        name = sub.name  # client_05
        try:
            cid = int(name.split("_")[-1])
        except Exception:
            continue
        if clients and cid not in clients:
            continue
        pt = sub / "global_payload.pt"
        if pt.exists():
            yield pt, name

# =========================
# 메인
# =========================
def main(split_dir: Optional[str] = None,
         eval_mode: str = "as_trained",
         global_ckpt: Optional[str] = None,
         clients: Optional[List[int]] = None,
         require_both: bool = False,
         global_dir: Optional[str] = None):
    cfg = Cfg()

    base_split_dir = Path(split_dir) if split_dir else _default_split_dir(cfg)
    test_csv = _resolve_test_csv(base_split_dir)

    label_csv = cfg.LABEL_CSV_NEGBIO if getattr(cfg, "USE_LABEL", "negbio") == "negbio" else cfg.LABEL_CSV_CHEXPERT
    label_table = load_label_table(str(label_csv), cfg.LABEL_COLUMNS)
    meta_picker = build_image_picker_from_metadata(str(cfg.METADATA_CSV), str(cfg.IMG_ROOT))

    dataset = TestDataset(str(test_csv), label_table, meta_picker=meta_picker, cfg=cfg)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    criterion = nn.BCEWithLogitsLoss()
    n_out = len(cfg.LABEL_COLUMNS)

    rows: List[Dict] = []

    def _parse_cid_from_tag(tag: str) -> Optional[int]:
        m = re.search(r"client_(\d+)", tag)
        return int(m.group(1)) if m else None

    def _eval_one(ckpt_path: Path, tag: str):
        cid = _parse_cid_from_tag(tag)
        if cid is None:
            raise RuntimeError(f"[ERR] tag에서 client id를 찾을 수 없습니다: tag='{tag}'")

        model, mode_ckpt = load_ckpt_as_model_from_payload(str(ckpt_path), n_out, DEVICE, cfg, cid=cid)
        has_token_type_ids = ("token_type_ids" in dataset[0]) if len(dataset) > 0 else False
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
        _eval_one(Path(global_ckpt), tag="GLOBAL_client_00")
    else:
        base_global_dir = Path(global_dir) if global_dir else (PROJ_ROOT / "global_output")
        found_any = False
        for pt, tag in _iter_global_payloads(base_global_dir, clients=clients):
            found_any = True
            _eval_one(pt, tag=tag)
        if not found_any:
            print(f"[WARN] 글로벌 페이로드를 찾지 못했습니다. dir='{base_global_dir}' (필터: {clients})")

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
                    help="단일 글로벌 페이로드(.pt) 경로. 지정되면 이것만 평가")
    ap.add_argument("--clients", type=int, nargs="+", default=None,
                    help="평가할 client id 목록 (예: --clients 5 19 20). 지정 없으면 global_output의 모든 client_* 탐색")
    ap.add_argument("--require_both", action="store_true",
                    help="fusion 평가 시 이미지+텍스트 모두 있는 샘플만 집계")
    ap.add_argument("--global_dir", type=str, default=None,
                    help="글로벌 페이로드들이 있는 루트 디렉토리(기본: PROJ_ROOT/global_output)")
    args = ap.parse_args()

    main(split_dir=args.split_dir,
         eval_mode=args.eval_mode,
         global_ckpt=args.global_ckpt,
         clients=args.clients,
         require_both=args.require_both,
         global_dir=args.global_dir)
