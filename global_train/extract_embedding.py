# -*- coding: utf-8 -*-
"""
MIMIC-CXR용 클라이언트 준비 스크립트 (로컬 파이프라인 연동판)
- (A) summary.csv에서 macro_auroc(및 loss) 읽어 client_{cid}_metrics.json 저장
- (B) client_{cid}.csv 전체에서 이미지/텍스트 임베딩 추출 → train_img_reps.npy / train_txt_reps.npy

필수: local_train 패키지에 있는 구성/모델/데이터를 그대로 사용
- from local_train.config import Cfg
- from local_train.data import ClientDataset, build_image_picker_from_metadata, load_label_table, img_transform
- from local_train.models import MultiModalLateFusion
"""
from __future__ import annotations
import os, argparse, json
from pathlib import Path
from typing import Optional, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# ---- 글로벌 설정 ----
from .config import cfg  # global_train/config.py 의 Config

# ---- 로컬 학습 구성/구현 재사용 ----
from local_train.config import Cfg as LocalCfg
from local_train.data import ClientDataset, build_image_picker_from_metadata, load_label_table, img_transform
from local_train.models import MultiModalLateFusion

# ---- 공용 유틸 ----
from .utils_io import ensure_dir, client_dir, save_json

# ===============================
# summary.csv → per-client metrics.json
# ===============================
def _load_summary(summary_csv: str) -> Dict[str, Dict[str, float]]:
    """
    summary.csv 를 읽어 tag(client_XX) → {macro_auroc, loss, ...} 매핑 반환
    """
    import csv
    out: Dict[str, Dict[str, float]] = {}
    with open(summary_csv, newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        for row in r:
            tag = row.get("tag", "").strip()
            if not tag:
                continue
            d: Dict[str, float] = {}
            # 핵심 메트릭들만 우선 파싱(없으면 NaN)
            for k in ["loss", "macro_auroc"]:
                try:
                    d[k] = float(row.get(k, "nan"))
                except Exception:
                    d[k] = float("nan")
            out[tag] = d
    return out

def _write_client_metric_from_summary(cid: int, summary_csv: str):
    """
    client_{cid}_metrics.json 을 summary.csv 기반으로 생성
    저장 위치: BASE_DIR/client_{cid}/client_{cid}_metrics.json
    """
    m = _load_summary(summary_csv)
    tag = f"client_{cid:02d}"
    metrics = m.get(tag, None)
    cdir = client_dir(cid)  # C:\HJHJ0808\김희진\연구\졸업프로젝트\mimic-cxr\mimic-cxr\local_train_outputs\client_01
    ensure_dir(cdir)

    if metrics is None:
        # 못 찾으면 빈 값이라도 남겨 다음 단계의 로직이 진행되게 함
        # metrics = {"macro_auroc": float("nan"), "loss": float("nan")}
        # print(f"[client_{cid}] WARNING: summary.csv에서 '{tag}' 항목을 찾지 못했습니다.")
        print("metrics를 찾지 못함.")
        exit(1) # 못 찾으면 종료되게 

    # global_train/config.py 의 METRIC_NAME를 키로도 맞춰줌
    if cfg.METRIC_NAME not in metrics and "macro_auroc" in metrics:
        metrics[cfg.METRIC_NAME] = metrics["macro_auroc"]

    save_json(metrics, os.path.join(cdir, f"client_{cid}_metrics.json"))
    print(f"[client_{cid}] metrics saved from summary.csv → {metrics}")

# ===============================
# 모델 로드 & 임베딩 추출
# ===============================
def _best_ckpt_path(cid: int) -> str:
    """ckpt 후보: BASE_DIR/client_{cid:02d}/best.pt"""
    p = os.path.join(cfg.BASE_DIR, f"client_{cid:02d}", cfg.CKPT_NAME)
    if not os.path.exists(p):
        raise FileNotFoundError(f"[client_{cid}] 체크포인트를 찾을 수 없습니다: {p}")
    return p

def build_model_from_ckpt(cid: int, device: torch.device) -> MultiModalLateFusion:
    """
    local_train에서 사용한 MultiModalLateFusion 아키텍처로 모델을 만들고 best.pt(state_dict)를 로드
    """
    ckpt_path = _best_ckpt_path(cid)
    # 로컬 학습과 동일한 클래스 수/텍스트 모델명 사용
    lcfg = LocalCfg()
    num_classes = len(lcfg.LABEL_COLUMNS)
    model = MultiModalLateFusion(num_classes, text_model_name=lcfg.TEXT_MODEL_NAME)
    sd = torch.load(ckpt_path, map_location=device)
    # train_local.py에서 저장한 포맷은 {"model": state_dict, ...}
    state = sd.get("model", sd)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model

def _client_csv_path(cid: int) -> Path:
    """client_splits/client_{cid:02d}.csv 우선, 없으면 client_{cid}.csv"""
    lcfg = LocalCfg()
    d = Path(lcfg.CLIENT_CSV_DIR)
    p1 = d / f"client_{cid:02d}.csv"
    p2 = d / f"client_{cid}.csv"
    if p1.exists(): return p1
    if p2.exists(): return p2
    raise FileNotFoundError(f"[client_{cid}] client_splits CSV를 찾을 수 없습니다: {p1} 또는 {p2}")

def _mode_for_client(cid: int) -> str:
    if 1 <= cid <= 16: return "multimodal"
    if cid in (17, 18): return "image_only"
    if cid in (19, 20): return "text_only"
    if cid == 0: return "test_mix"
    raise ValueError(f"invalid client id: {cid}")

def build_dataloader_for_reps(cid: int, batch_size: int) -> DataLoader:
    """
    로컬 파이프라인과 동일한 Dataset으로 '전체 client CSV'를 로딩 (train/val 구분 없이 전량 임베딩 추출)
    """
    lcfg = LocalCfg()

    # 라벨/메타 준비 (로컬 학습과 동일)
    label_csv = lcfg.LABEL_CSV_NEGBIO if lcfg.USE_LABEL == "negbio" else lcfg.LABEL_CSV_CHEXPERT
    label_table = load_label_table(str(label_csv), lcfg.LABEL_COLUMNS)
    meta_picker = build_image_picker_from_metadata(str(lcfg.METADATA_CSV), str(lcfg.IMG_ROOT))

    csv_path = _client_csv_path(cid)
    mode = _mode_for_client(cid)

    ds = ClientDataset(
        str(csv_path),
        label_table,
        mode,
        meta_picker,
        lcfg.TEXT_MODEL_NAME,
        lcfg.MAX_LEN,
        img_transform()
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=lcfg.NUM_WORKERS)
    return loader

@torch.no_grad()
def extract_reps_from_batch(model: MultiModalLateFusion, batch: dict, device: torch.device) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    MultiModalLateFusion 내부 인코더에서 임베딩만 추출
    반환: (img_rep[B, IMG_DIM] or None, txt_rep[B, TXT_DIM] or None)
    """
    pv   = batch.get("pixel_values", None)
    ids  = batch.get("input_ids", None)
    am   = batch.get("attention_mask", None)
    imsk = batch.get("img_mask", None)
    tmsk = batch.get("txt_mask", None)

    img_rep = txt_rep = None

    if pv is not None:
        pv = pv.to(device, non_blocking=True)
    if ids is not None:
        ids = ids.to(device, non_blocking=True)
    if am is not None:
        am = am.to(device, non_blocking=True)

    if pv is not None:
        zi = model.img_enc(pv)                # [B, IMG_DIM]
        if imsk is not None:
            imsk = imsk.to(device).to(zi.dtype)
            zi = zi * imsk                    # [B, IMG_DIM] (0 마스킹)
        img_rep = zi

    if ids is not None and am is not None:
        zt = model.txt_enc(ids, am)           # [B, TXT_DIM]
        if tmsk is not None:
            tmsk = tmsk.to(device).to(zt.dtype)
            zt = zt * tmsk
        txt_rep = zt

    return img_rep, txt_rep

@torch.no_grad()
def dump_train_reps(model: MultiModalLateFusion, loader: DataLoader, out_dir: str, max_batches: Optional[int] = None):
    img_buf, txt_buf = [], []
    for i, batch in enumerate(loader, 1):
        img_rep, txt_rep = extract_reps_from_batch(model, batch, model.device if hasattr(model, "device") else next(model.parameters()).device)
        if img_rep is not None:
            img_buf.append(img_rep.detach().cpu())
        if txt_rep is not None:
            txt_buf.append(txt_rep.detach().cpu())
        if max_batches and i >= max_batches:
            break

    if len(img_buf) > 0:
        img_np = torch.cat(img_buf, dim=0).numpy().astype("float32")
        np.save(os.path.join(out_dir, "train_img_reps.npy"), img_np)
        print("train_img_reps.npy 저장 완료")
    if len(txt_buf) > 0:
        txt_np = torch.cat(txt_buf, dim=0).numpy().astype("float32")
        np.save(os.path.join(out_dir, "train_txt_reps.npy"), txt_np)
        print("train_txt_reps.npy 저장 완료")


def run_for_client(cid: int, batch_size: int, device: torch.device, summary_csv: Optional[str]):
    cdir = client_dir(cid)
    ensure_dir(cdir)

    # (A) summary.csv → metrics.json
    if summary_csv and os.path.exists(summary_csv):
        _write_client_metric_from_summary(cid, summary_csv)
         # 파일 저장되는 위치 : 
         # C:\HJHJ0808\김희진\연구\졸업프로젝트\mimic-cxr\mimic-cxr\local_train_outputs\client_01
    else:
        print(f"[client_{cid}] summary.csv 경로가 없거나 찾지 못했습니다. (메트릭 JSON 생략)")

    # (B) reps 추출 저장
    model = build_model_from_ckpt(cid, device)
    loader = build_dataloader_for_reps(cid, batch_size)
    dump_train_reps(model, loader, cdir) # reps.npy 저장
    print(f"[client_{cid}] reps saved → {cdir}")

def parse_ids(text: str):
    text = text.strip()
    if "-" in text:
        a, b = text.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in text.split(",") if x]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cids", type=str, default="1-20", help='예: "1-20" 또는 "1,2,3"')
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--summary_csv", type=str,
                    default=r"C:\HJHJ0808\김희진\연구\졸업프로젝트\mimic-cxr\mimic-cxr\local_test_results\summary.csv")
    args = ap.parse_args()

    device = torch.device(args.device)
    ids = parse_ids(args.cids)
    for cid in ids:
        try:
            run_for_client(cid, args.batch, device, args.summary_csv)
        except Exception as e:
            print(f"[client_{cid}] ERROR:", e)

if __name__ == "__main__":
    main()
