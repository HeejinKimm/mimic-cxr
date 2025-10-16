# -*- coding: utf-8 -*-
import os, json, random
import numpy as np
import torch
from typing import Optional, Tuple
from .config import cfg

# ----- basic utils -----
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def ensure_dir(p: str): os.makedirs(p, exist_ok=True)
def client_dir(cid: int) -> str: return os.path.join(cfg.BASE_DIR, f"client_{cid:02d}")
def ckpt_path(cid: int) -> str:  return os.path.join(client_dir(cid), cfg.CKPT_NAME)

# ----- metrics -----
def load_client_metric(cid: int) -> Optional[float]:
    """client_{cid}_metrics.json 에서 cfg.METRIC_NAME 값을 읽음 (없으면 NaN)"""
    path = os.path.join(client_dir(cid), f"client_{cid}_metrics.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        try:
            return float(d.get(cfg.METRIC_NAME, np.nan))
        except Exception:
            return np.nan
    return np.nan

# ----- reps loader -----
def get_client_reps(cid: int, split: str = "train", max_samples: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
    """
    반환: (img_reps[N_i, IMG_DIM], txt_reps[N_t, TXT_DIM])
    - prep_clients.py 가 저장한 train_img_reps.npy / train_txt_reps.npy 를 읽음
    """
    base = client_dir(cid)
    img_path = os.path.join(base, f"{split}_img_reps.npy")
    txt_path = os.path.join(base, f"{split}_txt_reps.npy")
    if not (os.path.exists(img_path) or os.path.exists(txt_path)):
        raise FileNotFoundError(f"[client_{cid}] '{split}_img_reps.npy' 또는 '{split}_txt_reps.npy'가 없습니다: {base}")

    img = np.load(img_path) if os.path.exists(img_path) else np.zeros((0, cfg.IMG_DIM), dtype=np.float32)
    txt = np.load(txt_path) if os.path.exists(txt_path) else np.zeros((0, cfg.TXT_DIM), dtype=np.float32)

    def sample_rows(x, k):
        if len(x) <= k: return x
        idx = np.random.choice(len(x), size=k, replace=False)
        return x[idx]

    return sample_rows(img, max_samples), sample_rows(txt, max_samples)

# ----- persistence -----
def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f: json.dump(obj, f, indent=2, ensure_ascii=False)

def save_payload_for_client(cid: int, payload: dict):
    out_dir = os.path.join(cfg.OUT_GLOBAL_DIR, f"client_{cid:02d}")
    ensure_dir(out_dir)
    torch.save(payload, os.path.join(out_dir, "global_payload.pt"))
