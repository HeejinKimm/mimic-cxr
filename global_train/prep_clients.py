# prep_clients.py
# # -*- coding: utf-8 -*-
"""
repr_kd.py (verbose logging)
- prep_clients가 만든 repr_img.npy / repr_txt.npy(또는 *_kd.npy)를 사용해
  교사-학생 KD를 수행하고, 상세 로그/리포트를 남깁니다.

실행:
  python -m global_train.repr_kd --alpha 0.5 --kteach 32 --verbose
"""

from __future__ import annotations
import os, json, csv, argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans

# 패키지 import (모듈 실행 권장: python -m global_train.repr_kd ...)
from .config import cfg
from .utils_io import ensure_dir, client_dir, load_client_metric, save_json, get_client_reps

# --------------------
# 기본 파라미터
# --------------------
DEFAULT_ALPHA  = 0.5
DEFAULT_KTEACH = 32

GLOBAL_OUT = Path("./global_outputs"); ensure_dir(str(GLOBAL_OUT))

# --------------------
# 로깅 유틸
# --------------------
_VERBOSE = False
def vlog(*args, **kwargs):
    if _VERBOSE:
        print(*args, **kwargs)

# --------------------
# KMeans / Blend
# --------------------
def _kmeans_centers(X: np.ndarray, k: int) -> np.ndarray:
    if X is None or X.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    Xu = np.unique(X, axis=0)
    k  = max(1, min(k, Xu.shape[0]))
    km = KMeans(n_clusters=k, n_init=10, random_state=cfg.SEED).fit(Xu)
    C  = km.cluster_centers_.astype(np.float32)
    # L2 정규화(안정화)
    C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-8)
    return C

def _blend_to_prototypes(X: np.ndarray, P: np.ndarray, alpha: float) -> np.ndarray:
    if X is None or X.size == 0 or P is None or P.size == 0:
        return X
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True)+1e-8)
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True)+1e-8)
    idx = np.argmax(Xn @ Pn.T, axis=1)
    return (1.0 - alpha) * X + alpha * P[idx]

# --------------------
# 그룹핑 (metric 기반)
# --------------------
def _read_metric_map(cids: List[int]) -> Dict[int, float]:
    """cfg.METRIC_NAME 기준 메트릭 맵 로드(없으면 NaN)"""
    m = {}
    for cid in cids:
        m[cid] = load_client_metric(cid)
    return m

def _split_high_low(cids: List[int], metrics: Dict[int, float], metric_name: str):
    """
    중앙값 기준으로 high/low 분할
      - metric_name == "loss"   → 낮을수록 good (high=좋은 쪽)
      - 그 외(예: f1_macro 등) → 높을수록 good
    """
    vals = [metrics[c] for c in cids if not np.isnan(metrics[c])]
    if not vals:
        return [], []
    med = float(np.median(np.array(vals)))
    if metric_name == "loss":
        high = [c for c in cids if not np.isnan(metrics[c]) and metrics[c] <  med]
        low  = [c for c in cids if not np.isnan(metrics[c]) and metrics[c] >= med]
    else:
        high = [c for c in cids if not np.isnan(metrics[c]) and metrics[c] >  med]
        low  = [c for c in cids if not np.isnan(metrics[c]) and metrics[c] <= med]
    return high, low

def _best_pair(a: int, b: int, metrics: Dict[int, float], metric_name: str) -> Tuple[int,int]:
    va, vb = metrics.get(a, np.nan), metrics.get(b, np.nan)
    if np.isnan(va) and np.isnan(vb): return (a, b)
    if np.isnan(vb): return (a, b)
    if np.isnan(va): return (b, a)
    if metric_name == "loss":
        # 낮을수록 teacher
        return (a, b) if va <= vb else (b, a)
    else:
        # 높을수록 teacher
        return (a, b) if va >= vb else (b, a)

# --------------------
# 리포트 기록기
# --------------------
def _append_report_rows(rows: List[Dict]):
    rep = GLOBAL_OUT / "kd_report.csv"
    write_header = not rep.exists()
    with open(rep, "a", newline="", encoding="utf-8-sig") as f:
        cols = [
            "modality",
            "teacher_cids",
            "student_cid",
            "n_teacher_img",
            "n_teacher_txt",
            "n_student_img",
            "n_student_txt",
            "K_used_img",
            "K_used_txt",
            "alpha",
            "out_img_kd",
            "out_txt_kd"
        ]
        w = csv.DictWriter(f, fieldnames=cols)
        if write_header: w.writeheader()
        for r in rows:
            w.writerow(r)

# --------------------
# 메인 KD 로직
# --------------------
def main():
    global _VERBOSE
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha",  type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--kteach", type=int,   default=DEFAULT_KTEACH)
    ap.add_argument("--max_teacher_samples", type=int, default=10_000)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    _VERBOSE = args.verbose

    # 1) metric 맵
    all_cids = list(cfg.FUSION_CLIENTS) + list(cfg.IMAGE_ONLY) + list(cfg.TEXT_ONLY)
    metrics  = _read_metric_map(all_cids)

    vlog("\n[Metric snapshot]")
    for cid in sorted(all_cids):
        vlog(f"  client_{cid:02d}: {cfg.METRIC_NAME}={metrics[cid]}")

    # 2) 그룹 나누기
    mm_high, mm_low = _split_high_low(list(cfg.FUSION_CLIENTS), metrics, cfg.METRIC_NAME)
    vlog(f"\n[Groups] Fusion high (teachers): {mm_high}")
    vlog(f"[Groups] Fusion low  (students): {mm_low}")

    if len(cfg.IMAGE_ONLY) == 2:
        img_T, img_S = _best_pair(cfg.IMAGE_ONLY[0], cfg.IMAGE_ONLY[1], metrics, cfg.METRIC_NAME)
        vlog(f"[Groups] Image-only teacher→student: {img_T} → {img_S}")
    else:
        img_T = img_S = None

    if len(cfg.TEXT_ONLY) == 2:
        txt_T, txt_S = _best_pair(cfg.TEXT_ONLY[0], cfg.TEXT_ONLY[1], metrics, cfg.METRIC_NAME)
        vlog(f"[Groups] Text-only  teacher→student: {txt_T} → {txt_S}")
    else:
        txt_T = txt_S = None

    # 3) JSON으로 그룹 스냅샷
    groups_json = {
        "metric_name": cfg.METRIC_NAME,
        "fusion": {"high": mm_high, "low": mm_low},
        "image_only": {"teacher": img_T, "student": img_S} if img_T is not None else {},
        "text_only":  {"teacher": txt_T, "student": txt_S} if txt_T is not None else {},
        "alpha": args.alpha, "kteach": args.kteach
    }
    (GLOBAL_OUT / "kd_groups.json").write_text(json.dumps(groups_json, indent=2, ensure_ascii=False), encoding="utf-8")
    vlog(f"\n[Write] {GLOBAL_OUT/'kd_groups.json'}")

    # 4) 멀티모달: high 풀(교사) → low(학생)
    rows = []
    if len(mm_high) and len(mm_low):
        # 교사 풀 수집
        XTi_img, XTi_txt = [], []
        for c in mm_high:
            xi, xt = get_client_reps(c, split="train", max_samples=args.max_teacher_samples)
            if xi.size: XTi_img.append(xi)
            if xt.size: XTi_txt.append(xt)
        XiT = np.vstack(XTi_img) if XTi_img else np.zeros((0,0), np.float32)
        XtT = np.vstack(XTi_txt) if XTi_txt else np.zeros((0,0), np.float32)
        vlog(f"\n[Fusion] teacher-pool size: img={XiT.shape} txt={XtT.shape}")

        P_img = _kmeans_centers(XiT, args.kteach)
        P_txt = _kmeans_centers(XtT, args.kteach)
        vlog(f"[Fusion] prototypes: P_img={P_img.shape}, P_txt={P_txt.shape}")

        # 학생군 처리
        for s in mm_low:
            base = client_dir(s); ensure_dir(base)
            xiS, xtS = get_client_reps(s, split="train", max_samples=10_000)
            out_img = out_txt = ""
            Ki = P_img.shape[0] if P_img.size else 0
            Kt = P_txt.shape[0] if P_txt.size else 0
            vlog(f"  [student {s:02d}] n_img={xiS.shape[0] if xiS.size else 0}, n_txt={xtS.shape[0] if xtS.size else 0}, K_img={Ki}, K_txt={Kt}")

            if xiS.size and P_img.size:
                xiS_kd = _blend_to_prototypes(xiS, P_img, args.alpha)
                out_img = os.path.join(base, "repr_img_kd.npy")
                np.save(out_img, xiS_kd.astype(np.float32))
            if xtS.size and P_txt.size:
                xtS_kd = _blend_to_prototypes(xtS, P_txt, args.alpha)
                out_txt = os.path.join(base, "repr_txt_kd.npy")
                np.save(out_txt, xtS_kd.astype(np.float32))

            rows.append({
                "modality": "fusion",
                "teacher_cids": ",".join(map(lambda x: f"{x:02d}", mm_high)),
                "student_cid": f"{s:02d}",
                "n_teacher_img": int(XiT.shape[0]),
                "n_teacher_txt": int(XtT.shape[0]),
                "n_student_img": int(xiS.shape[0]) if xiS.size else 0,
                "n_student_txt": int(xtS.shape[0]) if xtS.size else 0,
                "K_used_img": int(Ki),
                "K_used_txt": int(Kt),
                "alpha": float(args.alpha),
                "out_img_kd": out_img,
                "out_txt_kd": out_txt
            })

    # 5) 이미지 전용
    if 'img_T' in locals() and img_T is not None and img_S is not None:
        XiT, _ = get_client_reps(img_T, split="train", max_samples=args.max_teacher_samples)
        P = _kmeans_centers(XiT, args.kteach)
        XiS, _ = get_client_reps(img_S, split="train", max_samples=10_000)
        out_img = ""
        if XiS.size and P.size:
            XiS_kd = _blend_to_prototypes(XiS, P, args.alpha)
            out_img = os.path.join(client_dir(img_S), "repr_img_kd.npy")
            np.save(out_img, XiS_kd.astype(np.float32))
        vlog(f"\n[Image-only] {img_T:02d}→{img_S:02d}: nT_img={XiT.shape[0] if XiT.size else 0}, K={P.shape[0] if P.size else 0}, nS_img={XiS.shape[0] if XiS.size else 0}")
        rows.append({
            "modality": "image_only",
            "teacher_cids": f"{img_T:02d}",
            "student_cid": f"{img_S:02d}",
            "n_teacher_img": int(XiT.shape[0]) if XiT.size else 0,
            "n_teacher_txt": 0,
            "n_student_img": int(XiS.shape[0]) if XiS.size else 0,
            "n_student_txt": 0,
            "K_used_img": int(P.shape[0]) if P.size else 0,
            "K_used_txt": 0,
            "alpha": float(args.alpha),
            "out_img_kd": out_img,
            "out_txt_kd": ""
        })

    # 6) 텍스트 전용
    if 'txt_T' in locals() and txt_T is not None and txt_S is not None:
        _, XtT = get_client_reps(txt_T, split="train", max_samples=args.max_teacher_samples)
        P = _kmeans_centers(XtT, args.kteach)
        _, XtS = get_client_reps(txt_S, split="train", max_samples=10_000)
        out_txt = ""
        if XtS.size and P.size:
            XtS_kd = _blend_to_prototypes(XtS, P, args.alpha)
            out_txt = os.path.join(client_dir(txt_S), "repr_txt_kd.npy")
            np.save(out_txt, XtS_kd.astype(np.float32))
        vlog(f"\n[Text-only]  {txt_T:02d}→{txt_S:02d}: nT_txt={XtT.shape[0] if XtT.size else 0}, K={P.shape[0] if P.size else 0}, nS_txt={XtS.shape[0] if XtS.size else 0}")
        rows.append({
            "modality": "text_only",
            "teacher_cids": f"{txt_T:02d}",
            "student_cid": f"{txt_S:02d}",
            "n_teacher_img": 0,
            "n_teacher_txt": int(XtT.shape[0]) if XtT.size else 0,
            "n_student_img": 0,
            "n_student_txt": int(XtS.shape[0]) if XtS.size else 0,
            "K_used_img": 0,
            "K_used_txt": int(P.shape[0]) if P.size else 0,
            "alpha": float(args.alpha),
            "out_img_kd": "",
            "out_txt_kd": out_txt
        })

    # 7) 리포트 저장
    if rows:
        _append_report_rows(rows)
        vlog(f"\n[Write] {GLOBAL_OUT/'kd_report.csv'}")

    print("[KD] representation KD done (repr_*_kd.npy written where applicable)")
    if _VERBOSE:
        print(f"[KD] groups → {GLOBAL_OUT/'kd_groups.json'}")
        print(f"[KD] report → {GLOBAL_OUT/'kd_report.csv'}")


# --- add to: global_train/prep_clients.py ---

import os, glob, pandas as pd, torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from global_train.config import cfg
from local import utils_fusion as uf  # MMClientDataset, mm_collate, build_label_space

def _find_csv(base: str, cid: int, split: str, mod: str):
    """client_{cid} 폴더 안에서 CSV 자동 탐색"""
    d = os.path.join(base, f"client_{cid}")
    pats = [
        os.path.join(d, f"{split}_{mod}.csv"),
        os.path.join(d, f"{split}_{'image' if mod in ['image','img'] else 'text'}.csv"),
        os.path.join(d, f"client_{cid}_{split}_{mod}.csv"),
        os.path.join(d, f"*{split}*{mod}*.csv"),
        os.path.join(d, f"*{split}*{'image' if mod in ['image','img'] else 'text'}*.csv"),
    ]
    for p in pats:
        hits = sorted(glob.glob(p))
        if hits:
            return hits[0]
    return None

def _read_csv_or_none(path: str):
    return pd.read_csv(path) if (path and os.path.exists(path)) else None

def _guess_img_dirs(base: str, cid: int):
    """
    이미지 루트 디렉토리 자동 추정.
    uf.resolve_image_path()는 기본적으로 train 베이스를 쓰고
    id 문자열에 'valid'/'test'가 포함되면 해당 베이스로 바꿉니다.
    """
    d = os.path.join(base, f"client_{cid}")
    candidates = {
        "train": ["images_train", "train_images", "train", "images", "imgs"],
        "valid": ["images_valid", "valid_images", "val_images", "valid", "val"],
        "test":  ["images_test",  "test_images",  "test"],
    }
    out = {}
    for split, names in candidates.items():
        found = None
        for name in names:
            p = os.path.join(d, name)
            if os.path.isdir(p):
                if glob.glob(os.path.join(p, "*.jpg")) or glob.glob(os.path.join(p, "*.png")):
                    found = p; break
        out[split] = found if found else d
    return out  # {"train": "...", "valid": "...", "test": "..."}

def build_dataloaders(cid: int, batch_size: int):
    """
    반환: (train_loader, val_loader)
    - 배치 샘플 dict 키: pixel_values, input_ids, attention_mask, img_mask, txt_mask, labels
    - labels 크기는 cfg.NUM_CLASSES로 패딩/절단
    """
    base = cfg.BASE_DIR  # 예) ...\_prepared\

    # 1) CSV 탐색 (train / valid 각각 image, text)
    img_train_csv = _find_csv(base, cid, "train", "image") or _find_csv(base, cid, "train", "img")
    txt_train_csv = _find_csv(base, cid, "train", "text")  or _find_csv(base, cid, "train", "caption")
    img_val_csv   = (_find_csv(base, cid, "valid", "image") or _find_csv(base, cid, "valid", "img")
                     or _find_csv(base, cid, "val", "image") or _find_csv(base, cid, "val", "img"))
    txt_val_csv   = (_find_csv(base, cid, "valid", "text")  or _find_csv(base, cid, "valid", "caption")
                     or _find_csv(base, cid, "val", "text")  or _find_csv(base, cid, "val", "caption"))

    df_img_tr = _read_csv_or_none(img_train_csv)
    df_txt_tr = _read_csv_or_none(txt_train_csv)
    df_img_va = _read_csv_or_none(img_val_csv)
    df_txt_va = _read_csv_or_none(txt_val_csv)

    if df_img_tr is None and df_txt_tr is None:
        raise FileNotFoundError(
            f"[client_{cid}] train CSV를 찾을 수 없습니다.\n"
            f"예) client_{cid}/train_image.csv, train_text.csv, 혹은 client_{cid}_train_*.csv"
        )
    if df_img_va is None and df_txt_va is None:
        # val이 없으면 train을 복제(평가 지표는 약간 부정확할 수 있음)
        df_img_va, df_txt_va = df_img_tr, df_txt_tr

    # 2) 라벨 공간 구성
    label2idx, _ = uf.build_label_space(
        *(x for x in [df_img_tr, df_txt_tr, df_img_va, df_txt_va] if x is not None)
    )
    num_classes_found = len(label2idx)
    if num_classes_found == 0:
        # 최소 1클래스라도 보장
        label2idx = {"__dummy__": 0}
        num_classes_found = 1

    # 3) 토크나이저, 이미지 디렉토리 추정
    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    img_dirs = _guess_img_dirs(base, cid)

    # 4) Dataset
    ds_train = uf.MMClientDataset(df_img_tr, df_txt_tr, label2idx=label2idx, img_dirs=img_dirs, transform=None, max_len=128)
    ds_val   = uf.MMClientDataset(df_img_va, df_txt_va, label2idx=label2idx, img_dirs=img_dirs, transform=None, max_len=128)

    # 5) Collate: labels를 cfg.NUM_CLASSES 크기에 맞춰 패딩/절단
    def _collate(b):
        d = uf.mm_collate(b, tokenizer, max_len=128)  # dict(keys: pixel_values, input_ids, attention_mask, img_mask, txt_mask, labels)
        y = d["labels"]
        B, C = y.shape
        T = int(cfg.NUM_CLASSES)
        if C < T:
            pad = torch.zeros(B, T - C, dtype=y.dtype)
            d["labels"] = torch.cat([y, pad], dim=1)
        elif C > T:
            d["labels"] = y[:, :T]
        return d

    # 6) Loader
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=False, collate_fn=_collate)
    val_loader   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False, collate_fn=_collate)

    # (옵션) 외부에서 필요하면 접근하도록 속성으로 붙여둠
    train_loader.num_classes = num_classes_found
    val_loader.num_classes   = num_classes_found
    return train_loader, val_loader



if __name__ == "__main__":
    main()
