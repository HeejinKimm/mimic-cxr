# global_train/utils_io.py  (UPDATED)

import os, json, random, csv
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from global_train.config import cfg

import numpy as np
import torch

# -------------------------------
# 프로젝트 표준 경로/파일
# -------------------------------
OUTPUTS_DIR   = Path("./outputs")
EVAL_SUMMARY  = Path("./eval_results/summary.csv")   # evaluate_all_clients_on_test.py가 저장
GLOBAL_DIR    = Path("./global_outputs")             # orchestrator 산출물 폴더

# -------------------------------
# 시드/경로 유틸
# -------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)

def client_dir(cid: int) -> Path:
    """ ./outputs/client_{cid:02d} """
    d = OUTPUTS_DIR / f"client_{cid:02d}"
    ensure_dir(d)
    return d

def ckpt_path(cid: int) -> Path:
    """
    해당 코드는 호환용 
    체크포인트 파일 경로를 추론:
    - 우선순위: best.pt
    - 그 외 과거/대체 네이밍도 일부 지원
    """
    base = client_dir(cid)
    candidates = [
        base / "best.pt",                         # 현재 train_local.py 저장 규칙
        base / f"client_{cid}_fusion_best.pt",    # 여분 호환
        base / f"client_{cid}_image_best.pt",
        base / f"client_{cid}_text_best.pt",
    ]
    for p in candidates:
        if p.exists():
            return p
    # 마지막으로 기본 경로 반환(없으면 이후 로더에서 FileNotFoundError)
    return candidates[0]

# -------------------------------
# 메트릭 로딩
# -------------------------------
def _read_metric_from_summary(cid: int, metric_name: str) -> Optional[float]:
    """
    eval_results/summary.csv에서 지정 metric 읽기.
    metric_name: 'macro_auroc' 또는 'loss' 권장.
    """
    if not EVAL_SUMMARY.exists():
        return None
    try:
        with open(EVAL_SUMMARY, "r", encoding="utf-8-sig") as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    if int(row["client_id"]) == cid and metric_name in row:
                        return float(row[metric_name])
                except Exception:
                    continue
    except Exception:
        pass
    return None

def _read_metric_from_json(cid: int, metric_name: str) -> Optional[float]:
    """
    outputs/client_xx/client_xx_metrics.json 에서 지정 metric 읽기.
    prep_clients.py가 저장하는 f1_micro/f1_macro/auc_macro/num_classes 등이 존재.
    """
    p = client_dir(cid) / f"client_{cid:02d}_metrics.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        if metric_name in d and isinstance(d[metric_name], (int, float)):
            return float(d[metric_name])
    except Exception:
        pass
    return None

def load_client_metric(cid: int,
                       prefer: List[str] = ("macro_auroc", "loss", "f1_macro", "auc_macro", "f1_micro")
                       ) -> float:
    """
    클라이언트 성능 스코어 하나를 로드(그룹핑/정렬용).
    우선순위: summary.csv의 macro_auroc → loss → (없으면) client_json의 f1_macro → auc_macro → f1_micro
    실패 시 np.nan
    """
    # summary.csv 우선
    for m in ("macro_auroc", "loss"):
        if m in prefer:
            v = _read_metric_from_summary(cid, m)
            if v is not None:
                return float(v)
    # per-client json 폴백
    for m in ("f1_macro", "auc_macro", "f1_micro"):
        if m in prefer:
            v = _read_metric_from_json(cid, m)
            if v is not None:
                return float(v)
    return float("nan")

# -------------------------------
# 임베딩 로딩
# -------------------------------
def _load_npy_if_exists(p: Path) -> np.ndarray:
    if not p.exists():
        return np.zeros((0, 0), dtype=np.float32)
    x = np.load(p)
    if x.ndim != 2:
        x = x.reshape(x.shape[0], -1)
    return x.astype(np.float32, copy=False)

def _l2norm_rows(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    return (x / n).astype(np.float32)

def get_client_reps(cid: int,
                    split: str = "train",      # 유지(과거 호환), 현재는 repr_*.npy만 사용
                    max_samples: int = 20000,
                    prefer_kd: bool = True
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    반환: (img_reps, txt_reps)  (없으면 빈 (0,0) 배열)
    - 우선 repr_*_kd.npy (학생에 대해서 KD 후 벡터가 생성됨)
    - 없으면 repr_img.npy / repr_txt.npy
    - 둘 다 없으면 (0,0)
    - 필요 시 max_samples로 다운샘플
    """
    base = client_dir(cid)
    # 파일 경로 결정(우선 KD)
    p_img = base / ("repr_img_kd.npy" if prefer_kd and (base / "repr_img_kd.npy").exists() else "repr_img.npy")
    p_txt = base / ("repr_txt_kd.npy" if prefer_kd and (base / "repr_txt_kd.npy").exists() else "repr_txt.npy")

    Xi = _load_npy_if_exists(p_img)
    Xt = _load_npy_if_exists(p_txt)

    # 다운샘플 (넘칠 때만)
    rng = np.random.RandomState(42)
    def _sample(x: np.ndarray) -> np.ndarray:
        if x.size == 0 or x.shape[0] <= max_samples:
            return x
        idx = rng.choice(x.shape[0], size=max_samples, replace=False)
        return x[idx]

    Xi = _l2norm_rows(_sample(Xi))
    Xt = _l2norm_rows(_sample(Xt))
    return Xi, Xt

# -------------------------------
# 직렬화/페이로드 저장
# -------------------------------
def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def save_payload_for_client(cid: int, payload: dict):
    """
    글로벌 페이로드는 텐서/넘파이를 포함하므로 JSON이 아닌 torch.save로 저장합니다.
    요약 정보만 별도의 JSON으로 같이 남겨두면 디버깅에 좋아요.
    """
    out_dir = os.path.join(cfg.OUT_GLOBAL_DIR, f"client_{cid}")
    ensure_dir(out_dir)

    # 1) 전체 payload는 바이너리로 저장
    bin_path = os.path.join(out_dir, "global_payload.pt")
    torch.save(payload, bin_path)

    # 2) 사람이 읽을 요약만 JSON으로 (큰 텐서는 제외)
    summary = {k: v for k, v in payload.items()
               if k not in {"Z", "Z_proxy_text", "Z_proxy_image"}}
    # 존재 여부 플래그만 기록
    summary["has_Z"] = bool(isinstance(payload.get("Z"), torch.Tensor))
    summary["has_Z_proxy_text"] = bool(isinstance(payload.get("Z_proxy_text"), torch.Tensor))
    summary["has_Z_proxy_image"] = bool(isinstance(payload.get("Z_proxy_image"), torch.Tensor))
    json_path = os.path.join(out_dir, "global_payload_summary.json")
    save_json(summary, json_path)