# local_gating/kd_utils.py
import os
import json
from typing import Tuple, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

# orchestrator가 저장한 설정을 참조 (형상/경로 등)
from global_train.config import cfg


# ----------------------------
# Loss utilities
# ----------------------------
def bce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Standard BCE with logits."""
    return F.binary_cross_entropy_with_logits(logits, labels)


def kd_logits_loss(student_logits: torch.Tensor,
                   teacher_logits: Optional[torch.Tensor],
                   T: float) -> torch.Tensor:
    """
    Logits KD (Hinton).  (T^2)*KL(soft_teacher || soft_student)
    teacher_logits가 없으면 0 반환.
    """
    if teacher_logits is None:
        return torch.zeros(1, device=student_logits.device, dtype=student_logits.dtype)
    p = F.log_softmax(student_logits / T, dim=-1)
    q = F.softmax(teacher_logits / T, dim=-1)
    return (T * T) * F.kl_div(p, q, reduction="batchmean")


def kd_repr_loss(R_local: torch.Tensor,
                 Z_global: Optional[torch.Tensor],
                 weight: float) -> torch.Tensor:
    """
    로컬 표현 R_local을 글로벌 벡터 Z_global에 정렬하는 MSE.
    - Z_global이 None이거나 weight<=0이면 0.
    - Z_global이 [d] 이면 배치에 맞게 [B,d]로 확장.
    - Z_global이 [B,d]면 그대로 MSE.
    """
    if Z_global is None or weight <= 0.0:
        dev = R_local.device if isinstance(R_local, torch.Tensor) else "cpu"
        return torch.zeros(1, device=dev, dtype=torch.float32)
    if Z_global.dim() == 1:
        Z_global = Z_global.unsqueeze(0).expand(R_local.size(0), -1)
    return weight * F.mse_loss(R_local, Z_global)


def compute_pos_weight(dataloader, num_labels: int) -> torch.Tensor:
    """
    BCEWithLogitsLoss의 pos_weight 계산.
    pos_weight[c] = (#neg_c) / (#pos_c),  pos가 전혀 없으면 1로 둠.
    """
    pos = torch.zeros(num_labels, dtype=torch.float64)
    neg = torch.zeros(num_labels, dtype=torch.float64)
    for b in dataloader:
        y = b["labels"].float()
        pos += y.sum(0)
        neg += (1.0 - y).sum(0)
    pw = torch.where(pos > 0, neg / (pos + 1e-8), torch.ones_like(neg))
    return pw.float()


# ----------------------------
# Payload loader
# ----------------------------
def _to_tensor_1d(x: Union[np.ndarray, list, torch.Tensor]) -> torch.Tensor:
    """
    입력이 [d] 또는 [K,d]일 수 있음. [K,d]면 평균해서 [d]로 변환.
    항상 float32 torch.Tensor로 반환.
    """
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().float()
    elif isinstance(x, (list, np.ndarray)):
        arr = torch.as_tensor(x, dtype=torch.float32)
    else:
        raise TypeError(f"Unsupported Z type: {type(x)}")

    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        # 여러 프로토타입이 온 경우 평균으로 풀링
        return arr.mean(dim=0)
    raise ValueError(f"Z must be 1D or 2D, got shape={tuple(arr.shape)}")


def _load_z_from_path(path: str) -> torch.Tensor:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Z path not found: {path}")
    z = np.load(path, allow_pickle=False)
    return _to_tensor_1d(z)


def load_payload_for_client(cid: int, map_location: str = "cpu"
                            ) -> Tuple[torch.Tensor, float, float, dict]:
    """
    orchestrator가 저장한 global_payload.json을 읽어
    - Z(또는 proxy Z) : torch.float32 ([d])  ← 2D면 평균으로 [d]로 압축
    - kd_temperature  : float
    - kd_rep_weight   : float
    - payload 전체 dict
    를 반환.

    우선순위:
      1) Z_proxy_text_path   (이미지 전용 클라이언트에 배포된 텍스트 기반 Z)
      2) Z_proxy_image_path  (텍스트 전용 클라이언트에 배포된 이미지 기반 Z)
      3) Z_path               (멀티모달 클라이언트용)
      4) (구버전 호환) Z_proxy_text / Z_proxy_image / Z (리스트/배열로 inline 저장된 경우)
    """
    payload_dir = os.path.join(cfg.OUT_GLOBAL_DIR, f"client_{cid:02d}")
    payload_path = os.path.join(payload_dir, "global_payload.json")
    if not os.path.exists(payload_path):
        raise FileNotFoundError(f"global payload not found: {payload_path}")

    with open(payload_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    # 경로 우선
    z = None
    for key in ("Z_proxy_text_path", "Z_proxy_image_path", "Z_path"):
        p = payload.get(key, None)
        if isinstance(p, str) and len(p):
            z = _load_z_from_path(p)
            break

    # 구버전 호환: inline 값이 있으면 사용
    if z is None:
        for key in ("Z_proxy_text", "Z_proxy_image", "Z"):
            val = payload.get(key, None)
            if val is not None:
                z = _to_tensor_1d(val)
                break

    if z is None:
        raise ValueError(f"Z not found for client {cid}: {payload.keys()}")

    # 하이퍼 파라미터 (없으면 config 기본값)
    T = float(payload.get("kd_temperature", cfg.KD_TEMP))
    W = float(payload.get("kd_rep_weight", cfg.KD_REP_WEIGHT))

    # map_location은 여기선 사용 X (로더가 텐서 반환)
    return z.float(), T, W, payload
