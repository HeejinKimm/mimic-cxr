# orchestrator.py (DROP-IN, safe & verbose)
import os
import numpy as np
from typing import Dict, List, Tuple

from .config import cfg
from .utils_io import (
    set_seed, ensure_dir, load_client_metric,
    save_json, save_payload_for_client
)
from .build_vectors import (
    stack_reps, kmeans_centroids, build_global_vectors
)

def _higher_is_better(metric_name: str) -> bool:
    """
    f1/auc 류는 높을수록 좋음, loss 류는 낮을수록 좋음.
    cfg.METRIC_NAME 기준으로 판단.
    """
    name = (metric_name or "").lower()
    if "loss" in name:
        return False
    # f1, auc, acc 등은 높을수록 좋다 가정
    return True

def _split_median(metrics: Dict[int, float], cids: List[int], higher_is_better: bool):
    vals = [metrics.get(c, np.nan) for c in cids]
    xs   = [v for v in vals if not np.isnan(v)]
    if not xs:
        return [], []
    med = float(np.median(np.array(xs)))
    if higher_is_better:
        high = [c for c in cids if not np.isnan(metrics.get(c, np.nan)) and metrics[c] >  med]
        low  = [c for c in cids if not np.isnan(metrics.get(c, np.nan)) and metrics[c] <= med]
    else:
        high = [c for c in cids if not np.isnan(metrics.get(c, np.nan)) and metrics[c] <  med]
        low  = [c for c in cids if not np.isnan(metrics.get(c, np.nan)) and metrics[c] >= med]
    return high, low

def _best_pair(a: int, b: int, metrics: Dict[int, float], higher_is_better: bool):
    va, vb = metrics.get(a, np.nan), metrics.get(b, np.nan)
    if np.isnan(va) and np.isnan(vb): return None
    if np.isnan(vb): return (a, b)
    if np.isnan(va): return (b, a)
    if higher_is_better:
        return (a, b) if va >= vb else (b, a)
    else:
        return (a, b) if va <= vb else (b, a)

def _build_fusion_kd_edges(high: List[int], low: List[int], metrics: Dict[int, float],
                           k_per_student: int = 2, higher_is_better: bool = True):
    key = (lambda c: metrics.get(c, float("-inf"))) if higher_is_better else (lambda c: -metrics.get(c, float("inf")))
    high_sorted = sorted(high, key=key, reverse=True)
    edges = []
    for s in low:
        teachers = high_sorted[:k_per_student] if len(high_sorted) >= k_per_student else high_sorted
        for t in teachers:
            edges.append((t, s))
    return edges

def main():
    set_seed(cfg.SEED)
    ensure_dir(cfg.OUT_GLOBAL_DIR)

    # 1) 성능지표 로드 (모든 관련 클라이언트)
    cids_all = list(cfg.FUSION_CLIENTS) + list(cfg.IMAGE_ONLY) + list(cfg.TEXT_ONLY)
    metrics = {cid: load_client_metric(cid) for cid in cids_all}
    save_json({"metric_name": cfg.METRIC_NAME, "metrics": metrics},
              os.path.join(cfg.OUT_GLOBAL_DIR, cfg.METRICS_SNAPSHOT))

    hib = _higher_is_better(cfg.METRIC_NAME)

    # 2) FUSION 그룹(중앙값 기준 상/하)
    fusion_high, fusion_low = _split_median(metrics, list(cfg.FUSION_CLIENTS), higher_is_better=hib)

    # 3) KD 간선
    kd_edges = {
        "fusion": _build_fusion_kd_edges(fusion_high, fusion_low, metrics,
                                         k_per_student=2, higher_is_better=hib),
        "image_only": [],
        "text_only": []
    }
    # 이미지 전용
    if len(cfg.IMAGE_ONLY) == 2:
        bp = _best_pair(cfg.IMAGE_ONLY[0], cfg.IMAGE_ONLY[1], metrics, higher_is_better=hib)
        if bp: kd_edges["image_only"].append(bp)
    # 텍스트 전용
    if len(cfg.TEXT_ONLY) == 2:
        bp = _best_pair(cfg.TEXT_ONLY[0], cfg.TEXT_ONLY[1], metrics, higher_is_better=hib)
        if bp: kd_edges["text_only"].append(bp)

    # 4) 글로벌 벡터: 멀티모달 전체 reps → KMeans → 헝가리안 매칭 + CrossAttention
    print(f"[GLOBAL] stacking reps from fusion clients: {list(cfg.FUSION_CLIENTS)}")
    img_all, txt_all = stack_reps(list(cfg.FUSION_CLIENTS), split="train", max_samples=cfg.SAMPLE_PER_CLIENT)

    print(f"[GLOBAL] stacked img reps: {tuple(img_all.shape) if isinstance(img_all, np.ndarray) else None}, "
          f"txt reps: {tuple(txt_all.shape) if isinstance(txt_all, np.ndarray) else None}")

    # 안전 가드: 임베딩이 비어있으면 제로 Z를 배포
    if not isinstance(img_all, np.ndarray) or img_all.size == 0 \
       or not isinstance(txt_all, np.ndarray) or txt_all.size == 0:
        print("[WARN] empty reps detected. Distributing zero global vectors.")
        import torch
        Zs = {
            "Z_mm":      torch.zeros(cfg.D_MODEL, dtype=torch.float32),
            "Z_img2txt": torch.zeros(cfg.D_MODEL, dtype=torch.float32),
            "Z_txt2img": torch.zeros(cfg.D_MODEL, dtype=torch.float32),
        }
    else:
        img_cent = kmeans_centroids(img_all, cfg.K_IMG)
        txt_cent = kmeans_centroids(txt_all, cfg.K_TXT)

        print(f"[GLOBAL] centroids -> img: {tuple(img_cent.shape)}, txt: {tuple(txt_cent.shape)}")

        # 또 한 번 가드
        if img_cent.size == 0 or txt_cent.size == 0:
            print("[WARN] empty centroids. Distributing zero global vectors.")
            import torch
            Zs = {
                "Z_mm":      torch.zeros(cfg.D_MODEL, dtype=torch.float32),
                "Z_img2txt": torch.zeros(cfg.D_MODEL, dtype=torch.float32),
                "Z_txt2img": torch.zeros(cfg.D_MODEL, dtype=torch.float32),
            }
        else:
            if img_cent.shape[0] != txt_cent.shape[0]:
                print(f"[warn] K mismatch: K_IMG={img_cent.shape[0]} vs K_TXT={txt_cent.shape[0]} → min(K)로 매칭")
            Zs = build_global_vectors(img_cent, txt_cent)  # torch.Tensor dict

    # 5) 계획 저장
    kd_plan = {
        "metric_name": cfg.METRIC_NAME,
        "higher_is_better": hib,
        "kd_rep_weight": cfg.KD_REP_WEIGHT,
        "kd_temperature": cfg.KD_TEMP,
        "fusion_groups": {"high": fusion_high, "low": fusion_low},
        "kd_edges": kd_edges,
        "image_only": list(cfg.IMAGE_ONLY),
        "text_only": list(cfg.TEXT_ONLY),
        "K_img": cfg.K_IMG, "K_txt": cfg.K_TXT,
        "d_model": cfg.D_MODEL,
        "sample_per_client": cfg.SAMPLE_PER_CLIENT,
    }
    save_json(kd_plan, os.path.join(cfg.OUT_GLOBAL_DIR, cfg.KD_PLAN_JSON))

    # 6) 클라이언트별 payload 저장
    fusion_teachers = {t for (t, s) in kd_edges["fusion"]}
    fusion_students = {s for (t, s) in kd_edges["fusion"]}
    teachers_by_student = {}
    for (t, s) in kd_edges["fusion"]:
        teachers_by_student.setdefault(s, []).append(t)

    # FUSION
    for cid in cfg.FUSION_CLIENTS:
        role = "teacher" if cid in fusion_teachers else ("student" if cid in fusion_students else "solo")
        payload = {
            "client_id": cid,
            "type": "fusion",
            "Z": Zs["Z_mm"],                          # torch.Tensor
            "d_global": cfg.D_MODEL,
            "kd_rep_weight": cfg.KD_REP_WEIGHT,
            "kd_temperature": cfg.KD_TEMP,
            "role": role,
            "teacher_ids": teachers_by_student.get(cid, []),
            "group": "high" if cid in fusion_high else ("low" if cid in fusion_low else "unknown"),
            "metric": metrics.get(cid, None),
        }
        save_payload_for_client(cid, payload)

    # IMAGE ONLY
    tpair = kd_edges["image_only"][0] if len(kd_edges["image_only"]) else None
    for cid in cfg.IMAGE_ONLY:
        t_for_me = (tpair[0] if tpair and tpair[1] == cid else None)
        payload = {
            "client_id": cid,
            "type": "image_only",
            "Z_proxy_text": Zs["Z_img2txt"],         # torch.Tensor
            "d_global": cfg.D_MODEL,
            "role": "student" if t_for_me is not None else "teacher",
            "teacher_ids": [t_for_me] if t_for_me is not None else [],
            "kd_rep_weight": cfg.KD_REP_WEIGHT,
            "kd_temperature": cfg.KD_TEMP,
            "metric": metrics.get(cid, None),
        }
        save_payload_for_client(cid, payload)

    # TEXT ONLY
    tpair = kd_edges["text_only"][0] if len(kd_edges["text_only"]) else None
    for cid in cfg.TEXT_ONLY:
        t_for_me = (tpair[0] if tpair and tpair[1] == cid else None)
        payload = {
            "client_id": cid,
            "type": "text_only",
            "Z_proxy_image": Zs["Z_txt2img"],        # torch.Tensor
            "d_global": cfg.D_MODEL,
            "role": "student" if t_for_me is not None else "teacher",
            "teacher_ids": [t_for_me] if t_for_me is not None else [],
            "kd_rep_weight": cfg.KD_REP_WEIGHT,
            "kd_temperature": cfg.KD_TEMP,
            "metric": metrics.get(cid, None),
        }
        save_payload_for_client(cid, payload)

    print("[GLOBAL] Done ->", cfg.OUT_GLOBAL_DIR)
    print("Fusion groups:", {"high": fusion_high, "low": fusion_low})
    print("KD edges:", kd_edges)

if __name__ == "__main__":
    main()
