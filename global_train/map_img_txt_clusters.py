# -*- coding: utf-8 -*-
"""
map_img_txt_clusters.py
- 클러스터링 단계에서 저장한 그룹별 서브센트로이드(.npy)를 읽어서
  "이미지 클러스터 → 텍스트 클러스터" 매핑을 cross-attention 기반 코사인 유사도로 계산.
- 기본 산출물:
    - img_txt_similarity_matrix_i2t.npy/.csv  : 이미지→텍스트 기대 코사인
    - img_txt_similarity_matrix_t2i.npy/.csv  : 텍스트→이미지 기대 코사인
    - img_txt_similarity_matrix_symmetric.npy/.csv : 양방향 평균
    - img_to_txt_mapping.csv/.json            : (중복 허용) 이미지 그룹별 top-1 텍스트 그룹
- 옵션(--one_to_one):
    - img_to_txt_mapping_optimal.csv/.json    : (중복 없음) 1:1 최적 매칭 결과
- 보조 디버그:
    - details/img_group_to_txt_attention.json : 이미지 그룹별 상위 텍스트 그룹/스코어
    - details/txt_group_to_img_attention.json : 텍스트 그룹별 상위 이미지 그룹/스코어

사용 예시:
python -m global_train.map_img_txt_clusters \
  --cluster_dir "C:\\HJHJ0808\\김희진\\연구\\졸업프로젝트\\mimic-cxr\\mimic-cxr\\global_output\\clustering" \
  --groups 4 --tau 0.07 --one_to_one --score_type sym
"""

import os
import json
import csv
import argparse
import itertools
import numpy as np

from .config import cfg  # 프로젝트 컨피그 사용 (기본 경로 등)


# ---------------------------
# 유틸
# ---------------------------
def _ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # x: (N, D)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return x / n

def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)

def _write_csv(path: str, rows: list, header: list):
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerows(rows)

def _load_centroids(base_dir: str, prefix: str, k_groups: int):
    """
    prefix: "img_clientgroup" or "txt_clientgroup"
    반환: list[np.ndarray], 길이 = k_groups
          각 원소 shape = (k_sub, D) 또는 (0, D)
    """
    cents = []
    for g in range(k_groups):
        p = os.path.join(base_dir, f"{prefix}_{g}_centroids.npy")
        if not os.path.exists(p):
            cents.append(np.zeros((0, 1), np.float32))  # 비어있으면 0개 행
        else:
            cents.append(np.load(p).astype(np.float32))
    return cents


# ---------------------------
# 코사인 기반 cross-attention 점수
# ---------------------------
def _pair_score_cross_attention(C_img: np.ndarray, C_txt: np.ndarray, tau: float = 0.07):
    """
    한 이미지 그룹(서브센트로이드 m개)과 한 텍스트 그룹(서브센트로이드 n개)의
    cross-attention 코사인 유사도 스코어를 계산.
    - 코사인: S = Q K^T (Q,K는 L2-정규화)
    - 이미지→텍스트: A_i2t = softmax(S/tau) (행 기준), sim_i2t = mean_i sum_j A_ij * S_ij
    - 텍스트→이미지: A_t2i = softmax(S^T/tau) (행 기준), sim_t2i = mean_j sum_i A_ji * S_ij
    - 대칭 점수: sim_sym = (sim_i2t + sim_t2i)/2
    """
    if C_img.size == 0 or C_txt.size == 0:
        return 0.0, 0.0, 0.0

    Q = _l2_normalize(C_img)  # (m, D)
    K = _l2_normalize(C_txt)  # (n, D)

    S = Q @ K.T               # (m, n) cosine
    A_i2t = _softmax(S / tau, axis=1)             # (m, n)
    sim_i2t = float(np.mean(np.sum(A_i2t * S, axis=1)))

    S2 = K @ Q.T              # (n, m)
    A_t2i = _softmax(S2 / tau, axis=1)            # (n, m)
    sim_t2i = float(np.mean(np.sum(A_t2i * S2, axis=1)))

    sim_sym = 0.5 * (sim_i2t + sim_t2i)
    return sim_i2t, sim_t2i, sim_sym


# ---------------------------
# 1:1 최적 매칭 (헝가리안 대체: 4x4 등 소규모 완전탐색)
# ---------------------------
def _optimal_one_to_one_mapping(sim_matrix: np.ndarray):
    """
    sim_matrix: (G, G) — i2t 또는 sym 등 정사각 유사도 행렬
    반환:
      - pairs: list[(img_group, txt_group, score)]
      - total_score: float (최대합)
    """
    G_img, G_txt = sim_matrix.shape
    assert G_img == G_txt, "정사각 행렬에서만 1:1 매칭을 적용할 수 있습니다."
    G = G_img
    best_perm = None
    best_score = -1e18
    for perm in itertools.permutations(range(G)):
        score = 0.0
        for gi, tj in enumerate(perm):
            score += float(sim_matrix[gi, tj])
        if score > best_score:
            best_score = score
            best_perm = perm
    pairs = [(gi, int(best_perm[gi]), float(sim_matrix[gi, best_perm[gi]])) for gi in range(G)]
    return pairs, best_score


# ---------------------------
# 메인 로직
# ---------------------------
def run(cluster_dir: str, groups: int, tau: float, one_to_one: bool, score_type: str):
    # 출력 디렉토리
    out_dir = cluster_dir
    det_dir = os.path.join(out_dir, "details")
    _ensure_dir(det_dir)

    # 1) 그룹별 서브센트로이드 로드
    img_cents = _load_centroids(out_dir, "img_clientgroup", groups)  # list of (m_g, D_img)
    txt_cents = _load_centroids(out_dir, "txt_clientgroup", groups)  # list of (n_h, D_txt)

    # 2) 모든 (gi, tj) 쌍에 대해 cross-attention 코사인 스코어 계산
    sim_i2t = np.zeros((groups, groups), dtype=np.float32)
    sim_t2i = np.zeros((groups, groups), dtype=np.float32)
    sim_sym = np.zeros((groups, groups), dtype=np.float32)

    img_to_txt_details = []
    txt_to_img_details = []

    for gi in range(groups):
        for tj in range(groups):
            s_i2t, s_t2i, s_sym = _pair_score_cross_attention(img_cents[gi], txt_cents[tj], tau=tau)
            sim_i2t[gi, tj] = s_i2t
            sim_t2i[gi, tj] = s_t2i
            sim_sym[gi, tj] = s_sym

        # 이미지 그룹 gi 기준 상세 (상위 3개)
        order = np.argsort(-sim_i2t[gi])
        img_to_txt_details.append({
            "img_group": gi,
            "top_txt_groups": [
                {"txt_group": int(order[k]),
                 "sim_i2t": float(sim_i2t[gi, order[k]]),
                 "sim_sym": float(sim_sym[gi, order[k]])}
                for k in range(min(3, groups))
            ],
            "row_sim_i2t": sim_i2t[gi].tolist(),
            "row_sim_sym": sim_sym[gi].tolist(),
            "n_img_centroids": int(img_cents[gi].shape[0])
        })

    for tj in range(groups):
        order = np.argsort(-sim_t2i[:, tj])
        txt_to_img_details.append({
            "txt_group": tj,
            "top_img_groups": [
                {"img_group": int(order[k]),
                 "sim_t2i": float(sim_t2i[order[k], tj]),
                 "sim_sym": float(sim_sym[order[k], tj])}
                for k in range(min(3, groups))
            ],
            "col_sim_t2i": sim_t2i[:, tj].tolist(),
            "col_sim_sym": sim_sym[:, tj].tolist(),
            "n_txt_centroids": int(txt_cents[tj].shape[0])
        })

    # 3) 파일 저장: 유사도 행렬
    np.save(os.path.join(out_dir, "img_txt_similarity_matrix_i2t.npy"), sim_i2t)
    np.save(os.path.join(out_dir, "img_txt_similarity_matrix_t2i.npy"), sim_t2i)
    np.save(os.path.join(out_dir, "img_txt_similarity_matrix_symmetric.npy"), sim_sym)

    _write_csv(os.path.join(out_dir, "img_txt_similarity_matrix_i2t.csv"),
               [[gi, tj, float(sim_i2t[gi, tj])]
                for gi in range(groups) for tj in range(groups)],
               header=["img_group", "txt_group", "sim_i2t"])
    _write_csv(os.path.join(out_dir, "img_txt_similarity_matrix_t2i.csv"),
               [[gi, tj, float(sim_t2i[gi, tj])]
                for gi in range(groups) for tj in range(groups)],
               header=["img_group", "txt_group", "sim_t2i"])
    _write_csv(os.path.join(out_dir, "img_txt_similarity_matrix_symmetric.csv"),
               [[gi, tj, float(sim_sym[gi, tj])]
                for gi in range(groups) for tj in range(groups)],
               header=["img_group", "txt_group", "sim_sym"])

    # 4) (중복 허용) 이미지 그룹별 top-1 텍스트 그룹
    top1_rows = []
    top1_json = []
    for gi in range(groups):
        best_txt = int(np.argmax(sim_i2t[gi]))
        top1_rows.append([gi, best_txt, float(sim_i2t[gi, best_txt]), float(sim_sym[gi, best_txt])])
        top1_json.append({
            "img_group": gi,
            "mapped_txt_group": best_txt,
            "sim_i2t": float(sim_i2t[gi, best_txt]),
            "sim_sym": float(sim_sym[gi, best_txt])
        })
    _write_csv(os.path.join(out_dir, "img_to_txt_mapping.csv"),
               top1_rows, header=["img_group", "mapped_txt_group", "sim_i2t", "sim_sym"])
    with open(os.path.join(out_dir, "img_to_txt_mapping.json"), "w", encoding="utf-8") as f:
        json.dump({"tau": tau, "mapping": top1_json}, f, ensure_ascii=False, indent=2)

    # 5) (선택) 1:1 최적 매칭 — 중복 없는 매핑
    if one_to_one:
        sim_for_opt = sim_i2t if score_type == "i2t" else sim_sym
        pairs, total = _optimal_one_to_one_mapping(sim_for_opt)
        # csv/json 저장
        _write_csv(os.path.join(out_dir, "img_to_txt_mapping_optimal.csv"),
                   [[gi, tj, float(sim_i2t[gi, tj]), float(sim_sym[gi, tj])]
                    for gi, tj, _ in pairs],
                   header=["img_group", "mapped_txt_group", "sim_i2t", "sim_sym"])
        with open(os.path.join(out_dir, "img_to_txt_mapping_optimal.json"), "w", encoding="utf-8") as f:
            json.dump({
                "tau": tau,
                "score_type": score_type,
                "total_score": float(total),
                "mapping": [
                    {"img_group": int(gi),
                     "mapped_txt_group": int(tj),
                     "sim_i2t": float(sim_i2t[gi, tj]),
                     "sim_sym": float(sim_sym[gi, tj])}
                    for gi, tj, _ in pairs
                ]
            }, f, ensure_ascii=False, indent=2)
        print(f"[MAP] one-to-one optimal mapping ({score_type}), total={total:.6f} saved.")

    # 6) 디테일 저장
    with open(os.path.join(det_dir, "img_group_to_txt_attention.json"), "w", encoding="utf-8") as f:
        json.dump({"tau": tau, "details": img_to_txt_details}, f, ensure_ascii=False, indent=2)
    with open(os.path.join(det_dir, "txt_group_to_img_attention.json"), "w", encoding="utf-8") as f:
        json.dump({"tau": tau, "details": txt_to_img_details}, f, ensure_ascii=False, indent=2)

    print(f"[MAP] saved matrices & mappings in: {out_dir}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster_dir", type=str, default=None,
                    help="클러스터링 산출물 디렉토리 (기본: cfg.OUT_GLOBAL_DIR/clustering)")
    ap.add_argument("--groups", type=int, default=4, help="그룹 개수 (정사각 매칭 가정)")
    ap.add_argument("--tau", type=float, default=0.07, help="cross-attention softmax temperature")
    ap.add_argument("--one_to_one", action="store_true",
                    help="중복 없는 1:1 최적 매칭 결과도 함께 저장")
    ap.add_argument("--score_type", type=str, default="i2t", choices=["i2t", "sym"],
                    help="1:1 매칭에 사용할 점수(i2t 또는 sym)")
    return ap.parse_args()


def main():
    args = parse_args()
    cluster_dir = args.cluster_dir or os.path.join(cfg.OUT_GLOBAL_DIR, "clustering")
    run(cluster_dir=cluster_dir,
        groups=int(args.groups),
        tau=float(args.tau),
        one_to_one=bool(args.one_to_one),
        score_type=str(args.score_type))


if __name__ == "__main__":
    main()
