# final_test/z_layers/cluster_posterior.py
import os, json, warnings
import numpy as np
import torch

# -------------------- helpers --------------------
def _to_tensor(x, dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return x.to(dtype)
    return torch.tensor(x, dtype=dtype)

def _l2norm(t: torch.Tensor, eps=1e-12):
    return t / t.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)

def _load_npy_or_pt(path: str):
    if path.endswith(".npy"):
        return np.load(path)
    return torch.load(path, map_location="cpu")

# -------------------- Z loader --------------------
def load_Z(z_dir: str, cid: int):
    """
    {z_dir}/client_{cid:02d or cid}/ 에서 Z.* 파일을 찾아 1D 텐서 반환
    지원: Z.pt, global_Z.pt, Z.npy, global_payload.pt(dict)
    """
    cand_dirs = [os.path.join(z_dir, f"client_{cid:02d}"),
                 os.path.join(z_dir, f"client_{cid}")]
    cdir = next((d for d in cand_dirs if os.path.isdir(d)), z_dir)
    names = ["Z.pt", "global_Z.pt", "Z.npy", "global_payload.pt"]
    for nm in names:
        p = os.path.join(cdir, nm)
        if not os.path.isfile(p): 
            continue
        if p.endswith(".npy"):
            return _to_tensor(np.load(p)).view(-1)
        obj = torch.load(p, map_location="cpu")
        if isinstance(obj, dict):
            for k in ("Z","z","Z_proxy","Z_proxy_image","Z_proxy_text"):
                if k in obj:
                    return _to_tensor(obj[k]).view(-1)
            # dict 안의 tensor 하나라도 꺼내기(보수적)
            for v in obj.values():
                if isinstance(v, torch.Tensor):
                    return v.view(-1).float()
            raise ValueError(f"[load_Z] dict에서 Z 키를 못 찾음: {p}")
        elif isinstance(obj, torch.Tensor):
            return obj.view(-1).float()
        else:
            raise TypeError(f"[load_Z] 지원하지 않는 타입: {type(obj)} in {p}")
    raise FileNotFoundError(f"[load_Z] Z 파일을 찾지 못함. 확인 디렉토리: {cdir}")

# -------------------- centroids loader --------------------
def load_paired_centroids_from_clustering(global_dir: str, mapping_json: str):
    """
    img/txt 클러스터 centroid를 매핑 JSON의 pairs로 1:1 매칭하여
    'paired mean centroid'를 만든다.
    파일 예:
      {global_dir}/clustering/img_clientgroup_{i}_centroids.npy
      {global_dir}/clustering/txt_clientgroup_{j}_centroids.npy
      mapping_json: { "pairs":[
        {"img_group":0, "txt_group":0, "img_centroids_file":"img_clientgroup_0_centroids.npy",
                                      "txt_centroids_file":"txt_clientgroup_0_centroids.npy"},
        ...
      ]}
    반환:
      C [K, D] torch.float32  (각 pair마다 img/txt centroid를 평균하여 만든 combined centroid)
    """
    base = os.path.join(global_dir, "clustering")
    with open(os.path.join(base, mapping_json), "r", encoding="utf-8") as f:
        j = json.load(f)
    pairs = j["pairs"]
    C_list = []
    for p in pairs:
        img_file = p.get("img_centroids_file")
        txt_file = p.get("txt_centroids_file")
        if not img_file or not txt_file:
            raise ValueError("[load_paired_centroids_from_clustering] mapping json에 파일명이 없습니다.")
        img_path = os.path.join(base, img_file)
        txt_path = os.path.join(base, txt_file)
        if not (os.path.isfile(img_path) and os.path.isfile(txt_path)):
            raise FileNotFoundError(f"centroids 파일을 찾지 못함: {img_path} / {txt_path}")

        img = _load_npy_or_pt(img_path)  # [D] or [K_i, D]?
        txt = _load_npy_or_pt(txt_path)
        # 파일들이 한 그룹의 centroid 하나만 담고 있다고 가정([D]).
        # 혹시 [K_i, D] 형태면 평균으로 하나 만들기.
        img = _to_tensor(img).float()
        txt = _to_tensor(txt).float()
        if img.ndim == 2: img = img.mean(dim=0)
        if txt.ndim == 2: txt = txt.mean(dim=0)

        # L2 정규화 후 평균 → 다시 정규화 (paired mean centroid)
        img_n = _l2norm(img.view(1, -1)).squeeze(0)  # [D]
        txt_n = _l2norm(txt.view(1, -1)).squeeze(0)
        comb = _l2norm((img_n + txt_n).view(1, -1)).squeeze(0)   # [D]
        C_list.append(comb)
    C = torch.stack(C_list, dim=0)  # [K, D]
    return C

# -------------------- cluster->class logits loader --------------------
def load_cluster_class_logits(global_dir: str, expect_num_classes: int = 13):
    """
    전역 테이블 M [K, C] 로드.
    우선순위:
      1) {global_dir}/clustering/cluster_class_logits.(npy|pt)
      2) {global_dir}/clusters/combined/cluster_class_logits.(npy|pt)
      3) {global_dir}/metrics_snapshot.json  or  kd_plan.json
    없으면 0행렬을 반환(경고).
    """
    bin_candidates = [
        os.path.join(global_dir, "clustering", "cluster_class_logits.npy"),
        os.path.join(global_dir, "clustering", "cluster_class_logits.pt"),
        os.path.join(global_dir, "clusters", "combined", "cluster_class_logits.npy"),
        os.path.join(global_dir, "clusters", "combined", "cluster_class_logits.pt"),
    ]
    for p in bin_candidates:
        if os.path.isfile(p):
            obj = _load_npy_or_pt(p)
            M = _to_tensor(obj).float()
            return M

    # JSON 파싱
    for nm in ("metrics_snapshot.json", "kd_plan.json"):
        p = os.path.join(global_dir, nm)
        if not os.path.isfile(p): 
            continue
        with open(p, "r", encoding="utf-8") as f:
            j = json.load(f)
        clusters = j.get("clusters")
        if clusters is None: 
            continue
        rows = []
        if isinstance(clusters, list):
            for item in clusters:
                v = item.get("class_logits") or item.get("logits") or item.get("prior_logits")
                if v is not None:
                    rows.append(v)
        elif isinstance(clusters, dict):
            for _, item in sorted(clusters.items(), key=lambda kv: kv[0]):
                v = item.get("class_logits") or item.get("logits") or item.get("prior_logits")
                if v is not None:
                    rows.append(v)
        if rows:
            M = _to_tensor(np.asarray(rows, dtype=np.float32)).float()
            # 클래스 수 맞춤
            if M.shape[1] != expect_num_classes:
                C = min(M.shape[1], expect_num_classes)
                M = M[:, :C]
            return M

    warnings.warn("[load_cluster_class_logits] 전역 테이블을 찾지 못해 0으로 대체합니다.")
    return torch.zeros((0, expect_num_classes), dtype=torch.float32)  # K=0 → 추후 안전 처리

# -------------------- posterior layer --------------------
def cluster_posterior_logits(Z: torch.Tensor,
                             centroids: torch.Tensor,
                             M: torch.Tensor,
                             tau: float = 10.0) -> torch.Tensor:
    """
    alpha = softmax( tau * cos(Z, C_k) )
    logits_Z = alpha^T * M
    Z: [D], centroids: [K, D], M: [K, C]
    """
    if centroids.ndim != 2: 
        raise ValueError("centroids shape expected [K,D]")
    if M.ndim != 2:
        raise ValueError("M shape expected [K,C]")
    if centroids.size(0) != M.size(0):
        raise ValueError(f"K mismatch: centroids({centroids.size(0)}) vs M({M.size(0)})")

    Zn = _l2norm(Z.view(1, -1))          # [1,D]
    Cn = _l2norm(centroids)              # [K,D]
    sim = (Zn @ Cn.t()).squeeze(0)       # [K]
    alpha = torch.softmax(tau * sim, dim=-1)  # [K]
    logits_Z = alpha @ M                 # [C]
    return logits_Z

def apply_cluster_posterior_layer(logits_base: torch.Tensor,
                                  Z: torch.Tensor,
                                  centroids: torch.Tensor,
                                  M: torch.Tensor,
                                  tau: float = 10.0,
                                  scale: float = 1.0) -> torch.Tensor:
    """
    logits_base: [B,C]
    반환: [B,C] = logits_base + scale * logits_Z  (logits_Z는 [C]를 배치로 확장)
    """
    if M.size(0) == 0:  # 테이블이 없으면 그냥 통과
        return logits_base
    logits_Z = cluster_posterior_logits(Z, centroids, M, tau=tau)  # [C]
    return logits_base + scale * logits_Z.view(1, -1).expand_as(logits_base)
