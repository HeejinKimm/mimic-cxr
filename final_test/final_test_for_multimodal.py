# final_test/final_test_for_multimodal.py
import os, re, json, argparse, warnings
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch

# ----------------------------
# Z / 클러스터 posterior layer
# ----------------------------
def _to_tensor(x, dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return x.to(dtype)
    return torch.tensor(x, dtype=dtype)

def _l2norm(t: torch.Tensor, eps=1e-12):
    return t / t.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)

def load_Z(z_dir: str, cid: int) -> torch.Tensor:
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
            for v in obj.values():
                if isinstance(v, torch.Tensor):
                    return v.view(-1).float()
            raise ValueError(f"[load_Z] dict에서 Z 키 탐색 실패: {p}")
        elif isinstance(obj, torch.Tensor):
            return obj.view(-1).float()
        else:
            raise TypeError(f"[load_Z] 지원하지 않는 타입: {type(obj)} in {p}")
    raise FileNotFoundError(f"[load_Z] Z 파일 없음. 확인: {cdir}")

def _load_npy_or_pt(path: str):
    if path.endswith(".npy"):
        return np.load(path)
    return torch.load(path, map_location="cpu")

def load_paired_centroids(global_dir: str, mapping_json: str) -> torch.Tensor:
    """
    img/txt centroid들을 pairs JSON으로 1:1 매칭해 'paired mean centroid' 생성.
    반환: C [K, D]
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
            raise ValueError("[load_paired_centroids] mapping json에 파일명이 없습니다.")
        img_path = os.path.join(base, img_file)
        txt_path = os.path.join(base, txt_file)
        if not (os.path.isfile(img_path) and os.path.isfile(txt_path)):
            raise FileNotFoundError(f"centroids 파일 없음: {img_path} / {txt_path}")
        img = _to_tensor(_load_npy_or_pt(img_path)).float()
        txt = _to_tensor(_load_npy_or_pt(txt_path)).float()
        if img.ndim == 2: img = img.mean(dim=0)
        if txt.ndim == 2: txt = txt.mean(dim=0)
        img_n = _l2norm(img.view(1, -1)).squeeze(0)
        txt_n = _l2norm(txt.view(1, -1)).squeeze(0)
        comb = _l2norm((img_n + txt_n).view(1, -1)).squeeze(0)
        C_list.append(comb)
    C = torch.stack(C_list, dim=0)  # [K, D]
    return C

def load_cluster_class_logits(global_dir: str, expect_num_classes: int = 13) -> torch.Tensor:
    """
    전역 테이블 M [K, C] 로드. 우선순위:
      clustering/cluster_class_logits.(npy|pt)
      clusters/combined/cluster_class_logits.(npy|pt)
      metrics_snapshot.json / kd_plan.json 의 class_logits|logits|prior_logits
    없으면 0행렬(K=0) 반환(그냥 통과).
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
            return _to_tensor(obj).float()

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
                if v is not None: rows.append(v)
        elif isinstance(clusters, dict):
            for _, item in sorted(clusters.items(), key=lambda kv: kv[0]):
                v = item.get("class_logits") or item.get("logits") or item.get("prior_logits")
                if v is not None: rows.append(v)
        if rows:
            M = _to_tensor(np.asarray(rows, dtype=np.float32)).float()
            if M.shape[1] != expect_num_classes:
                C = min(M.shape[1], expect_num_classes)
                M = M[:, :C]
            return M

    warnings.warn("[load_cluster_class_logits] 전역 테이블을 찾지 못해 0으로 대체합니다.")
    return torch.zeros((0, expect_num_classes), dtype=torch.float32)

def cluster_posterior_logits(Z: torch.Tensor,
                             centroids: torch.Tensor,
                             M: torch.Tensor,
                             tau: float = 10.0) -> torch.Tensor:
    if centroids.ndim != 2: 
        raise ValueError("centroids shape expected [K,D]")
    if M.ndim != 2:
        raise ValueError("M shape expected [K,C]")
    if centroids.size(0) != M.size(0):
        raise ValueError(f"K mismatch: centroids({centroids.size(0)}) vs M({M.size(0)})")
    Zn = _l2norm(Z.view(1, -1))
    Cn = _l2norm(centroids)
    sim = (Zn @ Cn.t()).squeeze(0)       # [K]
    alpha = torch.softmax(tau * sim, dim=-1)
    logits_Z = alpha @ M                 # [C]
    return logits_Z

def apply_cluster_posterior_layer(logits_base: torch.Tensor,
                                  Z: torch.Tensor,
                                  centroids: torch.Tensor,
                                  M: torch.Tensor,
                                  tau: float = 10.0,
                                  scale: float = 1.0) -> torch.Tensor:
    if M.size(0) == 0:
        return logits_base
    logits_Z = cluster_posterior_logits(Z, centroids, M, tau=tau)  # [C]
    return logits_base + scale * logits_Z.view(1, -1).expand_as(logits_base)

# ----------------------------
# 유틸
# ----------------------------
def parse_clients(s: str) -> List[int]:
    s = s.strip()
    if "-" in s and "," not in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    out = []
    for tok in re.split(r"[,\s]+", s):
        if not tok: continue
        if "-" in tok:
            a, b = tok.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(tok))
    return sorted(set(out))

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p

# ----------------------------
# (옵션1) 베이스 로짓: 파일에서 읽기
# ----------------------------
def load_base_logits_from_file(pattern: str, cid: int) -> torch.Tensor:
    """
    pattern 예: "C:/.../local_test_outputs/client_{cid:02d}_logits.npy"
    반환: [N, C] float32 torch.Tensor
    """
    path = pattern.format(cid=cid)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"[load_base_logits_from_file] 없음: {path}")
    arr = np.load(path)
    return _to_tensor(arr).float()

# ----------------------------
# (옵션2) 베이스 로짓: 모델로 직접 생성 (TODO 지점)
# ----------------------------
def build_model_and_loader_for_client(
    local_dir: str, cid: int, device: torch.device,
    test_csv: Optional[str] = None, batch: int = 256
) -> Tuple[torch.nn.Module, torch.utils.data.DataLoader]:
    """
    ⚠️ 프로젝트별로 구현 필요.
    - best.pt 경로: {local_dir}/client_{cid:02d}/best.pt
    - 여기서는 모형 코드 구조를 모르는 관계로 'NotImplementedError'를 냄.
    - 네 프로젝트의 기존 final_test 코드에서 모델/데이터 로더 부분을 이 함수에 복붙/이식하면 됨.
    """
    ckpt_dir = os.path.join(local_dir, f"client_{cid:02d}")
    ckpt = os.path.join(ckpt_dir, "best.pt")
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"[build_model_and_loader_for_client] best.pt 없음: {ckpt}")
    raise NotImplementedError(
        "여기에 네 프로젝트의 모델/데이터 로더 생성 코드를 이식해줘. "
        "모델.eval(), 모든 파라미터 requires_grad_(False) 후 반환하도록!"
    )

@torch.no_grad()
def infer_base_logits_with_model(model: torch.nn.Module,
                                 loader: torch.utils.data.DataLoader,
                                 device: torch.device) -> torch.Tensor:
    model.eval()
    outs = []
    for batch in loader:
        # ⚠️ 아래는 네 데이터셋 batch 구조에 맞게 수정 필요
        # 예: images, texts, labels = batch
        # logits = model(images.to(device), texts=..., ...)
        raise NotImplementedError("배치 → model forward 로짓 산출 코드를 네 구조에 맞게 작성해줘.")
    return torch.cat(outs, dim=0)  # [N, C]

# ----------------------------
# 메인
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", type=str, required=True, help="예: '1-16' or '1,4,7'")
    ap.add_argument("--global_dir", type=str, required=True, help=".../global_output")
    ap.add_argument("--z_dir", type=str, required=True, help=".../final_output/preparation")
    ap.add_argument("--out_dir", type=str, required=True, help="출력 루트")
    ap.add_argument("--num_classes", type=int, default=13)
    ap.add_argument("--tau", type=float, default=10.0, help="posterior 온도 파라미터")
    ap.add_argument("--scale", type=float, default=1.0, help="Z 로짓 합산 스케일")
    ap.add_argument("--mapping_json", type=str, default="img_txt_pairs.json",
                    help="global_output/clustering/ 아래의 pairs JSON 파일명")
    # 베이스 로짓 소스 선택
    ap.add_argument("--logits_pattern", type=str, default=None,
                    help="미리 저장된 베이스 로짓 파일 패턴 (예: '.../client_{cid:02d}_logits.npy')")
    ap.add_argument("--use_model", action="store_true",
                    help="best.pt를 로드해 직접 인퍼런스(프로젝트별 forward 구현 필요)")
    ap.add_argument("--local_dir", type=str, default=None,
                    help="--use_model일 때 best.pt가 들어있는 루트(예: .../local_train_outputs)")
    ap.add_argument("--test_csv", type=str, default=None,
                    help="--use_model일 때 테스트 CSV 경로(필요하면)")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    client_ids = parse_clients(args.clients)
    ensure_dir(args.out_dir)

    # 1) 전역 리소스 로딩 (초기 1회)
    C = load_paired_centroids(args.global_dir, args.mapping_json)              # [K, D]
    M = load_cluster_class_logits(args.global_dir, expect_num_classes=args.num_classes)  # [K, C]
    if C.size(0) != M.size(0):
        raise ValueError(f"K mismatch: centroids({C.size(0)}) vs M({M.size(0)})")

    # 2) 클라이언트별로 처리
    for cid in client_ids:
        # 2-1) 베이스 로짓 확보
        if args.logits_pattern:
            logits_base = load_base_logits_from_file(args.logits_pattern, cid)      # [N, C]
        elif args.use_model:
            if not args.local_dir:
                raise ValueError("--use_model 을 쓰려면 --local_dir 를 지정해야 합니다.")
            model, loader = build_model_and_loader_for_client(
                local_dir=args.local_dir, cid=cid, device=device,
                test_csv=args.test_csv, batch=args.batch
            )
            logits_base = infer_base_logits_with_model(model, loader, device)       # [N, C]
        else:
            raise ValueError("베이스 로짓 소스를 지정해 주세요: --logits_pattern 또는 --use_model")

        # 2-2) Z 로드 & posterior 레이어 적용 (학습 없음)
        Z = load_Z(args.z_dir, cid)                 # [D]
        logits_final = apply_cluster_posterior_layer(
            logits_base=logits_base,
            Z=Z,
            centroids=C,
            M=M,
            tau=args.tau,
            scale=args.scale
        )  # [N, C]

        # 2-3) 저장
        out_dir_client = ensure_dir(os.path.join(args.out_dir, f"client_{cid:02d}"))
        np.save(os.path.join(out_dir_client, "logits_base.npy"), logits_base.cpu().numpy())
        np.save(os.path.join(out_dir_client, "logits_final.npy"), logits_final.cpu().numpy())
        meta = {
            "cid": cid,
            "tau": args.tau,
            "scale": args.scale,
            "num_samples": int(logits_base.shape[0]),
            "num_classes": int(logits_base.shape[1]),
            "global_dir": args.global_dir,
            "z_dir": args.z_dir,
            "mapping_json": args.mapping_json
        }
        with open(os.path.join(out_dir_client, "posterior_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"[client_{cid:02d}] done  → {out_dir_client}")

if __name__ == "__main__":
    main()
