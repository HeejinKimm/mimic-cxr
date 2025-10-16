# -*- coding: utf-8 -*-
"""
build_final_output_z.py
- IMG(쿼리) × TXT(키/값) Cross-Attention Transformer로 Z 4개 생성
- 입력:
    * clustering 디렉토리:
        - img_clientgroup_{g}_centroids.npy
        - txt_clientgroup_{h}_centroids.npy
        - img_to_txt_mapping_optimal.json (있으면 우선)
          또는 img_to_txt_mapping.json
- 출력:
    * {OUT_GLOBAL_DIR}/final_output_z/
        - Z_img{gi}_txt{tj}.npy (총 4개)
        - final_output_z_vectors.npz (4개 Z를 한 번에)
        - final_output_z_index.json (하이퍼파라미터/매핑 메타)
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn

from .config import cfg  # IMG_DIM, TXT_DIM, D_MODEL, OUT_GLOBAL_DIR 등을 사용


# =========================
# Cross-Attention modules
# =========================
class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1, ffn_mult: int = 4):
        super().__init__()
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        hidden = d_model * ffn_mult
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, key_padding_mask=None):
        qn = self.ln_q(q)
        kn = self.ln_kv(k)
        vn = self.ln_kv(v)
        attn_out, _ = self.attn(qn, kn, vn, key_padding_mask=key_padding_mask, need_weights=False)
        x = q + self.dropout(attn_out)
        x = x + self.ffn(x)
        return x


class CrossAttentionFuser(nn.Module):
    """
    이미지 토큰(Q)과 텍스트 토큰(K/V)을 교차어텐션으로 융합해 (B, d_model) Z를 생성.
    """
    def __init__(self,
                 img_dim: int,
                 txt_dim: int,
                 d_model: int = 256,
                 n_heads: int = 8,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 use_cls: bool = False,
                 ffn_mult: int = 4):
        super().__init__()
        self.use_cls = use_cls
        self.proj_img = nn.Linear(img_dim, d_model, bias=False)
        self.proj_txt = nn.Linear(txt_dim, d_model, bias=False)

        if use_cls:
            self.cls_q = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_q, std=0.02)

        self.blocks = nn.ModuleList([
            CrossAttentionBlock(d_model, n_heads=n_heads, dropout=dropout, ffn_mult=ffn_mult)
            for _ in range(num_layers)
        ])
        self.final_ln = nn.LayerNorm(d_model)

    @torch.no_grad()
    def encode(self, img_tokens: torch.Tensor, txt_tokens: torch.Tensor, txt_pad_mask=None) -> torch.Tensor:
        """
        img_tokens: (Q, img_dim) or (B, Q, img_dim)
        txt_tokens: (K, txt_dim) or (B, K, txt_dim)
        return: Z -> (B, d_model)
        """
        if img_tokens.dim() == 2:
            img_tokens = img_tokens.unsqueeze(0)  # (1,Q,D)
        if txt_tokens.dim() == 2:
            txt_tokens = txt_tokens.unsqueeze(0)  # (1,K,D)

        qi = self.proj_img(img_tokens)
        kv = self.proj_txt(txt_tokens)

        if self.use_cls:
            cls_q = self.cls_q.repeat(qi.size(0), 1, 1)
            q = torch.cat([cls_q, qi], dim=1)
        else:
            q = qi

        k = kv
        v = kv

        for blk in self.blocks:
            q = blk(q, k, v, key_padding_mask=txt_pad_mask)

        q = self.final_ln(q)
        if self.use_cls:
            z = q[:, 0]              # (B,d)
        else:
            z = q.mean(dim=1)        # (B,d)
        return z


# =========================
# Helpers
# =========================
def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _pick_mapping(cluster_dir: str):
    """
    우선순위: img_to_txt_mapping_optimal.json -> img_to_txt_mapping.json
    반환: [{"img_group": gi, "mapped_txt_group": tj, ...}, ...]
    """
    opt = os.path.join(cluster_dir, "img_to_txt_mapping_optimal.json")
    top = os.path.join(cluster_dir, "img_to_txt_mapping.json")
    if os.path.exists(opt):
        data = _load_json(opt)
        return data["mapping"], "optimal"
    if os.path.exists(top):
        data = _load_json(top)
        return data["mapping"], "top1"
    raise FileNotFoundError("매핑 파일이 없습니다. map_img_txt_clusters 를 먼저 실행하세요.")

def _load_centroids(path: str) -> np.ndarray:
    if not os.path.exists(path):
        return np.zeros((0, 1), np.float32)
    return np.load(path).astype(np.float32)

def _ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def _infer_dim_from_centroids(cluster_dir: str, prefix: str, groups: int, fallback_dim: int) -> int:
    """
    prefix: "img_clientgroup" or "txt_clientgroup"
    군집 파일을 훑어 첫 번째 비어있지 않은 파일의 shape[1]을 차원으로 사용.
    없으면 fallback_dim 반환.
    """
    for g in range(groups):
        p = os.path.join(cluster_dir, f"{prefix}_{g}_centroids.npy")
        if os.path.exists(p):
            arr = np.load(p)
            if arr.size > 0 and arr.ndim == 2:
                return int(arr.shape[1])
    return int(fallback_dim)

# =========================
# Main logic
# =========================
def run(cluster_dir: str,
        out_final_dir: str,
        groups: int,
        img_dim: int,
        txt_dim: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        dropout: float,
        use_cls: bool,
        device: str):

    _ensure_dir(out_final_dir)

    mapping, mtype = _pick_mapping(cluster_dir)
    if len(mapping) < groups:
        print(f"[WARN] 매핑 쌍이 {len(mapping)}개입니다. groups={groups}. 그래도 진행합니다.")

    # 정확히 4개만 사용 (앞에서 optimal이면 보통 4개일 것)
    mapping = mapping[:groups]

    # 모델 준비
    model = CrossAttentionFuser(
        img_dim=img_dim, txt_dim=txt_dim,
        d_model=d_model, n_heads=n_heads,
        num_layers=num_layers, dropout=dropout,
        use_cls=use_cls
    ).to(device)
    model.eval()

    z_list = []
    meta_pairs = []

    with torch.no_grad():
        for m in mapping:
            gi = int(m["img_group"])
            tj = int(m["mapped_txt_group"])

            img_path = os.path.join(cluster_dir, f"img_clientgroup_{gi}_centroids.npy")
            txt_path = os.path.join(cluster_dir, f"txt_clientgroup_{tj}_centroids.npy")
            Ci = _load_centroids(img_path)
            Ct = _load_centroids(txt_path)

            if Ci.size == 0 or Ct.size == 0:
                Z = np.zeros((d_model,), np.float32)
                print(f"[WARN] empty centroids for (img={gi}, txt={tj}) → zero vector")
            else:
                Qi = torch.from_numpy(Ci).unsqueeze(0).to(device)  # (1,Q,IMG_DIM)
                Kt = torch.from_numpy(Ct).unsqueeze(0).to(device)  # (1,K,TXT_DIM)
                z = model.encode(Qi, Kt)                            # (1,d_model)
                Z = z.squeeze(0).cpu().numpy().astype(np.float32)   # (d_model,)

            # 개별 저장
            z_fname = f"Z_img{gi}_txt{tj}.npy"
            np.save(os.path.join(out_final_dir, z_fname), Z)

            z_list.append(Z)
            meta_pairs.append({
                "img_group": gi,
                "txt_group": tj,
                "img_centroids_file": os.path.basename(img_path),
                "txt_centroids_file": os.path.basename(txt_path),
                "z_file": z_fname
            })

    # 번들 저장(.npz)
    z_npz = {f"Z_{i}": z for i, z in enumerate(z_list)}
    np.savez(os.path.join(out_final_dir, "final_output_z_vectors.npz"), **z_npz)

    # 인덱스 저장
    idx = {
        "mapping_type": mtype,
        "d_model": int(d_model),
        "n_heads": int(n_heads),
        "num_layers": int(num_layers),
        "dropout": float(dropout),
        "use_cls": bool(use_cls),
        "device": device,
        "pairs": meta_pairs
    }
    with open(os.path.join(out_final_dir, "final_output_z_index.json"), "w", encoding="utf-8") as f:
        json.dump(idx, f, ensure_ascii=False, indent=2)

    print(f"[FINAL] saved {len(z_list)} Z vectors -> {out_final_dir}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster_dir", type=str, default=None,
                    help="클러스터링 산출물 디렉토리 (기본: {OUT_GLOBAL_DIR}/clustering)")
    ap.add_argument("--out_dir", type=str, default=None,
                    help="최종 Z 저장 디렉토리 (기본: {OUT_GLOBAL_DIR}/final_output_z)")
    ap.add_argument("--groups", type=int, default=4, help="생성할 Z 개수(매핑 페어 수)")
    # 모델 하이퍼파라미터
    ap.add_argument("--d_model", type=int, default=None)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--use_cls", action="store_true")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main():
    args = parse_args()

    cluster_dir = args.cluster_dir or os.path.join(cfg.OUT_GLOBAL_DIR, "clustering")
    out_dir     = args.out_dir     or os.path.join(cfg.OUT_GLOBAL_DIR, "final_output_z")

    # cfg 기본값
    cfg_img = int(getattr(cfg, "IMG_DIM", 256))
    cfg_txt = int(getattr(cfg, "TXT_DIM", 256))
    d_model = int(args.d_model) if args.d_model is not None else int(getattr(cfg, "D_MODEL", 256))

    # 🔎 센트로이드 파일에서 실제 차원 자동 추론
    IMG_DIM = _infer_dim_from_centroids(cluster_dir, "img_clientgroup", int(args.groups), cfg_img)
    TXT_DIM = _infer_dim_from_centroids(cluster_dir, "txt_clientgroup", int(args.groups), cfg_txt)
    print(f"[FINAL] inferred dims -> IMG_DIM={IMG_DIM}, TXT_DIM={TXT_DIM}, D_MODEL={d_model}")

    run(cluster_dir=cluster_dir,
        out_final_dir=out_dir,
        groups=int(args.groups),
        img_dim=IMG_DIM,
        txt_dim=TXT_DIM,
        d_model=d_model,
        n_heads=int(args.n_heads),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
        use_cls=bool(args.use_cls),
        device=str(args.device))



if __name__ == "__main__":
    main()
