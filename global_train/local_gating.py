import torch
import torch.nn as nn
import torch.nn.functional as F
from .models import GlobalLocalGatedFusion

class LocalClassifierWithGate(nn.Module):
    def __init__(
        self,
        d_img: int,
        d_txt: int,
        d_local_fused: int,
        d_global: int,
        n_labels: int,
        proj_img_dim: int | None = None,   # e.g. 256 (None이면 d_img 유지)
        proj_txt_dim: int | None = None,   # e.g. 256 or 128 (None이면 d_txt 유지)
        dropout: float = 0.1
    ):
        super().__init__()
        self.p_img = proj_img_dim if proj_img_dim is not None else d_img
        self.p_txt = proj_txt_dim if proj_txt_dim is not None else d_txt

        self.proj_img = nn.Linear(d_img, self.p_img)
        self.proj_txt = nn.Linear(d_txt, self.p_txt)

        self.fuse = nn.Linear(self.p_img + self.p_txt, d_local_fused)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(d_local_fused)
        self.dp = nn.Dropout(dropout)

        # 글로벌-로컬 게이트 (입력: d_local_fused, 글로벌: d_global, 출력: d_local_fused)
        self.gate = GlobalLocalGatedFusion(d_local_fused, d_global, d_local_fused)

        self.cls = nn.Linear(d_local_fused, n_labels)

    def _ensure_batch(self, x_img: torch.Tensor | None, x_txt: torch.Tensor | None) -> int:
        if x_img is not None:
            return x_img.size(0)
        if x_txt is not None:
            return x_txt.size(0)
        raise ValueError("at least one modality must be provided")

    def _prep_Z(self, Z_global: torch.Tensor | None, B: int) -> torch.Tensor | None:
        """
        Z_global: [K, d_global] or [B, K, d_global]
        - [K, d] 이면 배치 차원으로 타일
        - None이면 그대로 None 반환 (게이팅 생략)
        """
        if Z_global is None:
            return None
        if Z_global.dim() == 2:
            K, d = Z_global.shape
            return Z_global.unsqueeze(0).expand(B, K, d).contiguous()
        elif Z_global.dim() == 3:
            # [B, K, d] 가정
            return Z_global
        else:
            raise ValueError(f"Unexpected Z_global shape: {tuple(Z_global.shape)}")

    def forward(self, img_rep: torch.Tensor | None = None,
                txt_rep: torch.Tensor | None = None,
                Z_global: torch.Tensor | None = None):
        B = self._ensure_batch(img_rep, txt_rep)

        if img_rep is None:
            img_feat = img_rep = torch.zeros(B, self.p_img,
                                             device=txt_rep.device if txt_rep is not None else Z_global.device,
                                             dtype=txt_rep.dtype if txt_rep is not None else torch.float32)
        else:
            img_feat = F.relu(self.proj_img(img_rep))

        if txt_rep is None:
            txt_feat = torch.zeros(B, self.p_txt, device=img_feat.device, dtype=img_feat.dtype)
        else:
            txt_feat = F.relu(self.proj_txt(txt_rep))

        R_in = torch.cat([img_feat, txt_feat], dim=-1)        # [B, p_img+p_txt]
        R = self.dp(self.act(self.fuse(R_in)))               # [B, d_local_fused]

        Zb = self._prep_Z(Z_global, B)                       # [B, K, d_global] or None
        H = self.gate(R, Zb) if Zb is not None else R        # [B, d_local_fused]
        H = self.norm(self.dp(H))

        logits = self.cls(H)                                  # [B, n_labels]
        return logits, H
