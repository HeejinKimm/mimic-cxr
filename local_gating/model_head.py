# local_gating/model_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# 전역에서 쓰는 동일한 게이트를 재사용 (중복 정의 방지)
from global_train.models import GlobalLocalGatedFusion


class LocalClassifierWithAugment(nn.Module):
    """
    모달리티 결손 보완 + 글로벌 게이팅 분류기
    - 이미지 전용 클라이언트: Z로부터 text feature(128) 보완 (use_hallucinate=True인 경우)
    - 텍스트 전용 클라이언트: Z로부터 image feature(256) 보완 (use_hallucinate=True인 경우)
    - 멀티모달 클라이언트: 양쪽 모두 존재 시 그대로 사용
    - Z_global은 [d_global] 또는 [B, d_global] 모두 허용

    Args:
        d_img:           입력 이미지 임베딩 차원
        d_txt:           입력 텍스트 임베딩 차원
        d_local_fused:   로컬 결합 차원 (게이트 입력/출력도 이 차원)
        d_global:        글로벌 벡터 차원
        n_labels:        클래스 수
        use_hallucinate: 결손 모달을 Z로 보완할지 여부 (False면 0벡터로 대체)
        z_dropout:       Z 보완 시 Z에 대한 드롭아웃(과적합/의존 방지용)
        feat_dropout:    로컬 feature 드롭아웃
    """
    def __init__(
        self,
        d_img: int,
        d_txt: int,
        d_local_fused: int,
        d_global: int,
        n_labels: int,
        use_hallucinate: bool = False,
        z_dropout: float = 0.0,
        feat_dropout: float = 0.0,
    ):
        super().__init__()
        self.use_hallucinate = use_hallucinate

        # 입력 프로젝션(모델 내부 인코더 출력 → 고정 폭)
        self.proj_img = nn.Linear(d_img, 256)
        self.proj_txt = nn.Linear(d_txt, 128)

        # Z → 결손 모달 보완용
        self.z2img = nn.Linear(d_global, 256)
        self.z2txt = nn.Linear(d_global, 128)

        # 로컬 결합 + 게이트 + 분류
        self.fuse = nn.Linear(256 + 128, d_local_fused)
        self.gate = GlobalLocalGatedFusion(d_local_fused, d_global, d_local_fused)
        self.cls  = nn.Linear(d_local_fused, n_labels)

        # 정규화/드롭아웃
        self.feat_do = nn.Dropout(feat_dropout) if feat_dropout > 0 else nn.Identity()
        self.z_do    = nn.Dropout(z_dropout)    if z_dropout > 0 else nn.Identity()

    @staticmethod
    def _expand_z_to_batch(Z: torch.Tensor, B: int) -> torch.Tensor:
        """Z가 [d]이면 [B,d]로 확장."""
        if Z.dim() == 1:
            Z = Z.unsqueeze(0).expand(B, -1)
        return Z

    def forward(
        self,
        img_rep: torch.Tensor | None = None,   # [B, d_img] or None
        txt_rep: torch.Tensor | None = None,   # [B, d_txt] or None
        Z_global: torch.Tensor | None = None,  # [d_global] or [B, d_global]
        mode: str = "auto",
    ):
        if Z_global is None:
            raise ValueError("Z_global is required (global_payload에서 로드한 Z를 전달하세요).")
        if (img_rep is None) and (txt_rep is None):
            raise ValueError("at least one modality must be provided")

        # 배치 크기 결정
        B = (img_rep.size(0) if img_rep is not None else txt_rep.size(0))

        # ----- 결손 모달 보완(or zero) + 프로젝션 -----
        if img_rep is not None:
            img_feat = F.relu(self.proj_img(img_rep))
        else:
            if self.use_hallucinate:
                Zi = self._expand_z_to_batch(Z_global, B)
                Zi = self.z_do(Zi)
                img_feat = F.relu(self.z2img(Zi))
            else:
                img_feat = img_feat = torch.zeros(B, 256, device=Z_global.device if isinstance(Z_global, torch.Tensor) else (img_rep.device if img_rep is not None else txt_rep.device))

        if txt_rep is not None:
            txt_feat = F.relu(self.proj_txt(txt_rep))
        else:
            if self.use_hallucinate:
                Zt = self._expand_z_to_batch(Z_global, B)
                Zt = self.z_do(Zt)
                txt_feat = F.relu(self.z2txt(Zt))
            else:
                txt_feat = torch.zeros(B, 128, device=img_feat.device)

        # ----- 정규화(스케일 안정화) -----
        # eps를 줘서 0 division 방지
        img_feat = F.normalize(img_feat, dim=-1, eps=1e-6)
        txt_feat = F.normalize(txt_feat, dim=-1, eps=1e-6)

        # ----- 로컬 결합 → 게이트 → 로짓 -----
        R = self.fuse(torch.cat([img_feat, txt_feat], dim=-1))  # [B, d_local_fused]
        R = self.feat_do(R)

        Zg = self._expand_z_to_batch(Z_global, B)
        Zg = F.normalize(Zg, dim=-1, eps=1e-6)

        H = self.gate(R, Zg)                 # [B, d_local_fused]
        logits = self.cls(H)                 # [B, n_labels]

        # 학습에서 KD(표현 정렬)용으로 R을 쓰기 때문에 함께 반환
        return logits, R
