# models.py
import torch
from torch import nn
from torchvision import models
from transformers import AutoModel
from torchvision.models import MobileNet_V3_Small_Weights

class ImageHead(nn.Module):
    def __init__(self, trainable=True, out_dim=256):
        super().__init__()
        base = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.proj = nn.Linear(576, out_dim)
        if not trainable:
            for p in self.parameters(): 
                p.requires_grad_(False)

    def forward(self, x):
        h = self.pool(self.features(x)).flatten(1)   # [B, 576]
        return self.proj(h)                          # [B, out_dim]

class TextHead(nn.Module):
    def __init__(self, model_name='prajjwal1/bert-tiny', out_dim=256, trainable=True):
        super().__init__()
        self.base = AutoModel.from_pretrained(model_name)
        hid = self.base.config.hidden_size
        self.proj = nn.Linear(hid, out_dim)
        if not trainable:
            for p in self.base.parameters(): 
                p.requires_grad_(False)

    def forward(self, input_ids, attention_mask):
        out = self.base(input_ids=input_ids, attention_mask=attention_mask)
        last = out.last_hidden_state                  # [B, L, H]
        mask = attention_mask.unsqueeze(-1).to(last.dtype)
        s = (last * mask).sum(1) / mask.sum(1).clamp(min=1)  # masked mean → [B, H]
        return self.proj(s)                           # [B, out_dim]

class MultiModalLateFusion(nn.Module):
    def __init__(self, num_classes, img_out=256, txt_out=256, hidden=256, dropout=0.2,
                 text_model_name='prajjwal1/bert-tiny', image_trainable=True, text_trainable=True):
        super().__init__()
        self.img_out = img_out
        self.txt_out = txt_out
        self.img_enc = ImageHead(trainable=image_trainable, out_dim=img_out)
        self.txt_enc = TextHead(model_name=text_model_name, out_dim=txt_out, trainable=text_trainable)
        # self.fuse = nn.Sequential(
        #     nn.LazyLinear(hidden),   # in_features 자동 추론
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden, num_classes)
        # )   
        self.fuse = nn.Sequential(
            nn.Linear(img_out + txt_out, hidden),  # 512 -> hidden
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes)
        )

    def _to_bd(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 3:
            z = z.mean(dim=1)     # [B, D]
        elif z.dim() > 3:
            z = z.flatten(1)      # [B, D’]
        else:
            z = z.view(z.size(0), -1)
        return z

    def _to_b1_mask(self, m: torch.Tensor) -> torch.Tensor:
        """
        모달리티 존재 여부 마스크를 [B,1] 스칼라로 정규화.
        - m이 [B]면 [B,1]로
        - m이 [B,1]이면 그대로
        - m이 [B, ...] (예: [B,64], [B,1,64])면 '하나라도 True면 1' 로 축약 → [B,1]
        """
        if m.dim() == 1:
            return m.float().unsqueeze(1)
        elif m.dim() == 2 and m.size(1) == 1:
            return m.float()
        else:
            rest = tuple(range(1, m.dim()))
            m = (m.sum(dim=rest) > 0).float().unsqueeze(1)
            return m

    def forward(self, pixel_values, input_ids, attention_mask, img_mask, txt_mask):
        zi = self.img_enc(pixel_values)                      # [B, Di]
        zt = self.txt_enc(input_ids, attention_mask)         # [B, Dt]
        zi = self._to_bd(zi)
        zt = self._to_bd(zt)

        # ✅ 마스크를 반드시 [B,1]로 정규화
        img_mask = self._to_b1_mask(img_mask)
        txt_mask = self._to_b1_mask(txt_mask)

        # 브로드캐스트는 2D 기준으로만 일어나게!
        zi = zi * img_mask.to(zi.dtype)                      # [B, Di]
        zt = zt * txt_mask.to(zt.dtype)                      # [B, Dt]

        z = torch.cat([zi, zt], dim=1)                       # [B, Di+Dt]
        return self.fuse(z)                                  # [B, num_classes]

