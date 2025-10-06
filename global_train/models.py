# models.py
import math
import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    """
    Q, K, V -> cross attention
    - return_per_query=False: 전역 벡터 [d_model] 반환(기본, 이전과 동일)
    - return_per_query=True : 쿼리별 출력 [Nq, d_model] 반환
    """
    def __init__(self, d_q, d_k, d_v, d_model, return_per_query: bool = False, pool: str = "mean"):
        super().__init__()
        self.Wq = nn.Linear(d_q, d_model, bias=False)
        self.Wk = nn.Linear(d_k, d_model, bias=False)
        self.Wv = nn.Linear(d_v, d_model, bias=False)
        self.scale = math.sqrt(d_model)
        self.return_per_query = return_per_query
        self.pool = pool  # "mean" | "max"

    def forward(self, Q, K, V):
        # Q: [Nq, d_q], K: [Nk, d_k], V: [Nk, d_v]
        Qp = self.Wq(Q)              # [Nq, d_model]
        Kp = self.Wk(K)              # [Nk, d_model]
        Vp = self.Wv(V)              # [Nk, d_model]
        A  = torch.softmax(Qp @ Kp.T / self.scale, dim=-1)  # [Nq, Nk]
        Zq = A @ Vp  # [Nq, d_model]

        if self.return_per_query:
            return Zq  # [Nq, d_model]

        # 글로벌 벡터 하나로 풀링
        if self.pool == "max":
            return Zq.max(dim=0).values  # [d_model]
        # default: mean
        return Zq.mean(dim=0)            # [d_model]


class GlobalLocalGatedFusion(nn.Module):
    """
    R(local)과 Z(global)를 LSTM-gate 컨셉으로 결합.
    Z는 다음 중 하나를 받을 수 있음:
      - [d_global]          (전역 1 벡터)
      - [B, d_global]       (배치별 전역)
      - [K, d_global]       (K개 프로토타입)
      - [B, K, d_global]    (배치별 K개 프로토타입)
    K축은 pool='mean' 또는 'max'로 요약.
    """
    def __init__(self, d_local, d_global, d_out, pool: str = "mean"):
        super().__init__()
        self.proj_local  = nn.Linear(d_local,  d_out)
        self.proj_global = nn.Linear(d_global, d_out)
        self.Wi = nn.Linear(d_out * 2, d_out)  # input gate (R 비중)
        self.Wf = nn.Linear(d_out * 2, d_out)  # forget gate (Z 비중)
        assert pool in ("mean", "max")
        self.pool = pool

    def _pool_K(self, Z, B: int):
        """
        다양한 Z 모양을 [B, d_global]로 정규화.
        """
        if Z.dim() == 1:
            # [d] -> [B, d]
            return Z.unsqueeze(0).expand(B, -1)
        if Z.dim() == 2:
            # [B, d] or [K, d]
            if Z.size(0) == B:       # [B, d]
                return Z
            if Z.size(0) == 1:       # [1, d] -> tile
                return Z.expand(B, -1)
            # [K, d] -> pool over K -> [d] -> [B, d]
            if self.pool == "max":
                z = Z.max(dim=0).values
            else:
                z = Z.mean(dim=0)
            return z.unsqueeze(0).expand(B, -1)
        if Z.dim() == 3:
            # [B, K, d]
            if self.pool == "max":
                return Z.max(dim=1).values  # [B, d]
            else:
                return Z.mean(dim=1)        # [B, d]
        raise ValueError(f"Unexpected Z shape: {tuple(Z.shape)}")

    def forward(self, R, Z):
        """
        R: [B, d_local]
        Z: [d], [B, d], [K, d], or [B, K, d]
        """
        B = R.size(0)
        Zb = self._pool_K(Z, B)             # [B, d_global]

        r = self.proj_local(R)              # [B, d_out]
        z = self.proj_global(Zb)            # [B, d_out]
        cat = torch.cat([r, z], dim=-1)     # [B, 2*d_out]
        i = torch.sigmoid(self.Wi(cat))     # input-gate
        f = torch.sigmoid(self.Wf(cat))     # forget-gate
        return f * z + i * r                # [B, d_out]
