"""可学习编解码器（2026-10-05 用户定案：A 方案浅对称宽 MLP）。

encoder: n_in(19,843) → GELU → hidden(8,192) → n_latent(4,096)
decoder: n_latent     → GELU → hidden        → n_out(19,843)

独立两个 ckpt（encoder.pt / decoder.pt），推理时分别加载。
训练/推理均作用在 log1p 表达空间（与缓存/主模型口径一致）。
PCA 固定投影是"线性 autoencoder 最优解"——本模块是对照臂要超越的基线。
"""
import torch
from torch import nn


class AEEncoder(nn.Module):
    def __init__(self, n_in: int = 19843, hidden: int = 8192, n_latent: int = 4096):
        super().__init__()
        self.n_in = n_in
        self.n_latent = n_latent
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.GELU(),
            nn.Linear(hidden, n_latent),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AEDecoder(nn.Module):
    def __init__(self, n_out: int = 19843, hidden: int = 8192, n_latent: int = 4096):
        super().__init__()
        self.n_out = n_out
        self.n_latent = n_latent
        self.net = nn.Sequential(
            nn.Linear(n_latent, hidden), nn.GELU(),
            nn.Linear(hidden, n_out),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def load_ae(path: str, n_in: int = 19843, hidden: int = 8192, n_latent: int = 4096):
    """按文件名加载 encoder.pt / decoder.pt（两 ckpt 独立，2026-10-05 用户定调）。"""
    if path.endswith('encoder.pt'):
        m = AEEncoder(n_in, hidden, n_latent)
    elif path.endswith('decoder.pt'):
        m = AEDecoder(n_out=n_in, hidden=hidden, n_latent=n_latent)
    else:
        raise ValueError(f'path must end with encoder.pt/decoder.pt: {path}')
    m.load_state_dict(torch.load(path, map_location='cpu', weights_only=False))
    return m
