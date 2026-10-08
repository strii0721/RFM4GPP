"""可学习编解码器（2026-10-07 用户定案：加深版 = 阶梯 + 层内残差 + 对称跳连）。

结构（encoder，括号=每段输出，供 decoder 对称拼接）：
  n_in → ResBlock → hidden(8192, s1) → ResBlock → hidden//2(4096, s2) → Linear → n_latent(2048, z)
decoder（镜像）：
  z(2048) → Linear → 4096 → ⊕s2 ResBlock → 4096 → ⊕s1 ResBlock → 8192 → Linear → n_out
残差块 = Linear+GeLU+Linear 主支 + 等宽捷径（维度变化层用 Linear 捷径）。

forward 签名：encoder 返回 (z, skips=[s1, s2])；decoder 接收 (z, skips)。
独立两个 ckpt（encoder.pt / decoder.pt），推理时分别加载。
训练/推理均作用在 log1p 表达空间（与缓存/主模型口径一致）。
"""
import torch
import torch.nn.functional as F
from torch import nn


class ResBlock(nn.Module):
    """残差块：Linear→GeLU→Linear 主支 + 捷径（等宽 Identity / 变维 Linear）。"""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_out)
        self.fc2 = nn.Linear(d_out, d_out)
        self.shortcut = nn.Identity() if d_in == d_out else nn.Linear(d_in, d_out)
        # zero-init 残差末层（2026-10-07 修复 loss 爆炸）：残差块初始=恒等映射，
        # 避免主支+捷径方差逐层翻倍导致前向/梯度失控
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x))) + self.shortcut(x)


class AEEncoder(nn.Module):
    def __init__(self, n_in: int = 19843, hidden: int = 8192, n_latent: int = 2048):
        super().__init__()
        self.n_in = n_in
        self.n_latent = n_latent
        self.h1 = hidden
        self.h2 = hidden // 2
        self.enc1 = ResBlock(n_in, self.h1)     # n_in → 8192 (s1)
        self.enc2 = ResBlock(self.h1, self.h2)  # 8192 → 4096 (s2)
        self.z_proj = nn.Linear(self.h2, n_latent)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        z = self.z_proj(s2)
        return z, [s1, s2]


class AEDecoder(nn.Module):
    def __init__(self, n_out: int = 19843, hidden: int = 8192, n_latent: int = 2048):
        super().__init__()
        self.n_out = n_out
        self.n_latent = n_latent
        self.h1 = hidden
        self.h2 = hidden // 2
        self.z_in = nn.Linear(n_latent, self.h2)          # z → 4096
        self.dec2 = ResBlock(self.h2 * 2, self.h2)        # (4096 ⊕ s2) → 4096
        self.dec1 = ResBlock(self.h2 + self.h1, self.h1)  # (4096 ⊕ s1) → 8192
        self.out = nn.Linear(self.h1, n_out)

    def forward(self, z: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        s1, s2 = skips
        h = F.gelu(self.z_in(z))
        h = self.dec2(torch.cat([h, s2], dim=-1))
        h = self.dec1(torch.cat([h, s1], dim=-1))
        return self.out(h)


def load_ae(path: str, n_in: int = 19843, hidden: int = 8192, n_latent: int = 2048):
    """按文件名加载 encoder.pt / decoder.pt（两 ckpt 独立，2026-10-05 用户定调）。"""
    if path.endswith('encoder.pt'):
        m = AEEncoder(n_in, hidden, n_latent)
    elif path.endswith('decoder.pt'):
        m = AEDecoder(n_out=n_in, hidden=hidden, n_latent=n_latent)
    else:
        raise ValueError(f'path must end with encoder.pt/decoder.pt: {path}')
    m.load_state_dict(torch.load(path, map_location='cpu', weights_only=False))
    return m
