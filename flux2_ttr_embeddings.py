from __future__ import annotations

import math

import torch
from torch import nn


class ScalarSinusoidalEmbedding(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        dim = max(2, int(embed_dim))
        if dim % 2 != 0:
            dim += 1
        self.embed_dim = dim
        half = dim // 2
        exponents = torch.arange(half, dtype=torch.float32) / max(1, half - 1)
        inv_freq = torch.exp(-math.log(10000.0) * exponents)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 0:
            x = x.reshape(1)
        x = x.reshape(-1).float()
        angles = x[:, None] * self.inv_freq[None, :]
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
