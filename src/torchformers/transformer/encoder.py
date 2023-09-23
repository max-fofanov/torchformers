import torch
from torch import nn

from ..shared import SelfAttentionBlock, FeedForwardBlock


class TransformerEncoder(nn.Module):
    def __init__(self, d_model: int, N: int, num_heads: int = 1):
        super().__init__()

        self.encoders = [
            TransformerEncoderBlock(d_model, num_heads=num_heads) for _ in range(N)
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for encoder in self.encoders:
            x = encoder(x)

        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 1):
        super().__init__()

        self.self_att = SelfAttentionBlock(d_model, num_heads=num_heads)
        self.ff = FeedForwardBlock(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.self_attn(x, x, x)
        x = self.ff(x)

        return x
