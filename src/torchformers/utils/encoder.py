import torch
from torch import nn

from ..utils import SelfAttentionBlock, FeedForwardBlock


class Encoder(nn.Module):
    def __init__(self, d_model: int, N: int, num_heads: int = 1):
        super().__init__()

        self.encoders = nn.Sequential(
            *[EncoderBlock(d_model, num_heads=num_heads) for _ in range(N)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoders(x)


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 1):
        super().__init__()

        self.self_att = SelfAttentionBlock(d_model, num_heads=num_heads)
        self.ff = FeedForwardBlock(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.self_att(x)
        return self.ff(x)
