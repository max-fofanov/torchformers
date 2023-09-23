import torch
from torch import nn

from ..utils import MaskedSelfAttentionBlock, FeedForwardBlock


class GPTDecoder(nn.Module):
    def __init__(self, d_model: int, N: int, vocab_len: int, num_heads: int = 1):
        super().__init__()

        self.decoders = nn.Sequential(*[
            GPTDecoderBlock(d_model, vocab_len, num_heads=num_heads)
            for _ in range(N)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoders(x)


class GPTDecoderBlock(nn.Module):
    def __init__(self, d_model: int, vocab_len: int, num_heads: int = 1):
        super().__init__()

        self.masked_self_att = MaskedSelfAttentionBlock(
            d_model, vocab_len, num_heads=num_heads
        )

        self.ff = FeedForwardBlock(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.masked_self_att(x)
        return self.ff(x)
