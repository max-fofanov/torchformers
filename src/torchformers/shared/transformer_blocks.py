from functools import cached_property
from typing import Optional

import torch
from torch import nn


class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 1):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_att, _ = self.attn(x, x, x)
        x = self.norm(x_att + x)

        return x


class CrossAttentionBlock(SelfAttentionBlock):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_att, _ = self.attn(x, y, y)
        x = self.norm(x_att + x)

        return x


class MaskedSelfAttentionBlock(SelfAttentionBlock):
    def __init__(self, d_model: int, vocab_len: int, num_heads: int = 1):
        super().__init__(d_model, num_heads=num_heads)

        self.vocab_len = vocab_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_att, _ = self.attn(x, x, x, attn_mask=self.attn_mask)
        x = self.norm(x_att + x)

        return x

    @cached_property
    def attn_mask(self) -> torch.Tensor:
        mask = torch.triu(torch.ones(self.vocab_len, self.vocab_len), diagonal=1)
        return mask.masked_fill(mask == 1, float("-inf"))


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, hidden_size: Optional[int] = None):
        super().__init__()

        hidden_size = d_model if hidden_size is None else hidden_size

        self.ff = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=d_model),
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ff = self.ff(x)
        x = self.norm(x_ff + x)

        return self.ff(x)
