from functools import cached_property
from torch import nn
import torch


class Decoder(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()

        self.d_model = d_model
        self.max_len = max_len

        self.encoder_output = None

        self.attn_1 = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=1, batch_first=True
        )
        self.norm_1 = nn.LayerNorm(d_model)

        self.attn_2 = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=1, batch_first=True
        )
        self.norm_2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
        )
        self.norm_3 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.encoder_output is not None, "Encoder output should be initialized"

        x_att, _ = self.attn_1(x, x, x, attn_mask=self.attn_mask)
        x = self.norm_1(x_att + x)

        x_att, _ = self.attn_2(
            x, self.encoder_output, self.encoder_output, attn_mask=self.attn_mask
        )
        x = self.norm_2(x_att + x)

        x_ff = self.ff(x)
        x = self.norm_3(x + x_ff)

        return x

    @cached_property
    def attn_mask(self) -> torch.Tensor:
        mask = torch.triu(torch.ones(self.max_len, self.max_len), diagonal=1)
        return mask.masked_fill(mask == 1, float("-inf"))
