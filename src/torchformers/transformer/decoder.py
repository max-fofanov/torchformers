from functools import cached_property
from torch import nn
import torch


class TransformerDecoder(nn.Module):
    def __init__(self, d_model: int, N: int, vocab_len: int):
        super().__init__()

        self.d_model = d_model
        self.N = N
        self.vocab_len = vocab_len

        self.decoders = [
            TransformerDecoderBlock(self.d_model, self.vocab_len) for _ in range(self.N)
        ]

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        for decoder in self.decoders:
            x = decoder(x, encoder_output)

        return x


class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model: int, vocab_len: int):
        super().__init__()

        self.d_model = d_model
        self.vocab_len = vocab_len

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

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        x_att, _ = self.attn_1(x, x, x, attn_mask=self.attn_mask)
        x = self.norm_1(x_att + x)

        x_att, _ = self.attn_2(
            x, encoder_output, encoder_output, attn_mask=self.attn_mask
        )
        x = self.norm_2(x_att + x)

        x_ff = self.ff(x)
        x = self.norm_3(x + x_ff)

        return x

    @cached_property
    def attn_mask(self) -> torch.Tensor:
        mask = torch.triu(torch.ones(self.vocab_len, self.vocab_len), diagonal=1)
        return mask.masked_fill(mask == 1, float("-inf"))
