import torch
from torch import nn

from ..utils import MaskedSelfAttentionBlock, CrossAttentionBlock, FeedForwardBlock


class TransformerDecoder(nn.Module):
    def __init__(self, d_model: int, N: int, vocab_len: int, num_heads: int = 1):
        super().__init__()

        self.decoders = [
            TransformerDecoderBlock(d_model, vocab_len, num_heads=num_heads)
            for _ in range(N)
        ]

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        for decoder in self.decoders:
            x = decoder(x, encoder_output)
        return x


class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model: int, vocab_len: int, num_heads: int = 1):
        super().__init__()

        self.masked_self_att = MaskedSelfAttentionBlock(d_model, vocab_len, num_heads=num_heads)
        self.cross_att = CrossAttentionBlock(d_model, num_heads=num_heads)

        self.ff = FeedForwardBlock(d_model)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        x = self.masked_self_att(x)
        x = self.cross_att(x, encoder_output)
        return self.ff(x)
