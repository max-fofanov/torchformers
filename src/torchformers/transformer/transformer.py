import torch
from torch import nn

from ..utils import Encoder
from .decoder import TransformerDecoder


class Transformer(nn.Module):
    def __init__(self, d_model: int, N: int, vocab_len: int, num_heads: int = 1):
        super().__init__()

        self.encoder = Encoder(d_model, N, num_heads=num_heads)
        self.decoder = TransformerDecoder(d_model, N, vocab_len, num_heads=num_heads)

        self.head = nn.Sequential(nn.Linear(d_model, vocab_len), nn.Softmax(dim=-1))

    def forward(
        self, encoder_input: torch.Tensor, decoder_input: torch.Tensor
    ) -> torch.Tensor:
        encoder_output = self.encoder(encoder_input)
        decoder_output = self.decoder(decoder_input, encoder_output)
        return self.head(decoder_output)
