import torch
from torch import nn

from .encoder import TransformerEncoder
from .decoder import TransformerDecoder


class Transformer(nn.Module):
    def __init__(self, d_model: int, N: int, vocab_len: int, num_heads: int = 1):
        super().__init__()

        self.d_model = d_model
        self.N = N
        self.vocab_len = vocab_len

        self.encoder = TransformerEncoder(d_model, N, num_heads=num_heads)
        self.decoder = TransformerDecoder(d_model, N, vocab_len, num_heads=num_heads)

        self.linear = nn.Linear(d_model, vocab_len)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self, encoder_input: torch.Tensor, decoder_input: torch.Tensor
    ) -> torch.Tensor:
        encoder_output = self.encoder(encoder_input)
        decoder_output = self.decoder(decoder_input, encoder_output)

        output = self.linear(decoder_output)

        return self.softmax(output)
