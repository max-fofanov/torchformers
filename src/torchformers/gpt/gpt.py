import torch
from torch import nn

from .decoder import GPTDecoder


class GPT(nn.Module):
    def __init__(self, d_model: int, N: int, vocab_len: int, num_heads: int = 1):
        super().__init__()

        self.decoder = GPTDecoder(d_model, N, vocab_len, num_heads=num_heads)

        self.head = nn.Sequential(nn.Linear(d_model, vocab_len), nn.Softmax(dim=-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.decoder(x)
        return self.head(x)
