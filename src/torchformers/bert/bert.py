import torch
from torch import nn

from ..utils import Encoder


class BERT(nn.Module):
    def __init__(self, d_model: int, N: int, num_classes: int, num_heads: int = 1):
        super().__init__()

        self.encoder = Encoder(d_model, N, num_heads=num_heads)

        self.head = nn.Sequential(
            nn.Linear(d_model, num_classes),
            nn.Softmax(dim=-1)
        )

    def forward(self, x) -> torch.Tensor:
        encoder_output = self.encoder(x)
        return self.head(encoder_output)
