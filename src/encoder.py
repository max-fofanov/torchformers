from torch import nn


class Encoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=1)
        self.norm_1 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
        )
        self.norm_2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x_att, _ = self.attn(x, x, x)
        x = self.norm_1(x_att + x)

        x_ff = self.ff(x)
        x = self.norm_2(x + x_ff)

        return x
