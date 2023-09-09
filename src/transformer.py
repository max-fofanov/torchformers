from torch import nn
from .encoder import Encoder
from .decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, d_model, N, max_len):
        super().__init__()

        self.d_model = d_model
        self.N = N
        self.max_len = max_len

        self.encoders = nn.Sequential(*[Encoder(self.d_model) for _ in range(self.N)])
        self.decoders = nn.Sequential(
            *[Decoder(self.d_model, self.max_len) for _ in range(self.N)]
        )

        self.linear = nn.Linear(self.d_model, self.max_len)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, encoder_input, decoder_input):
        encoder_output = self.encoders(encoder_input)
        self.inject_encoder_output(encoder_output)

        decoder_output = self.decoders(decoder_input)
        output = self.linear(decoder_output)

        return self.softmax(output)

    def inject_encoder_output(self, encoder_output):
        for decoder in self.decoders:
            decoder.encoder_output = encoder_output
