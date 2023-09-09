import torch
from src import Transformer


d_model = 512
N = 6
max_len = 77

model = Transformer(d_model, N, max_len).eval()

encoder_input = torch.rand(77, 512)
decoder_input = torch.rand(77, 512)

output = model(encoder_input, decoder_input)

assert torch.allclose(torch.sum(output, dim=1), torch.ones(max_len), atol=1e-9)
print("OK")
