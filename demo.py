from src import Transformer
import torch


d_model = 512
max_len = 20
batch_num = 10

model = Transformer(d_model=d_model, N=6, max_len=max_len)

encoder_input = torch.rand((batch_num, max_len, d_model))
decoder_input = torch.rand((batch_num, max_len, d_model))

output = model(encoder_input, decoder_input)

assert torch.allclose(
    torch.sum(output, dim=-1), torch.tensor(1.0), atol=1e-9
) and output.size() == (batch_num, max_len, max_len)
print("OK")
