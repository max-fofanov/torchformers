from torchformers.transformer import Transformer
import torch


# embedding arity
d_model = 512
# number of encoder/decoder blocks
N = 6
# vocabulary length
vocab_len = 300

model = Transformer(d_model, N, vocab_len).eval()

# single input
encoder_input = torch.rand((vocab_len, d_model))
decoder_input = torch.rand((vocab_len, d_model))

output = model(encoder_input, decoder_input)
assert output.size() == (
    vocab_len,
    vocab_len,
), f"Output shape doesn't match the expected ({vocab_len}, {vocab_len})"

batch_size = 30

# batched input
encoder_input_batched = torch.rand((batch_size, vocab_len, d_model))
decoder_input_batched = torch.rand((batch_size, vocab_len, d_model))

output = model(encoder_input_batched, decoder_input_batched)
assert output.size() == (
    batch_size,
    vocab_len,
    vocab_len,
), f"Output shape doesn't match the expected ({batch_size}, {vocab_len}, {vocab_len})"

# if all checks pass print OK
print("OK")
