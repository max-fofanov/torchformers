# transformer

## Description
This repository contains a PyTorch implementation of the Transformer model as described in the seminal paper "[Attention Is All You Need](https://arxiv.org/abs/1706.03762)" by Vaswani et al.
## Installation
To install this package from repository simply run
```bash
pip install git+https://github.com/max-fofanov/torchformers.git
```
## Training

```python
from torchformers import Transformer

d_model = 512
N = 6
vocab_len = 300

model = Transformer(d_model, N, vocab_len).train()
data =  # training data
loss_func =  # some loss function
oprimizer =  # some optimizer

# train loop
for epoch in range(1, 10):
    for encoder_input, decoder_input, target in data:
        output = model(encoder_input, decoder_input)
        loss = loss_func(output, target)

        loss.backward()
        oprimizer.step()

```
## Inference
```python
model = # load model

encoder = model.encoder.eval()
decoder = model.decoder.eval()

encoder_input = # encoder input
encoder_output = encoder(encoder_input)

decoder_input = # decoder input (usually just bos token)

while # some criterion
    decoder_output = decoder(decoder_input, encoder_output)
    decoder_input = # append to decoder_input according to your inference strategy
```
## Cooperation
If you have any questions regarding this repository or have any research/work related offers you can reach me via 
[email](mailto:max.fofanov@gmail.com) or [telegram](https://t.me/Max_Fofanov).