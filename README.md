# 🌟 Torchformers: A Lightweight and Highly Customizable Transformers Library 🌟

## Overview:
Torchformers is a minimalist, lightweight Python package designed for ease of use, high customization, and efficiency, focusing on providing core Transformer models and their building blocks essential for Natural Language Processing tasks. It is built on top of PyTorch, ensuring optimized implementations of the models.

## 🔥 Features:

- Lightweight Design: Torchformers is extremely lightweight, making it an ideal choice for developers who prefer a clean and uncomplicated interface for Transformer models.
- Core Models and Building Blocks: The package includes implementations of three fundamental Transformer models: the original Transformer, BERT, and GPT, along with their essential building blocks, allowing users to construct their own models with ease.
- High Customization: Torchformers offers extensive customization options, enabling users to modify and adapt the models and their components to suit a variety of NLP tasks and requirements.
- Single Dependency: Torchformers requires only PyTorch as a dependency, ensuring a smooth and straightforward installation and setup process.

## 🛠 Core Models and Building Blocks:

- Transformer: The base model and its building blocks provide the foundational architecture for transforming input sequences into output sequences.
- BERT: A versatile model and its components are designed for a range of NLP tasks, offering bidirectional context-based representations.
- GPT: A generative model and its essential elements are capable of producing coherent and contextually relevant text based on the provided input.

## 🏗 Build Your Own Model:

With the provided building blocks and custom layers, users can easily assemble their custom models tailored to their specific needs:
```python
from torch import nn
from torchformers import SelfAttentionBlock, TransformerDecoderBlock

class MyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Initialize custom layers and building blocks
        self.layer1 = TransformerDecoderBlock(...)  # Specify the required parameters
        # Initialize more layers and blocks as needed
        
    def forward(self, x):
        # Define the forward pass using the initialized blocks
        x = self.layer1(x)
        # Continue the forward pass with other layers and blocks as needed
        return x
```

## 💻 Installation

To install this package from repository simply run
```bash
pip install git+https://github.com/max-fofanov/torchformers.git
```

## 🤝 Cooperation

If you have any questions regarding this repository or have any research/work related offers you can reach me via 
[email](mailto:max.fofanov@gmail.com) or [telegram](https://t.me/Max_Fofanov).