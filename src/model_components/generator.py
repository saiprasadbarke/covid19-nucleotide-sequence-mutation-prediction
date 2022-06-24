import torch.nn as nn


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, hidden_size, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x):
        return self.proj(x)
