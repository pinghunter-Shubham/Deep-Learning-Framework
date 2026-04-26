import numpy as np
from .module import Module, Parameter

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Kaiming initialization
        bound = 1 / np.sqrt(in_features)
        
        # Weight shape matches PyTorch format (out, in), but we'll use (in, out) for simpler MatMul
        # Let's use (in, out) so we can do X @ W
        self.weight = Parameter(np.random.uniform(-bound, bound, (in_features, out_features)))
        
        if bias:
            self.bias = Parameter(np.random.uniform(-bound, bound, (out_features,)))
        else:
            self.bias = None

    def forward(self, x):
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out
