import numpy as np
from .module import Module
from minitorch.tensor import Tensor


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        mask = (np.random.random(x.shape) > self.p).astype(np.float32) / (1.0 - self.p)
        return x * Tensor(mask)
