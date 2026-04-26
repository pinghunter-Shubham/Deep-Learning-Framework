import numpy as np

class Optimizer:
    def __init__(self, parameters, lr):
        self.parameters = [p for p in parameters if p.requires_grad]
        self.lr = lr

    def zero_grad(self):
        for p in self.parameters:
            if p.grad is not None:
                p.grad = np.zeros_like(p.grad)

    def step(self):
        raise NotImplementedError
