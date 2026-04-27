import numpy as np
from .optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, parameters, lr, momentum=0.0):
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.velocity = [np.zeros_like(p.data) for p in self.parameters]

    def step(self):
        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue
            if self.momentum > 0.0:
                self.velocity[i] = self.momentum * self.velocity[i] + p.grad
                p.data -= self.lr * self.velocity[i]
            else:
                p.data -= self.lr * p.grad
