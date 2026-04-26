from .optimizer import Optimizer

class SGD(Optimizer):
    def step(self):
        for p in self.parameters:
            if p.grad is not None:
                p.data -= self.lr * p.grad
