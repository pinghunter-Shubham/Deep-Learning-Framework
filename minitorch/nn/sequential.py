from .module import Module

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        for i, m in enumerate(modules):
            setattr(self, str(i), m)

    def forward(self, x):
        for name, module in self._modules.items():
            x = module(x)
        return x
