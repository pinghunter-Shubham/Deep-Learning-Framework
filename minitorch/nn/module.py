from minitorch.tensor import Tensor

class Parameter(Tensor):
    """
    A kind of Tensor that is to be considered a module parameter.
    Parameters are Tensors that have `requires_grad=True` by default.
    """
    def __init__(self, data, requires_grad=True):
        super().__init__(data, requires_grad=requires_grad)

class Module:
    """
    Base class for all neural network modules.
    Your models should also subclass this class.
    """
    def __init__(self):
        self._parameters = {}
        self._modules = {}
        self.training = True
        
    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        elif isinstance(value, list) or isinstance(value, tuple):
            for i, v in enumerate(value):
                if isinstance(v, Module):
                    self._modules[f"{name}.{i}"] = v
        super().__setattr__(name, value)
        
    def parameters(self):
        """Returns an iterator over module parameters."""
        params = list(self._parameters.values())
        for m in self._modules.values():
            params.extend(m.parameters())
        return params
        
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
        
    def train(self, mode=True):
        self.training = mode
        for m in self._modules.values():
            m.train(mode)
            
    def eval(self):
        self.train(False)
