import numpy as np
from .storage import Storage

class Tensor:
    def __init__(self, data=None, requires_grad=False, shape=None, strides=None, offset=0, _storage=None):
        if _storage is not None:
            self.storage = _storage
            self.shape = shape
            self.strides = strides
            self.offset = offset
        else:
            if isinstance(data, (int, float)):
                data = np.array([data], dtype=np.float32)
            elif isinstance(data, list):
                data = np.array(data, dtype=np.float32)
            elif not isinstance(data, np.ndarray):
                data = np.array(data)
                
            self.storage = Storage(data)
            self.shape = data.shape
            # Calculate strides in elements (NumPy provides strides in bytes)
            self.strides = tuple(s // data.itemsize for s in data.strides)
            self.offset = 0

        self.requires_grad = requires_grad
        
        # Initialize gradients matching shape
        self.grad = np.zeros(self.shape, dtype=np.float32) if requires_grad else None
        
        # Graph construction
        self._prev = []
        self._creator = None
        self._ctx = None

    @property
    def data(self):
        """Reconstruct a NumPy view interpreting the storage according to Tensor metadata."""
        itemsize = self.storage.data.itemsize
        byte_strides = tuple(s * itemsize for s in self.strides)
        
        from numpy.lib.stride_tricks import as_strided
        # Slice from offset, then apply shape and strides
        view = as_strided(self.storage.data[self.offset:], shape=self.shape, strides=byte_strides, writeable=True)
        return view
        
    @data.setter
    def data(self, new_data):
        """Allows writing recursively to the view (e.g. w.data -= lr * grad)"""
        if isinstance(new_data, np.ndarray):
            self.data[...] = new_data
        else:
            self.data[...] = np.array(new_data)

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad}, shape={self.shape}, strides={self.strides})"

    def backward(self):
        from .autograd import backward
        backward(self)

    # --- Tensor Views (Zero Copy) ---
    def reshape(self, new_shape):
        from .ops import reshape
        return reshape(self, new_shape)

    def transpose(self, dim0=0, dim1=1):
        from .ops import transpose
        return transpose(self, dim0, dim1)
            
    # Operator overloading for syntactic sugar
    def __add__(self, other):
        from .ops import add
        return add(self, other)
        
    def __radd__(self, other):
        from .ops import add
        return add(other, self)
        
    def __mul__(self, other):
        from .ops import mul
        return mul(self, other)
        
    def __rmul__(self, other):
        from .ops import mul
        return mul(other, self)
        
    def __matmul__(self, other):
        from .ops import matmul
        return matmul(self, other)
        
    def sum(self, axis=None, keepdims=False):
        from .ops import sum_op, sum_axis
        if axis is None and not keepdims:
            return sum_op(self)
        return sum_axis(self, axis, keepdims)

    def mean(self, axis=None, keepdims=False):
        if axis is None:
            return self.sum() * (1.0 / float(np.prod(self.shape)))
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / float(self.shape[axis]))
        
    def relu(self):
        from .ops import relu
        return relu(self)

    def softmax(self, dim=-1):
        from .ops import softmax_op
        return softmax_op(self, dim)

    def sigmoid(self):
        from .ops import sigmoid
        return sigmoid(self)

    def tanh(self):
        from .ops import tanh_op
        return tanh_op(self)

    def exp(self):
        from .ops import exp
        return exp(self)

    def log(self):
        from .ops import log
        return log(self)

    def __pow__(self, power):
        from .ops import pow_op
        return pow_op(self, power)

    def __neg__(self):
        from .ops import neg
        return neg(self)

    def __sub__(self, other):
        from .ops import sub
        return sub(self, other)

    def __rsub__(self, other):
        from .ops import sub
        return sub(other, self)

    def __truediv__(self, other):
        from .ops import div
        return div(self, other)

    def __rtruediv__(self, other):
        from .ops import div
        return div(other, self)
