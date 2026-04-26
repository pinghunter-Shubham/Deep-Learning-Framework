import numpy as np
from .tensor import Tensor
from .autograd import Function

def unbroadcast(grad, target_shape):
    """
    Sum gradients over broadcasted dimensions to match target shape.
    """
    if grad.shape == target_shape:
        return grad
    
    ndims_diff = len(grad.shape) - len(target_shape)
    if ndims_diff > 0:
        # Sum over prepended broadcasted dimensions
        grad = grad.sum(axis=tuple(range(ndims_diff)))
        
    # Sum over dimensions that were broadcasted from 1 to N
    for i, dim in enumerate(target_shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)
            
    return grad

def _coerce(obj):
    if not isinstance(obj, Tensor):
        return Tensor(obj, requires_grad=False)
    return obj

class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        return Tensor(a.data + b.data)
        
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output

class Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a.data, b.data)
        return Tensor(a.data * b.data)
        
    @staticmethod
    def backward(ctx, grad_output):
        a_data, b_data = ctx.saved_tensors
        return grad_output * b_data, grad_output * a_data

class Sub(Function):
    @staticmethod
    def forward(ctx, a, b):
        return Tensor(a.data - b.data)
        
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, -grad_output

class MatMul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a.data, b.data)
        return Tensor(a.data @ b.data)
        
    @staticmethod
    def backward(ctx, grad_output):
        a_data, b_data = ctx.saved_tensors
        return grad_output @ b_data.T, a_data.T @ grad_output

class Sum(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a.shape)
        return Tensor(a.data.sum())
        
    @staticmethod
    def backward(ctx, grad_output):
        a_shape, = ctx.saved_tensors
        return grad_output * np.ones(a_shape, dtype=np.float32)

class Reshape(Function):
    @staticmethod
    def forward(ctx, a, new_shape):
        ctx.save_for_backward(a.shape)
        # Calculate new contiguous strides
        new_strides = []
        stride = 1
        for s in reversed(new_shape):
            new_strides.insert(0, stride)
            stride *= s
        return Tensor(data=None, shape=tuple(new_shape), strides=tuple(new_strides), offset=a.offset, _storage=a.storage)

    @staticmethod
    def backward(ctx, grad_output):
        original_shape, = ctx.saved_tensors
        return grad_output.reshape(original_shape), None

class Transpose(Function):
    @staticmethod
    def forward(ctx, a, dim0, dim1):
        ctx.save_for_backward(dim0, dim1)
        new_shape = list(a.shape)
        new_strides = list(a.strides)
        new_shape[dim0], new_shape[dim1] = new_shape[dim1], new_shape[dim0]
        new_strides[dim0], new_strides[dim1] = new_strides[dim1], new_strides[dim0]
        return Tensor(data=None, shape=tuple(new_shape), strides=tuple(new_strides), offset=a.offset, _storage=a.storage)
        
    @staticmethod
    def backward(ctx, grad_output):
        dim0, dim1 = ctx.saved_tensors
        return np.swapaxes(grad_output, dim0, dim1), None, None

class ReLU(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a.data)
        return Tensor(np.maximum(a.data, 0))
        
    @staticmethod
    def backward(ctx, grad_output):
        a_data, = ctx.saved_tensors
        grad_a = grad_output.copy()
        grad_a[a_data <= 0] = 0
        return grad_a

class Pow(Function):
    @staticmethod
    def forward(ctx, a, power):
        ctx.save_for_backward(a.data, power)
        return Tensor(np.power(a.data, power))
        
    @staticmethod
    def backward(ctx, grad_output):
        a_data, power = ctx.saved_tensors
        return grad_output * power * np.power(a_data, power - 1), None

class LogSoftmax(Function):
    @staticmethod
    def forward(ctx, a, dim):
        # Numeric stability
        max_a = np.max(a.data, axis=dim, keepdims=True)
        exp_a = np.exp(a.data - max_a)
        sum_exp = np.sum(exp_a, axis=dim, keepdims=True)
        log_softmax = a.data - max_a - np.log(sum_exp)
        
        ctx.save_for_backward(log_softmax, dim)
        return Tensor(log_softmax)
        
    @staticmethod
    def backward(ctx, grad_output):
        log_softmax, dim = ctx.saved_tensors
        softmax = np.exp(log_softmax)
        return grad_output - softmax * np.sum(grad_output, axis=dim, keepdims=True), None

# Public API wrappers mapping to Function nodes
def add(a, b): return Add.apply(_coerce(a), _coerce(b))
def mul(a, b): return Mul.apply(_coerce(a), _coerce(b))
def sub(a, b): return Sub.apply(_coerce(a), _coerce(b))
def matmul(a, b): return MatMul.apply(_coerce(a), _coerce(b))
def sum_op(a): return Sum.apply(_coerce(a))
def reshape(a, new_shape): return Reshape.apply(a, new_shape)
def transpose(a, dim0=0, dim1=1): return Transpose.apply(a, dim0, dim1)
def relu(a): return ReLU.apply(_coerce(a))
def pow_op(a, power): return Pow.apply(_coerce(a), power)
def log_softmax(a, dim=-1): return LogSoftmax.apply(_coerce(a), dim)
