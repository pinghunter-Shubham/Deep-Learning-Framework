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

class Softmax(Function):
    @staticmethod
    def forward(ctx, a, dim):
        max_a = np.max(a.data, axis=dim, keepdims=True)
        exp_a = np.exp(a.data - max_a)
        s = exp_a / np.sum(exp_a, axis=dim, keepdims=True)
        ctx.save_for_backward(s, dim)
        return Tensor(s)

    @staticmethod
    def backward(ctx, grad_output):
        s, dim = ctx.saved_tensors
        dot = np.sum(grad_output * s, axis=dim, keepdims=True)
        return s * (grad_output - dot), None


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

class Neg(Function):
    @staticmethod
    def forward(ctx, a):
        return Tensor(-a.data)

    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output


class Div(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a.data, b.data)
        return Tensor(a.data / b.data)

    @staticmethod
    def backward(ctx, grad_output):
        a_data, b_data = ctx.saved_tensors
        return grad_output / b_data, -grad_output * a_data / (b_data ** 2)


class Sigmoid(Function):
    @staticmethod
    def forward(ctx, a):
        sig = 1.0 / (1.0 + np.exp(-a.data))
        ctx.save_for_backward(sig)
        return Tensor(sig)

    @staticmethod
    def backward(ctx, grad_output):
        sig, = ctx.saved_tensors
        return grad_output * sig * (1.0 - sig)


class Tanh(Function):
    @staticmethod
    def forward(ctx, a):
        t = np.tanh(a.data)
        ctx.save_for_backward(t)
        return Tensor(t)

    @staticmethod
    def backward(ctx, grad_output):
        t, = ctx.saved_tensors
        return grad_output * (1.0 - t ** 2)


class Exp(Function):
    @staticmethod
    def forward(ctx, a):
        e = np.exp(a.data)
        ctx.save_for_backward(e)
        return Tensor(e)

    @staticmethod
    def backward(ctx, grad_output):
        e, = ctx.saved_tensors
        return grad_output * e


class Log(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a.data)
        return Tensor(np.log(a.data))

    @staticmethod
    def backward(ctx, grad_output):
        a_data, = ctx.saved_tensors
        return grad_output / a_data


class SumAxis(Function):
    @staticmethod
    def forward(ctx, a, axis, keepdims):
        ctx.save_for_backward(a.shape, axis, keepdims)
        return Tensor(a.data.sum(axis=axis, keepdims=keepdims))

    @staticmethod
    def backward(ctx, grad_output):
        a_shape, axis, keepdims = ctx.saved_tensors
        grad = grad_output
        if not keepdims:
            if axis is None:
                pass  # scalar grad broadcasts to any shape
            else:
                grad = np.expand_dims(grad, axis=axis)
        return np.broadcast_to(grad, a_shape).copy(), None, None


# Public API wrappers mapping to Function nodes
def add(a, b): return Add.apply(_coerce(a), _coerce(b))
def mul(a, b): return Mul.apply(_coerce(a), _coerce(b))
def sub(a, b): return Sub.apply(_coerce(a), _coerce(b))
def matmul(a, b): return MatMul.apply(_coerce(a), _coerce(b))
def neg(a): return Neg.apply(_coerce(a))
def div(a, b): return Div.apply(_coerce(a), _coerce(b))
def sigmoid(a): return Sigmoid.apply(_coerce(a))
def tanh_op(a): return Tanh.apply(_coerce(a))
def exp(a): return Exp.apply(_coerce(a))
def log(a): return Log.apply(_coerce(a))
def sum_op(a): return Sum.apply(_coerce(a))
def sum_axis(a, axis, keepdims=False): return SumAxis.apply(_coerce(a), axis, keepdims)
def reshape(a, new_shape): return Reshape.apply(a, new_shape)
def transpose(a, dim0=0, dim1=1): return Transpose.apply(a, dim0, dim1)
def relu(a): return ReLU.apply(_coerce(a))
def pow_op(a, power): return Pow.apply(_coerce(a), power)
def softmax_op(a, dim=-1): return Softmax.apply(_coerce(a), dim)
def log_softmax(a, dim=-1): return LogSoftmax.apply(_coerce(a), dim)
