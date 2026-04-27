import numpy as np
import pytest
from minitorch.tensor import Tensor
from minitorch.testing import grad_check

def test_grad_check_add():
    a = Tensor(np.random.randn(3, 3), requires_grad=True)
    b = Tensor(np.random.randn(3, 3), requires_grad=True)
    assert grad_check(lambda x, y: x + y, a, b)

def test_grad_check_broadcast_add():
    a = Tensor(np.random.randn(3, 3), requires_grad=True)
    b = Tensor(np.random.randn(3,), requires_grad=True)
    assert grad_check(lambda x, y: x + y, a, b)

def test_grad_check_mul():
    a = Tensor(np.random.randn(3, 3), requires_grad=True)
    b = Tensor(np.random.randn(3, 3), requires_grad=True)
    assert grad_check(lambda x, y: x * y, a, b)

def test_grad_check_sub():
    a = Tensor(np.random.randn(3, 3), requires_grad=True)
    b = Tensor(np.random.randn(3, 3), requires_grad=True)
    assert grad_check(lambda x, y: x - y, a, b)

def test_grad_check_matmul():
    a = Tensor(np.random.randn(3, 4), requires_grad=True)
    b = Tensor(np.random.randn(4, 2), requires_grad=True)
    assert grad_check(lambda x, y: x @ y, a, b)

def test_grad_check_reshape():
    a = Tensor(np.random.randn(4, 4), requires_grad=True)
    assert grad_check(lambda x: x.reshape((2, 8)), a)

def test_grad_check_transpose():
    a = Tensor(np.random.randn(3, 4), requires_grad=True)
    assert grad_check(lambda x: x.transpose(0, 1), a)

def test_grad_check_complex_graph():
    a = Tensor(np.random.randn(3, 4), requires_grad=True)
    b = Tensor(np.random.randn(4, 2), requires_grad=True)
    c = Tensor(np.random.randn(3, 2), requires_grad=True)

    # y = (a @ b) + c
    assert grad_check(lambda x, y, z: (x @ y) + z, a, b, c)

def test_grad_check_neg():
    a = Tensor(np.random.randn(3, 3), requires_grad=True)
    assert grad_check(lambda x: -x, a)

def test_grad_check_div():
    a = Tensor(np.random.randn(3, 3), requires_grad=True)
    # Keep denominator away from zero
    b = Tensor(np.random.randn(3, 3) + 2.0, requires_grad=True)
    assert grad_check(lambda x, y: x / y, a, b)

def test_grad_check_sigmoid():
    a = Tensor(np.random.randn(3, 3), requires_grad=True)
    assert grad_check(lambda x: x.sigmoid(), a)

def test_grad_check_tanh():
    a = Tensor(np.random.randn(3, 3), requires_grad=True)
    assert grad_check(lambda x: x.tanh(), a)

def test_grad_check_relu():
    # Avoid inputs near 0 where relu is non-differentiable
    a = Tensor(np.random.randn(3, 3) + 1.0, requires_grad=True)
    assert grad_check(lambda x: x.relu(), a)

def test_grad_check_exp():
    a = Tensor(np.random.randn(3, 3), requires_grad=True)
    assert grad_check(lambda x: x.exp(), a)

def test_grad_check_log():
    # Keep inputs positive for valid log
    a = Tensor(np.abs(np.random.randn(3, 3)) + 0.5, requires_grad=True)
    assert grad_check(lambda x: x.log(), a)

def test_grad_check_sum_axis():
    a = Tensor(np.random.randn(3, 4), requires_grad=True)
    assert grad_check(lambda x: x.sum(axis=0), a)

def test_grad_check_sum_axis_keepdims():
    a = Tensor(np.random.randn(3, 4), requires_grad=True)
    assert grad_check(lambda x: x.sum(axis=1, keepdims=True), a)

def test_grad_check_softmax():
    a = Tensor(np.random.randn(4, 6), requires_grad=True)
    assert grad_check(lambda x: x.softmax(dim=-1), a)
