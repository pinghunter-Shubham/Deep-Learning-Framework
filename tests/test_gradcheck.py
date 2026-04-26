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
