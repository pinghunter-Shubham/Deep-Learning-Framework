import numpy as np
from minitorch import Tensor

def test_add_backward():
    a = Tensor([1.0, 2.0], requires_grad=True)
    b = Tensor([3.0, 4.0], requires_grad=True)
    
    c = a + b
    c.backward()
    
    assert np.allclose(a.grad, [1.0, 1.0])
    assert np.allclose(b.grad, [1.0, 1.0])

def test_mul_backward():
    a = Tensor([1.0, 2.0], requires_grad=True)
    b = Tensor([3.0, 4.0], requires_grad=False)
    
    c = a * b
    c.backward()
    
    assert np.allclose(a.grad, [3.0, 4.0])

def test_matmul_backward():
    a = Tensor([[1.0, 2.0]], requires_grad=True) # 1x2
    b = Tensor([[3.0], [4.0]], requires_grad=True) # 2x1
    
    c = a @ b
    c.backward()
    
    assert np.allclose(a.grad, [[3.0, 4.0]])
    assert np.allclose(b.grad, [[1.0], [2.0]])

def test_broadcasting_add():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True) # 2x2
    b = Tensor([1.0, 2.0], requires_grad=True) # 1x2 broadcasted to 2x2
    
    c = a + b
    d = c.sum()
    d.backward()
    
    assert np.allclose(a.grad, [[1.0, 1.0], [1.0, 1.0]])
    assert np.allclose(b.grad, [2.0, 2.0]) # Un-broadcast sum!
    
def test_linear_regression_mock():
    # Simple training loop mock to prove it works
    x = Tensor([[1.0], [2.0], [3.0]])
    y = Tensor([[2.0], [4.0], [6.0]])
    
    w = Tensor([[0.1]], requires_grad=True)
    
    for _ in range(50):
        # zero grad
        w.grad = np.zeros_like(w.data)
        
        pred = x @ w
        diff = pred - y
        loss = (diff * diff).sum()
        loss.backward()
        
        # SGD
        w.data = w.data - 0.01 * w.grad
        
    print(f"Final w: {w.data}")
    # w should approach 2.0
    assert w.data[0,0] > 1.8

def test_reshape_view():
    a = Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    b = a.reshape((2, 2))
    
    assert b.shape == (2, 2)
    assert b.strides == (2, 1) # (2, 1) elements = (8, 4) bytes
    assert np.allclose(b.data, [[1.0, 2.0], [3.0, 4.0]])
    
    # Prove zero copy sharing
    b.data[0, 0] = 99.0
    assert a.data[0] == 99.0
    
    # Test gradients flow through reshape
    c = b.sum()
    c.backward()
    assert np.allclose(a.grad, [1.0, 1.0, 1.0, 1.0])

def test_transpose_view():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = a.transpose(0, 1)
    
    assert b.shape == (2, 2)
    assert b.strides == (1, 2) # Inherits shape/strides swapped
    assert np.allclose(b.data, [[1.0, 3.0], [2.0, 4.0]])
    
    # Prove zero copy
    b.data[0, 0] = 99.0
    assert a.data[0, 0] == 99.0
    
    # Test gradients flow through transpose
    w = Tensor([[1.0, 1.0], [1.0, 1.0]])
    c = (b * w).sum()
    c.backward()
    assert np.allclose(a.grad, [[1.0, 1.0], [1.0, 1.0]])

