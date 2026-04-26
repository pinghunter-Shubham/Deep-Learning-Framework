import numpy as np
from minitorch.tensor import Tensor
from minitorch.testing import grad_check

def debug_broadcast():
    a = Tensor(np.ones((3, 3)), requires_grad=True)
    b = Tensor(np.ones((3,)), requires_grad=True)
    
    out = a + b
    out.sum().backward()
    
    print("A Analytical:")
    print(a.grad)
    print("B Analytical:")
    print(b.grad)
    
    print("Running grad check:")
    res = grad_check(lambda x,y: (x+y).sum(), a, b)
    print("Grad check returned:", res)

if __name__ == "__main__":
    debug_broadcast()
