import sys
from minitorch.tensor import Tensor
import numpy as np

def debug_transpose():
    a = Tensor(np.zeros((3, 4)), requires_grad=True)
    out = a.transpose(0, 1)
    
    # Simulate testing.py flow
    out = out.sum()
    out.backward()

if __name__ == "__main__":
    debug_transpose()
