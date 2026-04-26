import numpy as np
import pytest
from minitorch.tensor import Tensor
import minitorch.nn as nn
import minitorch.optim as optim

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(2, 8)
        self.l2 = nn.Linear(8, 1)
        
    def forward(self, x):
        return self.l2(self.l1(x).relu())

def test_mlp_xor_sgd():
    np.random.seed(42)
    model = MLP()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.MSELoss()
    
    # XOR dataset
    X = Tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    Y = Tensor([[0.0], [1.0], [1.0], [0.0]])
    
    for epoch in range(200):
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, Y)
        loss.backward()
        optimizer.step()
        
    pred = model(X)
    assert pred.data[0, 0] < 0.2
    assert pred.data[1, 0] > 0.8
    assert pred.data[2, 0] > 0.8
    assert pred.data[3, 0] < 0.2

def test_mlp_xor_adam():
    np.random.seed(42)
    model = MLP()
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    criterion = nn.MSELoss()
    
    # XOR dataset
    X = Tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    Y = Tensor([[0.0], [1.0], [1.0], [0.0]])
    
    for epoch in range(100):
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, Y)
        loss.backward()
        optimizer.step()
        
    pred = model(X)
    assert pred.data[0, 0] < 0.2
    assert pred.data[1, 0] > 0.8
    assert pred.data[2, 0] > 0.8
    assert pred.data[3, 0] < 0.2
