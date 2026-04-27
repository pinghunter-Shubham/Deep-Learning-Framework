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

def test_activation_modules():
    x = Tensor([[1.0, -1.0], [0.0, 2.0]])

    relu = nn.ReLU()
    out = relu(x)
    assert np.allclose(out.data, [[1.0, 0.0], [0.0, 2.0]])

    sig = nn.Sigmoid()
    out = sig(x)
    expected = 1.0 / (1.0 + np.exp(-x.data))
    assert np.allclose(out.data, expected)

    tanh = nn.Tanh()
    out = tanh(x)
    assert np.allclose(out.data, np.tanh(x.data))

def test_sequential_with_activations():
    np.random.seed(0)
    model = nn.Sequential(
        nn.Linear(2, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
        nn.Sigmoid(),
    )
    optimizer = optim.Adam(model.parameters(), lr=0.05)
    criterion = nn.MSELoss()

    X = Tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    Y = Tensor([[0.0], [1.0], [1.0], [0.0]])

    for _ in range(200):
        optimizer.zero_grad()
        loss = criterion(model(X), Y)
        loss.backward()
        optimizer.step()

    pred = model(X)
    assert pred.data[0, 0] < 0.3
    assert pred.data[1, 0] > 0.7
    assert pred.data[2, 0] > 0.7
    assert pred.data[3, 0] < 0.3

def test_dropout_zeroes_in_training():
    np.random.seed(42)
    dropout = nn.Dropout(p=0.9)
    dropout.train()
    x = Tensor(np.ones((100, 100)))
    out = dropout(x)
    # With p=0.9 most values should be zero
    zero_fraction = np.mean(out.data == 0.0)
    assert zero_fraction > 0.7

def test_dropout_passthrough_in_eval():
    dropout = nn.Dropout(p=0.9)
    dropout.eval()
    x = Tensor(np.ones((10, 10)))
    out = dropout(x)
    assert np.allclose(out.data, x.data)

def test_dropout_gradient_flows():
    np.random.seed(1)
    dropout = nn.Dropout(p=0.5)
    dropout.train()
    x = Tensor(np.ones((4, 4)), requires_grad=True)
    out = dropout(x)
    out.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape

# ---------------------------------------------------------------------------
# Softmax
# ---------------------------------------------------------------------------

def test_softmax_in_network():
    np.random.seed(0)
    model = nn.Sequential(nn.Linear(4, 3))
    x = Tensor(np.random.randn(5, 4).astype(np.float32))
    logits = model(x)
    probs = logits.softmax(dim=-1)
    # Each row sums to 1
    assert np.allclose(probs.data.sum(axis=1), np.ones(5), atol=1e-5)

# ---------------------------------------------------------------------------
# BatchNorm1d
# ---------------------------------------------------------------------------

def test_batchnorm1d_training():
    np.random.seed(42)
    bn = nn.BatchNorm1d(4)
    x = Tensor(np.random.randn(8, 4).astype(np.float32), requires_grad=True)
    bn.train()
    out = bn(x)
    assert out.shape == (8, 4)
    out.sum().backward()
    assert x.grad is not None

def test_batchnorm1d_running_stats():
    np.random.seed(42)
    bn = nn.BatchNorm1d(4)
    x = Tensor(np.random.randn(16, 4).astype(np.float32))
    bn.train()
    for _ in range(20):
        bn(x)
    # Running mean should be close to batch mean after many updates
    assert np.allclose(bn.running_mean, x.data.mean(axis=0), atol=0.2)

def test_batchnorm1d_eval():
    np.random.seed(42)
    bn = nn.BatchNorm1d(4)
    x = Tensor(np.random.randn(8, 4).astype(np.float32))
    bn.train()
    for _ in range(50):
        bn(x)
    bn.eval()
    out = bn(x)
    assert out.shape == (8, 4)

# ---------------------------------------------------------------------------
# Conv2d
# ---------------------------------------------------------------------------

def test_conv2d_output_shape():
    np.random.seed(0)
    conv = nn.Conv2d(1, 8, kernel_size=3, padding=1)
    x = Tensor(np.random.randn(4, 1, 28, 28).astype(np.float32))
    out = conv(x)
    assert out.shape == (4, 8, 28, 28)  # same spatial size with padding=1

def test_conv2d_no_padding_shape():
    conv = nn.Conv2d(3, 16, kernel_size=3, padding=0)
    x = Tensor(np.random.randn(2, 3, 8, 8).astype(np.float32))
    out = conv(x)
    assert out.shape == (2, 16, 6, 6)

def test_conv2d_gradient_flows():
    np.random.seed(1)
    conv = nn.Conv2d(1, 2, kernel_size=3, padding=1)
    x = Tensor(np.random.randn(2, 1, 5, 5).astype(np.float32), requires_grad=True)
    out = conv(x)
    out.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    w = list(conv.parameters())[0]
    assert w.grad is not None

def test_conv2d_1x1_matches_linear():
    # A 1×1 conv with no bias is equivalent to a linear projection per pixel
    np.random.seed(2)
    N, C_in, H, W = 1, 3, 2, 2
    C_out = 4
    conv = nn.Conv2d(C_in, C_out, kernel_size=1, bias=False)
    x = Tensor(np.random.randn(N, C_in, H, W).astype(np.float32))
    out = conv(x)          # (1, 4, 2, 2)
    # Manual: w shape (C_out, C_in, 1, 1) → apply per position
    w = conv.weight.data.reshape(C_out, C_in)
    x_flat = x.data.reshape(N, C_in, H * W)        # (1, 3, 4)
    expected = (w @ x_flat).reshape(N, C_out, H, W) # (1, 4, 2, 2)
    assert np.allclose(out.data, expected, atol=1e-5)

# ---------------------------------------------------------------------------
# MaxPool2d
# ---------------------------------------------------------------------------

def test_maxpool2d_forward():
    pool = nn.MaxPool2d(kernel_size=2, stride=2)
    x = Tensor(np.array([[[[1., 2., 3., 4.],
                            [5., 6., 7., 8.],
                            [9., 10., 11., 12.],
                            [13., 14., 15., 16.]]]]))
    out = pool(x)
    assert out.shape == (1, 1, 2, 2)
    assert np.allclose(out.data, [[[[6., 8.], [14., 16.]]]])

def test_maxpool2d_gradient_flows():
    np.random.seed(3)
    pool = nn.MaxPool2d(kernel_size=2, stride=2)
    x = Tensor(np.random.randn(2, 1, 4, 4).astype(np.float32), requires_grad=True)
    out = pool(x)
    out.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape

# ---------------------------------------------------------------------------
# Flatten
# ---------------------------------------------------------------------------

def test_flatten():
    flat = nn.Flatten()
    x = Tensor(np.ones((3, 2, 4, 4), dtype=np.float32))
    out = flat(x)
    assert out.shape == (3, 32)

def test_flatten_gradient_flows():
    flat = nn.Flatten()
    x = Tensor(np.ones((2, 3, 3), dtype=np.float32), requires_grad=True)
    out = flat(x)
    out.sum().backward()
    assert np.allclose(x.grad, np.ones_like(x.data))

# ---------------------------------------------------------------------------
# Conv2d + MaxPool2d + Flatten end-to-end
# ---------------------------------------------------------------------------

def test_conv_pool_flatten_pipeline():
    np.random.seed(5)
    model = nn.Sequential(
        nn.Conv2d(1, 4, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(4 * 14 * 14, 10),
    )
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    x = Tensor(np.random.randn(4, 1, 28, 28).astype(np.float32))
    y = Tensor(np.eye(10, dtype=np.float32)[[0, 1, 2, 3]])
    criterion = nn.CrossEntropyLoss()

    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()
    assert loss.data.item() > 0.0

# ---------------------------------------------------------------------------
# SGD momentum
# ---------------------------------------------------------------------------

def test_sgd_momentum_updates_correctly():
    # Verify that velocity accumulates and parameters shift more with momentum
    np.random.seed(7)
    model = nn.Linear(2, 1)
    for p in model.parameters():
        p.data[:] = 0.0
    opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    X = Tensor([[1.0, 1.0]])
    Y = Tensor([[1.0]])
    criterion = nn.MSELoss()

    # Step 1
    opt.zero_grad()
    criterion(model(X), Y).backward()
    opt.step()
    params_after_1 = [p.data.copy() for p in model.parameters()]

    # Step 2 — momentum should amplify the update
    opt.zero_grad()
    criterion(model(X), Y).backward()
    opt.step()
    params_after_2 = [p.data.copy() for p in model.parameters()]

    # Each parameter should have moved further in step 2 than step 1 (velocity builds up)
    for p1, p2, p0 in zip(params_after_1, params_after_2, [np.zeros_like(p.data) for p in model.parameters()]):
        delta1 = np.abs(p1 - p0).sum()
        delta2 = np.abs(p2 - p1).sum()
        assert delta2 > delta1 * 0.5   # second step is non-trivial

# ---------------------------------------------------------------------------
# StepLR scheduler
# ---------------------------------------------------------------------------

def test_step_lr():
    model = nn.Linear(2, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    scheduler = optim.StepLR(optimizer, step_size=2, gamma=0.5)
    scheduler.step()
    assert abs(optimizer.lr - 0.1) < 1e-9   # no decay yet (epoch 1)
    scheduler.step()
    assert abs(optimizer.lr - 0.05) < 1e-9  # halved at epoch 2
    scheduler.step()
    assert abs(optimizer.lr - 0.05) < 1e-9  # no decay (epoch 3)
    scheduler.step()
    assert abs(optimizer.lr - 0.025) < 1e-9 # halved at epoch 4

# ---------------------------------------------------------------------------
# Model save / load
# ---------------------------------------------------------------------------

def test_model_save_load():
    import os, tempfile, minitorch
    np.random.seed(9)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    x = Tensor(np.ones((3, 4), dtype=np.float32))
    out_before = model(x).data.copy()

    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        path = f.name

    try:
        minitorch.save_model(model, path)
        # Corrupt weights
        for p in model.parameters():
            p.data += 99.0
        minitorch.load_model(model, path)
        out_after = model(x).data
        assert np.allclose(out_before, out_after, atol=1e-5)
    finally:
        os.unlink(path)
