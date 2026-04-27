import numpy as np
from .module import Module, Parameter
from minitorch.tensor import Tensor


class BatchNorm1d(Module):
    """
    Normalizes each feature across the batch during training.
    Uses running statistics for inference (eval mode).
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.gamma = Parameter(np.ones(num_features, dtype=np.float32))
        self.beta = Parameter(np.zeros(num_features, dtype=np.float32))

        # Running stats are not Parameters — not updated by the optimizer
        self.running_mean = np.zeros(num_features, dtype=np.float32)
        self.running_var = np.ones(num_features, dtype=np.float32)

    def forward(self, x):
        if self.training:
            # x: (N, F)  — compute mean and var per feature across the batch
            mean = x.mean(axis=0, keepdims=True)          # (1, F)
            diff = x - mean                                # (N, F)
            var = (diff ** 2).mean(axis=0, keepdims=True)  # (1, F)

            # Update running stats with plain numpy (no grad needed)
            self.running_mean = (
                (1 - self.momentum) * self.running_mean
                + self.momentum * mean.data[0]
            )
            self.running_var = (
                (1 - self.momentum) * self.running_var
                + self.momentum * var.data[0]
            )

            x_norm = diff / (var + self.eps) ** 0.5       # (N, F)
        else:
            x_norm = (
                (x - Tensor(self.running_mean))
                / Tensor(np.sqrt(self.running_var + self.eps))
            )

        return x_norm * self.gamma + self.beta
