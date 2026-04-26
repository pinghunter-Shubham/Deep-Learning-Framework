from minitorch.tensor import Tensor

class MSELoss:
    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        diff = pred - target
        return (diff ** 2).mean()

class CrossEntropyLoss:
    def __call__(self, logits: Tensor, target: Tensor) -> Tensor:
        """
        logits: (N, C) unnormalized scores
        target: (N, C) one-hot encoded true labels
        """
        from minitorch.ops import log_softmax
        log_probs = log_softmax(logits, dim=-1)
        # -sum(target * log_probs) / N
        return (target * log_probs).sum() * (-1.0 / logits.shape[0])
