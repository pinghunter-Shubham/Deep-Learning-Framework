class StepLR:
    """Decays the learning rate by gamma every step_size epochs."""
    def __init__(self, optimizer, step_size, gamma=0.1):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self._epoch = 0

    def step(self):
        self._epoch += 1
        if self._epoch % self.step_size == 0:
            self.optimizer.lr *= self.gamma

    def get_lr(self):
        return self.optimizer.lr
