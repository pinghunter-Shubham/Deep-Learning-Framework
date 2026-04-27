from .module import Module


class Flatten(Module):
    """Reshape (N, *) → (N, product_of_remaining_dims)."""
    def forward(self, x):
        n = x.shape[0]
        flat_dim = 1
        for d in x.shape[1:]:
            flat_dim *= d
        return x.reshape((n, flat_dim))
