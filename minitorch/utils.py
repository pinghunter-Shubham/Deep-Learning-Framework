import numpy as np


def save_model(model, path):
    """Save all model parameters to a .npz file."""
    if not path.endswith('.npz'):
        path += '.npz'
    weights = {f'p{i}': p.data for i, p in enumerate(model.parameters())}
    np.savez(path, **weights)


def load_model(model, path):
    """Load parameters from a .npz file into the model in-place."""
    if not path.endswith('.npz'):
        path += '.npz'
    ckpt = np.load(path)
    for i, p in enumerate(model.parameters()):
        p.data = ckpt[f'p{i}'].astype(np.float32)
