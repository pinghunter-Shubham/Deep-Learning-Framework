import numpy as np

class Storage:
    """
    Storage represents the actual memory buffer.
    For now, it wraps a flattened 1D NumPy array simulating contiguous memory.
    """
    def __init__(self, data):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
            
        # Ensure it's stored conceptually as a contiguous 1D memory array buffer
        self.data = np.ascontiguousarray(data).flatten()
        
    def __repr__(self):
        return f"Storage({self.data})"
