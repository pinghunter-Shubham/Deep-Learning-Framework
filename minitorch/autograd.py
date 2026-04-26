import numpy as np

class Context:
    def __init__(self):
        self.saved_tensors = ()
        
    def save_for_backward(self, *args):
        self.saved_tensors = args

class Function:
    @classmethod
    def apply(cls, *inputs, **kwargs):
        from .tensor import Tensor
        ctx = Context()
        
        # Run forward pass (which returns a new Tensor)
        out = cls.forward(ctx, *inputs, **kwargs)
        
        # Are there any tensor inputs that require gradients?
        requires_grad = any(isinstance(t, Tensor) and t.requires_grad for t in inputs)
        
        out.requires_grad = requires_grad
        
        # Link graph
        if requires_grad:
            out._creator = cls
            out._ctx = ctx
            out._prev = [t for t in inputs if isinstance(t, Tensor)]
            
        return out

    @staticmethod
    def forward(ctx, *inputs, **kwargs):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError

def backward(tensor):
    # Topological sort
    topo = []
    visited = set()

    def build(v):
        if id(v) not in visited:
            visited.add(id(v))
            if hasattr(v, '_prev'):
                for child in v._prev:
                    build(child)
            topo.append(v)

    build(tensor)
    
    tensor.grad = np.ones(tensor.shape, dtype=np.float32)

    # Work queue simulation
    for node in reversed(topo):
        if not hasattr(node, '_creator') or node._creator is None:
            continue
            
        grads = node._creator.backward(node._ctx, node.grad)
        
        if not isinstance(grads, tuple):
            grads = (grads,)
            
        for prev_node, grad in zip(node._prev, grads):
            if prev_node.requires_grad and grad is not None:
                from .ops import unbroadcast
                if prev_node.grad is None:
                    prev_node.grad = np.zeros(prev_node.shape, dtype=np.float32)
                prev_node.grad += unbroadcast(grad, prev_node.shape)
