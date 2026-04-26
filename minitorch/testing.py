import numpy as np

def grad_check(forward_fn, *args, eps=1e-3, atol=1e-2, rtol=1e-2):
    """
    Checks the analytical gradients of a function against numerical gradients defined by:
    df/dx ≈ (f(x+eps) - f(x-eps)) / (2*eps)
    """
    # 1. Compute analytical gradients
    out = forward_fn(*args)
    if not isinstance(out, float):
        out = out.sum()
        
    out.backward()
    
    # Collect analytical grads
    analytical_grads = []
    for arg in args:
        if hasattr(arg, 'requires_grad') and arg.requires_grad:
            analytical_grads.append(np.copy(arg.grad))
        else:
            analytical_grads.append(None)
            
    # 2. Compute numerical gradients
    for i, arg in enumerate(args):
        if not hasattr(arg, 'requires_grad') or not arg.requires_grad:
            continue
            
        numerical_grad = np.zeros_like(arg.data)
        it = np.nditer(arg.data, flags=['multi_index'], op_flags=['readwrite'])
        
        while not it.finished:
            idx = it.multi_index
            orig_val = arg.data[idx]
            
            # f(x + eps)
            arg.data[idx] = orig_val + eps
            out_pos = forward_fn(*args)
            if not isinstance(out_pos, float):
                out_pos = out_pos.sum().data.item()
                
            # f(x - eps)
            arg.data[idx] = orig_val - eps
            out_neg = forward_fn(*args)
            if not isinstance(out_neg, float):
                out_neg = out_neg.sum().data.item()
                
            numerical_grad[idx] = (out_pos - out_neg) / (2 * eps)
            
            # Restore
            arg.data[idx] = orig_val
            it.iternext()
            
        # 3. Compare
        if not np.allclose(analytical_grads[i], numerical_grad, atol=atol, rtol=rtol):
            print(f"Gradient check failed for argument {i}!")
            print(f"Analytical:\n{analytical_grads[i]}")
            print(f"Numerical:\n{numerical_grad}")
            return False
            
    return True
