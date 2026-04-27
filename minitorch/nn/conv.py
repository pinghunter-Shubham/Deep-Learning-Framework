import numpy as np
from .module import Module, Parameter
from minitorch.tensor import Tensor
from minitorch.autograd import Function


# ---------------------------------------------------------------------------
# im2col / col2im helpers
# ---------------------------------------------------------------------------

def _make_indices(C, kH, kW, out_h, out_w, stride):
    """Precompute the index arrays shared by im2col and col2im."""
    i0 = np.tile(np.repeat(np.arange(kH), kW), C)           # (C*kH*kW,)
    i1 = stride * np.repeat(np.arange(out_h), out_w)        # (out_h*out_w,)
    j0 = np.tile(np.tile(np.arange(kW), kH), C)             # (C*kH*kW,)
    j1 = stride * np.tile(np.arange(out_w), out_h)          # (out_h*out_w,)
    c_idx = np.repeat(np.arange(C), kH * kW)                # (C*kH*kW,)
    rows = i0[None, :] + i1[:, None]   # (out_h*out_w, C*kH*kW)
    cols = j0[None, :] + j1[:, None]   # (out_h*out_w, C*kH*kW)
    return c_idx, rows, cols


def _im2col(x_pad, C, kH, kW, out_h, out_w, stride):
    """(N,C,H_pad,W_pad) → (N, out_h*out_w, C*kH*kW)"""
    c_idx, rows, cols = _make_indices(C, kH, kW, out_h, out_w, stride)
    return x_pad[:, c_idx, rows, cols]


def _col2im(col, N, C, H, W, kH, kW, out_h, out_w, stride, padding):
    """(N, out_h*out_w, C*kH*kW) → (N,C,H,W)"""
    H_pad = H + 2 * padding
    W_pad = W + 2 * padding
    c_idx, rows, cols = _make_indices(C, kH, kW, out_h, out_w, stride)
    # broadcast c_idx over the out_h*out_w dimension
    c_2d = np.broadcast_to(c_idx[None, :], (out_h * out_w, kH * kW * C))
    dx_pad = np.zeros((N, C, H_pad, W_pad), dtype=col.dtype)
    for n in range(N):
        np.add.at(dx_pad[n], (c_2d, rows, cols), col[n])
    if padding > 0:
        return dx_pad[:, :, padding:-padding, padding:-padding]
    return dx_pad


# ---------------------------------------------------------------------------
# Conv2d autograd Function
# ---------------------------------------------------------------------------

class Conv2dFunction(Function):
    @staticmethod
    def forward(ctx, x, weight, stride, padding):
        N, C_in, H, W = x.shape
        C_out, _, kH, kW = weight.shape

        out_h = (H + 2 * padding - kH) // stride + 1
        out_w = (W + 2 * padding - kW) // stride + 1

        x_pad = np.pad(x.data, ((0,0),(0,0),(padding,padding),(padding,padding))) if padding > 0 else x.data
        col = _im2col(x_pad, C_in, kH, kW, out_h, out_w, stride)   # (N, oh*ow, C_in*kH*kW)

        w_flat = weight.data.reshape(C_out, -1)                      # (C_out, C_in*kH*kW)
        out = col @ w_flat.T                                          # (N, oh*ow, C_out)
        # transpose then reshape produces a non-contiguous view; force a C-contiguous copy
        # so that Tensor(data) stores strides consistent with the flattened storage
        out = np.ascontiguousarray(
            out.transpose(0, 2, 1).reshape(N, C_out, out_h, out_w), dtype=np.float32
        )

        ctx.save_for_backward(col, w_flat, x.shape, weight.shape, stride, padding, out_h, out_w)
        return Tensor(out)

    @staticmethod
    def backward(ctx, grad_output):
        col, w_flat, x_shape, w_shape, stride, padding, out_h, out_w = ctx.saved_tensors
        N, C_in, H, W = x_shape
        C_out, _, kH, kW = w_shape

        # grad_output: (N, C_out, out_h, out_w) → (N, oh*ow, C_out)
        g = grad_output.reshape(N, C_out, -1).transpose(0, 2, 1)

        # gradient w.r.t. weight
        g_2d = g.reshape(N * out_h * out_w, C_out)
        col_2d = col.reshape(N * out_h * out_w, C_in * kH * kW)
        dw_flat = g_2d.T @ col_2d                                    # (C_out, C_in*kH*kW)
        dw = dw_flat.reshape(w_shape)

        # gradient w.r.t. input
        dcol = g @ w_flat                                            # (N, oh*ow, C_in*kH*kW)
        dx = _col2im(dcol, N, C_in, H, W, kH, kW, out_h, out_w, stride, padding)

        return dx.astype(np.float32), dw.astype(np.float32), None, None


# ---------------------------------------------------------------------------
# MaxPool2d autograd Function
# ---------------------------------------------------------------------------

class MaxPool2dFunction(Function):
    @staticmethod
    def forward(ctx, x, kernel_size, stride):
        N, C, H, W = x.shape
        kH, kW = kernel_size
        sH, sW = stride
        out_h = (H - kH) // sH + 1
        out_w = (W - kW) // sW + 1

        out = np.zeros((N, C, out_h, out_w), dtype=x.data.dtype)
        mask = np.zeros_like(x.data)

        for i in range(out_h):
            for j in range(out_w):
                window = x.data[:, :, i*sH:i*sH+kH, j*sW:j*sW+kW]
                max_val = window.max(axis=(2, 3), keepdims=True)
                out[:, :, i, j] = max_val[:, :, 0, 0]
                mask[:, :, i*sH:i*sH+kH, j*sW:j*sW+kW] += (window == max_val)

        ctx.save_for_backward(mask, x.shape, kernel_size, stride, out_h, out_w)
        return Tensor(out)

    @staticmethod
    def backward(ctx, grad_output):
        mask, x_shape, kernel_size, stride, out_h, out_w = ctx.saved_tensors
        kH, kW = kernel_size
        sH, sW = stride
        dx = np.zeros(x_shape, dtype=grad_output.dtype)

        for i in range(out_h):
            for j in range(out_w):
                dx[:, :, i*sH:i*sH+kH, j*sW:j*sW+kW] += (
                    mask[:, :, i*sH:i*sH+kH, j*sW:j*sW+kW]
                    * grad_output[:, :, i, j][:, :, None, None]
                )
        return dx, None, None


# ---------------------------------------------------------------------------
# nn Modules
# ---------------------------------------------------------------------------

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding

        kH, kW = self.kernel_size
        bound = 1.0 / np.sqrt(in_channels * kH * kW)
        self.weight = Parameter(
            np.random.uniform(-bound, bound, (out_channels, in_channels, kH, kW)).astype(np.float32)
        )
        if bias:
            self.bias = Parameter(np.zeros(out_channels, dtype=np.float32))
        else:
            self.bias = None

    def forward(self, x):
        out = Conv2dFunction.apply(x, self.weight, self.stride, self.padding)
        if self.bias is not None:
            # bias shape (C_out,) → (1, C_out, 1, 1) for broadcasting
            out = out + self.bias.reshape((1, self.out_channels, 1, 1))
        return out


class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if stride is not None else self.kernel_size
        if not isinstance(self.stride, tuple):
            self.stride = (self.stride, self.stride)

    def forward(self, x):
        return MaxPool2dFunction.apply(x, self.kernel_size, self.stride)
