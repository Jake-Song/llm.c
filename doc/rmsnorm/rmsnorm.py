import torch
import torch.nn as nn

eps = 1e-6

class RMSNorm:

    @staticmethod
    def forward(x, w, b: float = 0.0):
        B, T, C = x.size()
        x2 = x ** 2 # B,T,C
        x2_sum = x2.sum(-1, keepdim=True) # B,T,1
        mean = x2_sum / C # B,T,1
        rsqrt = (mean + eps) ** -0.5 # B,T,1
        norm = x * rsqrt # B,T,C
        out = norm * w + b # B,T,C

        cache = (x, w, mean, rsqrt)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        x, w, mean, rsqrt = cache
        # recompute the norm (save memory at the cost of compute)
        norm = x * rsqrt
        # gradients for weights, bias
        db = dout.sum((0, 1))
        dw = (dout * norm).sum((0, 1))
        # gradients for input
        dnorm = dout * w # B,T,C
        dx = dnorm * rsqrt # B,T,C
        dx = dx.clone()
        drsqrt = (dnorm * x).sum(-1, keepdim=True) # B,T,1
        dmean = drsqrt * -0.5 * (mean + eps) ** -1.5 # B,T,1
        dx2_sum = dmean * C ** -1 # B,T,1
        dx2 = dx2_sum * torch.ones_like(x) # B,T,C
        dx += dx2 * (2 * x) # B,T,C     
         
        return dx, dw, db


# create a small dummy example and check w.r.t PyTorch backward
B = 2
T = 3
C = 4
x = torch.randn(B, T, C, requires_grad=True)
w = torch.randn(C, requires_grad=True)
b = torch.randn(C, requires_grad=True)
out, cache = RMSNorm.forward(x, w, b)

dout = torch.randn(B, T, C)
dx, dw, db = RMSNorm.backward(dout, cache)

# compare to PyTorch autograd
fakeloss = (out * dout).sum()
fakeloss.backward()

print("dx error:", (x.grad - dx).abs().max().item())
print("dw error:", (w.grad - dw).abs().max().item())
print("db error:", (b.grad - db).abs().max().item())

# for reference checking in C also
x, w, mean, rsqrt = cache

def write(tensor, handle):
    handle.write(tensor.detach().numpy().astype("float32").tobytes())

# Write to file
with open('ln.bin', 'wb') as file:
    write(x, file) # (B, T, C)
    write(w, file) # (C, )
    write(b, file) # (C, )
    write(out, file) # (B, T, C)
    write(mean, file) # (B, T)
    write(rsqrt, file) # (B, T)
    write(dout, file) # (B, T, C)
    write(dx, file) # (B, T, C)
    write(dw, file) # (C, )
    write(db, file) # (C, )
