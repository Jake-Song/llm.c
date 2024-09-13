import torch
from typing import Tuple
import math

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_scaling(freqs: torch.Tensor):
    # Values obtained from grid search
    scale_factor = 8
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192  # original llama3 length

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scale_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)

def precompute_freqs_cis(
    dim: int, end: int, theta: float = 10000.0, use_scaled: bool = False
):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    if use_scaled:
        freqs = apply_scaling(freqs)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

# create a small dummy example

n_embd = 4096
n_head = 32
block_size = 8192
rope_theta = 500000.0
use_scaled_rope = True

freqs_cis = precompute_freqs_cis(
            n_embd // n_head,
            block_size,
            rope_theta,
            use_scaled_rope,
        )

print(freqs_cis, '\n', freqs_cis.shape)
freqs_cis_real = torch.view_as_real(freqs_cis).flatten(1)
print(freqs_cis_real, '\n', freqs_cis_real.shape)

B = 4
T = block_size
NH = n_head
HS = n_embd // n_head

x = torch.randn(B, T, NH, HS)
q, k = apply_rotary_emb(x, x, freqs_cis=freqs_cis) 
print (q.shape, k.shape)

def write(tensor, handle):
    handle.write(tensor.detach().numpy().astype("float32").tobytes())

# Write to file
with open('ln.bin', 'wb') as file:
    write(freqs_cis_real, file)
    write(x, file)
    write(q, file) 
    write(k, file)
   