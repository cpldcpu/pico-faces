import numpy as np
import torch


def sincos_2d(dim, h, w):
    """Fixed 2D sin-cos positional embedding, shape (h*w, dim).

    dim is split evenly between the y and x axes; each half is the standard
    1D sin/cos embedding of the row/col index. Matches the DiT/MAE convention.
    """
    assert dim % 4 == 0
    d4 = dim // 4
    omega = 1.0 / (10000 ** (np.arange(d4, dtype=np.float64) / d4))
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    def emb(pos):
        out = np.einsum("p,d->pd", pos.reshape(-1).astype(np.float64), omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    e = np.concatenate([emb(ys), emb(xs)], axis=1)  # (h*w, dim)
    return torch.from_numpy(e).float()


def timestep_embedding(t, dim, max_period=10000):
    """Sinusoidal timestep embedding. t: (B,) float in [0,1] -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -np.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t.float()[:, None] * freqs[None] * 1000.0  # scale t to ~[0,1000] like DDPM
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
