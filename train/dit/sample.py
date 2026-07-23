"""Rectified-flow Euler sampling + grid rendering (shared by training eval,
bake-off comparison, reflow, and calibration)."""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


@torch.no_grad()
def euler_sample(model, z1, schedule, y=None):
    """Integrate v from t=1 (noise) to t=0. schedule: descending list of t values,
    e.g. [1.0, 0.75, 0.5, 0.25]; the last step integrates down to t=0.
    y: optional class labels for conditional models."""
    z = z1
    ts = list(schedule) + [0.0]
    for t, t_next in zip(ts[:-1], ts[1:]):
        tb = torch.full((z.shape[0],), t, device=z.device)
        v = model(z, tb, y) if y is not None else model(z, tb)
        z = z - (t - t_next) * v
    return z


def uniform_schedule(k):
    return [1.0 - i / k for i in range(k)]


@torch.no_grad()
def decode_latents(vae_decoder, z_norm, mean, std):
    """Normalized latents -> images in [-1,1]. mean/std: per-channel tensors."""
    z = z_norm * std[None, :, None, None] + mean[None, :, None, None]
    return vae_decoder(z).clamp(-1, 1)


def save_grid_gray(imgs, path, nrow=8):
    """imgs: (N,C,H,W) float in [-1,1], C in {1,3} (name kept from the gray era)."""
    from PIL import Image

    x = ((imgs.clamp(-1, 1) + 1) * 127.5).to(torch.uint8).cpu().numpy()
    n, c, h, w = x.shape
    x = x.transpose(0, 2, 3, 1)  # NHWC
    rows = (n + nrow - 1) // nrow
    grid = np.zeros((rows * h, nrow * w, c), dtype=np.uint8)
    for i in range(n):
        r, col = divmod(i, nrow)
        grid[r * h:(r + 1) * h, col * w:(col + 1) * w] = x[i]
    Image.fromarray(grid.squeeze()).save(path)
