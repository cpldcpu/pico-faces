"""Encode the full dataset (plus horizontal flips) into VAE latents, compute
per-channel normalization stats, and save normalized fp16 latents for
rectified-flow training.

Layout: rows are interleaved [img0, img0_flip, img1, img1_flip, ...] so that a
"last N rows" validation split never mixes orientations of the same image
across train/val.

Run from repo root (WSL):  python train/latents.py --model m1_gray
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import rfpaths
from train.vae.model import build_vae


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    cfg = rfpaths.cfg(args.model, "vae")
    dev = "cuda"
    ckpt = torch.load(os.path.join(ROOT, cfg["out_dir"], "vae_final.pt"),
                      map_location=dev, weights_only=False)
    model = build_vae(cfg).to(dev)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # DiT latents may come from a subset of the VAE training data
    src = cfg.get("latents_data", cfg["data"])
    data = torch.from_numpy(np.load(rfpaths.resolve(src))).to(dev)
    data = data.unsqueeze(1) if data.ndim == 3 else data.permute(0, 3, 1, 2).contiguous()
    n = len(data)
    labels = None
    if "labels" in cfg:
        labels = np.load(rfpaths.resolve(cfg["labels"]))
        assert len(labels) == n, (len(labels), n)
    zc, zhw = cfg["latent_ch"], cfg["latent_hw"]

    # K rows per image, grouped so a "last N rows" val split never leaks an
    # image across train/val. Legacy (no aug_variants key): K=2, [img, flip].
    # With aug_variants: v0 = identity, v1.. = random flip + shift jitter
    # (reflect-pad by aug_pad px, random 128x128 crop).
    K = int(cfg.get("aug_variants", 0))
    P = int(cfg.get("aug_pad", 6))
    g = torch.Generator(device="cpu").manual_seed(int(cfg.get("seed", 0)))
    rows = (K or 2)
    out = torch.empty(rows * n, zc, zhw, zhw, dtype=torch.float16, device=dev)

    bs = 256
    for i in range(0, n, bs):
        x = data[i:i + bs].float() / 127.5 - 1
        B = x.shape[0]
        if K == 0:
            for f, off in ((False, 0), (True, 1)):
                mu, _ = model.encode(x.flip(-1) if f else x)
                out[2 * i + off:2 * (i + bs) + off:2] = mu.half()
            continue
        xpad = F.pad(x, (P, P, P, P), mode="reflect")
        H, W = x.shape[-2], x.shape[-1]
        for v in range(K):
            if v == 0:
                xv = x
            else:
                ox = torch.randint(0, 2 * P + 1, (B,), generator=g)
                oy = torch.randint(0, 2 * P + 1, (B,), generator=g)
                xv = torch.stack([xpad[b, :, oy[b]:oy[b] + H, ox[b]:ox[b] + W]
                                  for b in range(B)])
                flip = (torch.rand(B, generator=g) < 0.5).to(dev)
                xv[flip] = xv[flip].flip(-1)
            mu, _ = model.encode(xv)
            out[i * K + v:(i + B) * K + v:K] = mu.half()

    mean = out.float().mean(dim=(0, 2, 3))
    std = out.float().std(dim=(0, 2, 3))
    out = ((out.float() - mean[None, :, None, None]) / std[None, :, None, None]).half()
    print(f"latents {tuple(out.shape)}  mean {mean.tolist()}  std {std.tolist()}")

    arrays = dict(latents=out.cpu().numpy(), mean=mean.cpu().numpy(),
                  std=std.cpu().numpy())
    if labels is not None:
        arrays["labels"] = np.repeat(labels, rows).astype(np.int64)
    np.savez(os.path.join(ROOT, cfg["out_dir"], "latents.npz"), **arrays)
    print("saved", os.path.join(cfg["out_dir"], "latents.npz"))


if __name__ == "__main__":
    main()
