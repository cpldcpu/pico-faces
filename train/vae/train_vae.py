"""Train the asymmetric VAE on FFHQ-gray-128.

The whole uint8 dataset (~1.15 GB) is kept resident on the GPU; batches are
sampled/augmented there, so there is no dataloader and no host I/O in the loop.

Run from repo root (WSL):  python train/vae/train_vae.py --model m1_gray
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from train.vae.model import build_vae


def save_grid(tensors, path, nrow=8):
    """tensors: (N,C,H,W) in [-1,1], C in {1,3} -> PNG grid."""
    x = ((tensors.clamp(-1, 1) + 1) * 127.5).to(torch.uint8).cpu().numpy()
    n, c, h, w = x.shape
    x = x.transpose(0, 2, 3, 1)  # NHWC
    rows = (n + nrow - 1) // nrow
    grid = np.zeros((rows * h, nrow * w, c), dtype=np.uint8)
    for i in range(n):
        r, col = divmod(i, nrow)
        grid[r * h:(r + 1) * h, col * w:(col + 1) * w] = x[i]
    Image.fromarray(grid.squeeze()).save(path)


@torch.no_grad()
def evaluate(model, val, batch=64):
    model.eval()
    mse_sum, n = 0.0, 0
    for i in range(0, len(val), batch):
        x = val[i:i + batch].float() / 127.5 - 1
        mu, _ = model.encode(x)
        rec = model.decoder(mu).clamp(-1, 1)
        mse_sum += F.mse_loss(rec, x, reduction="sum").item() / x[0].numel()
        n += x.shape[0]
    model.train()
    mse01 = (mse_sum / n) / 4.0  # [-1,1] -> [0,1] scale
    return -10 * np.log10(mse01)


def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    torch.manual_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = "cuda"
    out_dir = os.path.join(ROOT, cfg["out_dir"])
    os.makedirs(out_dir, exist_ok=True)

    data = torch.from_numpy(np.load(os.path.join(ROOT, cfg["data"]))).to(dev)
    # unify to (N,C,H,W) uint8 on the GPU: gray arrays are (N,H,W), RGB (N,H,W,3)
    data = data.unsqueeze(1) if data.ndim == 3 else data.permute(0, 3, 1, 2).contiguous()
    n_val = cfg["n_val"]
    train, val = data[:-n_val], data[-n_val:]
    print(f"train {len(train)}  val {len(val)}", flush=True)

    model = build_vae(cfg).to(dev)
    n_dec = sum(p.numel() for p in model.decoder.parameters())
    print(f"decoder params: {n_dec/1e3:.1f}K", flush=True)

    import lpips
    lpips_fn = lpips.LPIPS(net="vgg").to(dev).eval()
    for p in lpips_fn.parameters():
        p.requires_grad_(False)

    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.99))
    steps, bs = cfg["steps"], cfg["batch_size"]
    w_lpips, w_kl = cfg["lpips_weight"], cfg["kl_weight"]

    fixed_val = val[:32].float() / 127.5 - 1  # for recon grids

    t0 = time.time()
    for step in range(1, steps + 1):
        lr = cfg["lr"] * min(1.0, step / cfg["warmup"])
        for g in opt.param_groups:
            g["lr"] = lr

        idx = torch.randint(0, len(train), (bs,), device=dev)
        x = train[idx].float() / 127.5 - 1
        flip = torch.rand(bs, device=dev) < 0.5
        x[flip] = x[flip].flip(-1)

        rep = (lambda y: y.repeat(1, 3, 1, 1)) if x.shape[1] == 1 else (lambda y: y)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            rec, mu, logvar = model(x)
            l1 = (rec - x).abs().mean()
            lp = lpips_fn(rep(rec.clamp(-1, 1)), rep(x)).mean()
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=(1, 2, 3)).mean()
            loss = l1 + w_lpips * lp + w_kl * kl

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % cfg["log_every"] == 0:
            ips = step * bs / (time.time() - t0)
            print(
                f"step {step:6d}  l1 {l1.item():.4f}  lpips {lp.item():.4f} "
                f"kl {kl.item():9.1f}  {ips:.0f} img/s",
                flush=True,
            )
        if step % cfg["eval_every"] == 0 or step == steps:
            psnr = evaluate(model, val)
            print(f"step {step:6d}  val PSNR {psnr:.2f} dB", flush=True)
            with torch.no_grad():
                mu_g, _ = model.encode(fixed_val)
                rec_g = model.decoder(mu_g)
            save_grid(
                torch.cat([fixed_val, rec_g]),
                os.path.join(out_dir, f"recon_{step:06d}.png"),
            )
            torch.save(
                {"model": model.state_dict(), "cfg": cfg, "step": step, "psnr": psnr},
                os.path.join(out_dir, "vae_last.pt"),
            )

    torch.save({"model": model.state_dict(), "cfg": cfg, "psnr": psnr},
               os.path.join(out_dir, "vae_final.pt"))
    print(f"done in {(time.time()-t0)/60:.1f} min, final val PSNR {psnr:.2f} dB", flush=True)


if __name__ == "__main__":
    import rfpaths
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    main(os.path.join(rfpaths.model_dir(args.model), "vae.yaml"))
