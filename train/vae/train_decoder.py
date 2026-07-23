"""Decoder-only retrain: freeze a trained encoder, train a NEW (bigger) decoder
to reconstruct from the SAME latent space the shipping DiT already targets.

This isolates decoder capacity: because the encoder and its latent stats are
frozen (loaded from --enc-ckpt), the DiT that was trained against those latents
drops onto the new decoder unchanged. Any reconstruction-FID delta vs the old
decoder is the decoder alone.

The forward/loss mirror train_vae.py exactly (sampled z, L1 + LPIPS + tiny KL),
except the encoder params never receive gradients. KL is still reported but its
gradient only reaches the (frozen) encoder, so it's inert here -- kept for a
like-for-like log.

Run from repo root (WSL):
  python train/vae/train_decoder.py --model m3_decD --enc-ckpt checkpoints/m3_long/vae_final.pt
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
from train.vae.train_vae import save_grid, evaluate


def main(cfg_path, enc_ckpt):
    cfg = yaml.safe_load(open(cfg_path))
    torch.manual_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = "cuda"
    out_dir = os.path.join(ROOT, cfg["out_dir"])
    os.makedirs(out_dir, exist_ok=True)

    data = torch.from_numpy(np.load(os.path.join(ROOT, cfg["data"]))).to(dev)
    data = data.unsqueeze(1) if data.ndim == 3 else data.permute(0, 3, 1, 2).contiguous()
    n_val = cfg["n_val"]
    train, val = data[:-n_val], data[-n_val:]
    print(f"train {len(train)}  val {len(val)}", flush=True)

    model = build_vae(cfg).to(dev)

    # load the trained encoder (from a full-VAE checkpoint) and FREEZE it. The
    # decoder in `model` is the new (bigger) one from cfg["dec_plan"], randomly
    # initialised -- we only take encoder weights from the checkpoint.
    ck = torch.load(os.path.join(ROOT, enc_ckpt), map_location=dev, weights_only=False)
    sd = ck["model"] if "model" in ck else ck
    enc_sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    missing, unexpected = model.encoder.load_state_dict(enc_sd, strict=True)
    model.encoder.eval()
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_dec = sum(p.numel() for p in model.decoder.parameters())
    print(f"FROZEN encoder {n_enc/1e6:.2f}M (from {enc_ckpt})", flush=True)
    print(f"TRAIN  decoder {n_dec/1e3:.1f}K params ({len(cfg['dec_plan'])} conv layers)", flush=True)

    import lpips
    lpips_fn = lpips.LPIPS(net="vgg").to(dev).eval()
    for p in lpips_fn.parameters():
        p.requires_grad_(False)

    # optimise ONLY the decoder
    opt = torch.optim.Adam(model.decoder.parameters(), lr=cfg["lr"], betas=(0.9, 0.99))
    steps, bs = cfg["steps"], cfg["batch_size"]
    w_lpips, w_kl = cfg["lpips_weight"], cfg["kl_weight"]

    fixed_val = val[:32].float() / 127.5 - 1

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
            with torch.no_grad():
                mu, logvar = model.encode(x)
                z = model.reparameterize(mu, logvar)
            rec = model.decoder(z)
            l1 = (rec - x).abs().mean()
            lp = lpips_fn(rep(rec.clamp(-1, 1)), rep(x)).mean()
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=(1, 2, 3)).mean()
            loss = l1 + w_lpips * lp  # KL inert (encoder frozen); logged only

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % cfg["log_every"] == 0:
            ips = step * bs / (time.time() - t0)
            print(f"step {step:6d}  l1 {l1.item():.4f}  lpips {lp.item():.4f} "
                  f"kl {kl.item():9.1f}  {ips:.0f} img/s", flush=True)
        if step % cfg["eval_every"] == 0 or step == steps:
            psnr = evaluate(model, val)
            print(f"step {step:6d}  val PSNR {psnr:.2f} dB", flush=True)
            with torch.no_grad():
                mu_g, _ = model.encode(fixed_val)
                rec_g = model.decoder(mu_g)
            save_grid(torch.cat([fixed_val, rec_g]),
                      os.path.join(out_dir, f"recon_{step:06d}.png"))
            torch.save({"model": model.state_dict(), "cfg": cfg, "step": step,
                        "psnr": psnr, "enc_ckpt": enc_ckpt},
                       os.path.join(out_dir, "vae_last.pt"))

    torch.save({"model": model.state_dict(), "cfg": cfg, "psnr": psnr,
                "enc_ckpt": enc_ckpt}, os.path.join(out_dir, "vae_final.pt"))
    print(f"done in {(time.time()-t0)/60:.1f} min, final val PSNR {psnr:.2f} dB", flush=True)


if __name__ == "__main__":
    import rfpaths
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--enc-ckpt", required=True,
                    help="full-VAE checkpoint to take the FROZEN encoder from")
    args = ap.parse_args()
    main(os.path.join(rfpaths.model_dir(args.model), "vae.yaml"), args.enc_ckpt)
