"""Rectified-flow training for the DiT (and the U-Net bake-off variant).

All latents stay resident on the GPU; each step samples indices, noises them,
and regresses v = eps - x0 with MSE. t ~ logit-normal(0,1) (SD3 recipe).

Optional Self-Flow training (arXiv 2603.06507, see doc/self_flow_plan.md),
enabled by dit.yaml keys — the default path is byte-identical to before:
  dual_timestep: true    # Dual-Timestep Scheduling: sample a second timestep
  mask_ratio: 0.25       # s and noise a random token subset at it (per-token
                         # adaLN conditioning; flow loss target unchanged)
  self_flow:             # implies dual_timestep; adds the representation loss:
    gamma: 0.8           # an EMA teacher sees the input noised at min(t,s)
    mask_ratio: 0.25     # everywhere; the student must reconstruct the
    l_student: 2         # teacher's block-l_teacher features from its mixed
    l_teacher: 6         # view, via a train-only projection head (cosine loss).
                         # Class dropout moves into the loop so teacher and
                         # student see identical labels.

Run from repo root (WSL):
  python train/dit/train_dit.py --model m3_long_cfg          # full run per config
  python train/dit/train_dit.py --model sf1_selfflow         # Self-Flow bake-off
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import rfpaths
from train.common.ema import EMA
from train.dit.model import build_model, device_params
from train.dit.sample import euler_sample, uniform_schedule, decode_latents, save_grid_gray
from train.vae.model import build_vae


def load_vae_decoder(dev, model_name, ckpt_path=None):
    vcfg = rfpaths.cfg(model_name, "vae")
    ckpt = torch.load(
        rfpaths.resolve(ckpt_path or os.path.join(vcfg["out_dir"], "vae_final.pt")),
        map_location=dev, weights_only=False,
    )
    vae = build_vae(vcfg).to(dev)
    vae.load_state_dict(ckpt["model"])
    vae.eval()
    return vae.decoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--arch", default=None)
    ap.add_argument("--act", default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--resume", default=None,
                    help="checkpoint to load model+EMA from before training "
                         "(a fresh AdamW is used -- opt state is not saved). "
                         "With cooldown_frac=1.0 this is a pure-cooldown "
                         "finetune of an already-converged model.")
    args = ap.parse_args()

    cfg = rfpaths.cfg(args.model, "dit")
    cfg["model"] = args.model  # provenance: saved into every checkpoint
    for k in ("arch", "act", "steps"):
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)
    out_dir = os.path.join(ROOT, args.out or cfg["out_dir"])
    os.makedirs(out_dir, exist_ok=True)
    dev = "cuda"
    torch.manual_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    z = np.load(rfpaths.resolve(cfg["latents"]))
    latents = torch.from_numpy(z["latents"]).to(dev)  # fp16, normalized
    n_val = cfg["n_val"]
    train, val = latents[:-n_val], latents[-n_val:]
    lat_mean = torch.from_numpy(z["mean"]).to(dev)
    lat_std = torch.from_numpy(z["std"]).to(dev)
    y_all = (torch.from_numpy(z["labels"]).long().to(dev)
             if cfg.get("n_classes") else None)
    y_train = y_all[:-n_val] if y_all is not None else None
    y_val = y_all[-n_val:] if y_all is not None else None
    print(f"latents train {len(train)}  val {len(val)}"
          + (f"  classes {torch.bincount(y_all).tolist()}" if y_all is not None else ""),
          flush=True)

    model = build_model(cfg).to(dev)
    print(f"{cfg['arch']}/{cfg['act']}: total "
          f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M  "
          f"device {device_params(model)/1e6:.2f}M", flush=True)

    if args.resume:
        rck = torch.load(rfpaths.resolve(args.resume), map_location=dev,
                         weights_only=False)
        clean = {k.replace("_orig_mod.", ""): v for k, v in rck["model"].items()}
        model.load_state_dict(clean)
        print(f"resumed model from {args.resume} (step {rck.get('step')}, "
              f"val_loss {rck.get('val_loss')})", flush=True)

    # --- Self-Flow / Dual-Timestep setup (default off; see module docstring)
    sf = cfg.get("self_flow")
    dual = bool(cfg.get("dual_timestep")) or sf is not None
    mask_ratio = float((sf or cfg).get("mask_ratio", 0.25))
    gside = cfg.get("latent_hw", 16) // cfg["dit"]["patch"]  # token grid (8)
    teacher = sf_head = None
    drop_p = 0.0
    if sf is not None:
        assert cfg["arch"] == "dit", "self_flow is DiT-only"
        l_s, l_t = int(sf["l_student"]), int(sf["l_teacher"])
        assert l_s < l_t < cfg["dit"]["depth"]
        sf_gamma = float(sf.get("gamma", 0.8))
        # class dropout moves into the loop: teacher and student must see
        # identical labels, so the model's own dropout is disabled
        drop_p, model.class_dropout = model.class_dropout, 0.0
        teacher = build_model(cfg).to(dev)
        teacher.load_state_dict(model.state_dict())
        teacher.requires_grad_(False)
        teacher.class_dropout = 0.0
        dim = cfg["dit"]["dim"]
        sf_head = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(),
                                nn.Linear(dim, dim)).to(dev)
        print(f"self_flow: gamma {sf_gamma}  mask {mask_ratio}  "
              f"layers {l_s}->{l_t} (teacher)", flush=True)
    elif dual:
        print(f"dual_timestep: mask {mask_ratio}", flush=True)

    if args.compile:
        model = torch.compile(model)
    ema = EMA(model, cfg["ema"])
    if args.resume and "ema" in rck:  # keep the converged EMA as the eval copy
        ema.load_state_dict({k.replace("_orig_mod.", ""): v
                             for k, v in rck["ema"].items()})
    eval_model = build_model(cfg).to(dev)
    eval_model.eval()

    vae_decoder = load_vae_decoder(dev, args.model, cfg.get("vae_ckpt"))

    params = list(model.parameters())
    if sf_head is not None:  # projection head trains too (never exported)
        params += list(sf_head.parameters())
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0)
    steps, bs = cfg["steps"], cfg["batch_size"]

    # fixed eval data: deterministic val (eps, t) pairs and grid noise
    g = torch.Generator(device="cpu").manual_seed(42)
    val_eps = torch.randn(len(val), *latents.shape[1:], generator=g).to(dev)
    val_t = torch.sigmoid(torch.randn(len(val), generator=g)).to(dev)
    grid_noise = torch.randn(64, *latents.shape[1:], generator=g).to(dev)
    # grid labels: contiguous blocks, one per class (incl. the null class)
    y_grid = ((torch.arange(64, device=dev) * (cfg["n_classes"] + 1)) // 64
              if y_all is not None else None)

    @torch.no_grad()
    def evaluate(step):
        eval_model.load_state_dict(
            {k.replace("_orig_mod.", ""): v for k, v in ema.state_dict().items()}
        )
        vloss, n = 0.0, 0
        for i in range(0, len(val), 512):
            x0 = val[i:i + 512].float()
            e, t = val_eps[i:i + 512], val_t[i:i + 512]
            zt = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * e
            v = (eval_model(zt, t, y_val[i:i + 512]) if y_val is not None
                 else eval_model(zt, t))
            vloss += F.mse_loss(v, e - x0, reduction="sum").item()
            n += x0.numel()
        vloss /= n
        for k_steps in (64, 4):
            zs = euler_sample(eval_model, grid_noise, uniform_schedule(k_steps), y_grid)
            imgs = decode_latents(vae_decoder, zs, lat_mean, lat_std)
            save_grid_gray(imgs, os.path.join(out_dir, f"grid_{step:06d}_k{k_steps}.png"))
        print(f"step {step:6d}  val v-loss {vloss:.5f}", flush=True)
        return vloss

    # LR: linear warmup, constant, then optional cosine cooldown over the last
    # cooldown_frac of steps down to cooldown_floor * lr (0 = no cooldown, the
    # historical constant-LR schedule -- byte-identical for models that omit it)
    cd_frac = float(cfg.get("cooldown_frac", 0.0))
    cd_floor = float(cfg.get("cooldown_floor", 0.05))
    cd_start = int(steps * (1.0 - cd_frac)) if cd_frac > 0 else steps + 1

    def lr_at(step):
        if step < cfg["warmup"]:
            return cfg["lr"] * step / cfg["warmup"]
        if step < cd_start:
            return cfg["lr"]
        prog = (step - cd_start) / max(1, steps - cd_start)  # 0->1
        cos = 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
        return cfg["lr"] * (cd_floor + (1.0 - cd_floor) * cos)

    t0 = time.time()
    run_loss = run_rep = 0.0
    for step in range(1, steps + 1):
        lr = lr_at(step)
        for gparam in opt.param_groups:
            gparam["lr"] = lr

        idx = torch.randint(0, len(train), (bs,), device=dev)
        x0 = train[idx].float()
        eps = torch.randn_like(x0)
        t = torch.sigmoid(torch.randn(bs, device=dev))
        y = y_train[idx] if y_train is not None else None

        if dual:
            # Dual-Timestep Scheduling: tokens in a random subset are noised
            # at a second timestep s; each token conditions on its own tau.
            # The flow target stays v = eps - x0 per token.
            s = torch.sigmoid(torch.randn(bs, device=dev))
            msk = torch.rand(bs, gside * gside, device=dev) < mask_ratio
            tau = torch.where(msk, s[:, None], t[:, None])       # (B, N)
            gmap = tau.view(bs, gside, gside)                     # token grid
            gmap = gmap.repeat_interleave(2, 1).repeat_interleave(2, 2)[:, None]
            zt = (1 - gmap) * x0 + gmap * eps
        else:
            zt = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * eps

        with torch.autocast("cuda", dtype=torch.bfloat16):
            if sf is not None:
                if drop_p > 0 and y is not None:  # loop-side class dropout,
                    dm = torch.rand(bs, device=dev) < drop_p  # shared t/s
                    y = torch.where(dm, torch.full_like(y, cfg["n_classes"]), y)
                v, feats = model(zt, tau, y, return_feats=[l_s])
                loss_gen = F.mse_loss(v.float(), (eps - x0))
                with torch.no_grad():  # teacher sees the CLEANER uniform view
                    tmin = torch.minimum(t, s)
                    z_min = ((1 - tmin[:, None, None, None]) * x0
                             + tmin[:, None, None, None] * eps)
                    _, tfeats = teacher(z_min, tmin, y, return_feats=[l_t])
                proj = sf_head(feats[l_s])
                loss_rep = -F.cosine_similarity(
                    proj.float(), tfeats[l_t].float(), dim=-1).mean()
                loss = loss_gen + sf_gamma * loss_rep
                run_rep += loss_rep.item()
            else:
                v = model(zt, tau if dual else t, y) if y is not None \
                    else model(zt, tau if dual else t)
                loss = F.mse_loss(v.float(), (eps - x0))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        ema.update(model)
        if teacher is not None:  # live EMA teacher, same decay as the eval EMA
            with torch.no_grad():
                for pt, ps in zip(teacher.parameters(), model.parameters()):
                    pt.mul_(cfg["ema"]).add_(ps, alpha=1 - cfg["ema"])
        # run_loss tracks the FLOW loss only, so it stays comparable across
        # baseline / dual-t / self-flow twins
        run_loss += (loss_gen if sf is not None else loss).item()

        if step % cfg["log_every"] == 0:
            ips = step * bs / (time.time() - t0)
            rep = (f"  rep {run_rep/cfg['log_every']:+.4f}"
                   if sf is not None else "")
            print(f"step {step:6d}  loss {run_loss/cfg['log_every']:.5f}{rep}"
                  f"  {ips:.0f} img/s", flush=True)
            run_loss = run_rep = 0.0
        if step % cfg["eval_every"] == 0 or step == steps:
            vloss = evaluate(step)
            ck = {"model": model.state_dict(), "ema": ema.state_dict(),
                  "cfg": cfg, "step": step, "val_loss": vloss}
            if sf_head is not None:  # train-only; downstream tools ignore it
                ck["sf_head"] = sf_head.state_dict()
            torch.save(ck, os.path.join(out_dir, "ckpt_last.pt"))

    ck = {"model": model.state_dict(), "ema": ema.state_dict(),
          "cfg": cfg, "step": steps, "val_loss": vloss}
    if sf_head is not None:
        ck["sf_head"] = sf_head.state_dict()
    torch.save(ck, os.path.join(out_dir, "ckpt_final.pt"))
    print(f"done in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
