"""QAT finetune: continue training the exported checkpoint with fake-quant
in the loop (quant/fake_quant.py), so the weights adapt to the int8 grid.
Frozen activation scales from the PTQ calibration; same RF objective and
data as train_dit. Saves a plain DiT checkpoint fold.py can consume.

Run from repo root (WSL):
  python quant/qat.py --model m3_faces
Then: calibrate.py --ckpt <qat ckpt> -> fold.py --ckpt <qat ckpt> -> verify.
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import rfpaths
from quant.fake_quant import QATDiT, headroom_scales
from train.common.ema import EMA
from train.dit.model import build_model
from train.dit.sample import (decode_latents, euler_sample, save_grid_gray,
                              uniform_schedule)
from train.dit.train_dit import load_vae_decoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ckpt", default=None)    # base ckpt (default: export.yaml)
    ap.add_argument("--calib", default=None)   # frozen scales (default: export.yaml)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--headroom", action="store_true",
                    help="train against fold's post-headroom DEPLOYED scales "
                         "and freeze the conditioning tower (so fold "
                         "recomputes the identical inflation)")
    ap.add_argument("--distill", action="store_true",
                    help="self-distillation: train the fake-quant student to "
                         "match the FROZEN fp teacher's velocity instead of "
                         "the data target. Anchors the model at its current "
                         "functional point (data-target QAT drifts the cooled "
                         "EMA optimum: fq val-loss improves but FID worsens); "
                         "the objective is exactly 'int8 latents == fp "
                         "latents'.")
    ap.add_argument("--cosine", action="store_true",
                    help="cosine LR decay to 0.1*lr after warmup (long runs; "
                         "the flat-LR 15k recipe leaves the tail unsettled)")
    ap.add_argument("--traj-pool", type=int, default=0, metavar="N",
                    help="distill only: mix 50%% of each batch from a pool of "
                         "TEACHER TRAJECTORY states (N seeds per w in "
                         "{plain,4,6,8}, K=8 Euler) instead of pure data-noise "
                         "interpolants -- covers the z-distribution the "
                         "deployed sampler actually visits (esp. guided "
                         "high-w states, where the int8 residual is largest; "
                         "same insight as the union calibration)")
    ap.add_argument("--teacher-ckpt", default=None,
                    help="distill teacher checkpoint (default: the student's "
                         "start ckpt). REQUIRED when continuing from a distill "
                         "ckpt: the teacher must stay the base fp model, or "
                         "distill-of-distill compounds its errors")
    args = ap.parse_args()
    assert not args.traj_pool or args.distill, "--traj-pool requires --distill"

    dev = "cuda"
    exp = rfpaths.cfg(args.model, "export")
    steps = args.steps or int(exp.get("qat_steps", 15000))
    lr0 = float(exp.get("qat_lr", 1e-5))
    warmup = 200
    out_dir = rfpaths.resolve(args.out
                              or rfpaths.art(args.model, "runs", "qat"))
    os.makedirs(out_dir, exist_ok=True)

    ckpt = torch.load(rfpaths.resolve(args.ckpt or exp["checkpoint"]),
                      map_location=dev, weights_only=False)
    cfg = ckpt["cfg"]
    assert cfg["arch"] == "dit"
    # keep <name>__pc arrays: QATDiT builds per-channel grids from them (v7)
    scales = dict(np.load(rfpaths.resolve(args.calib or exp["calib"])).items())
    act = cfg.get("act", "relu2")
    drop_attn = set(exp.get("drop_attn", []))
    if drop_attn:
        print(f"drop_attn blocks {sorted(drop_attn)}: attention branch "
              "skipped, QAT heals the surgery", flush=True)

    # start from the EMA weights (that is what was exported and evaluated)
    model = build_model(cfg).to(dev)
    model.load_state_dict(
        {k.replace("_orig_mod.", ""): v for k, v in ckpt["ema"].items()})
    if args.headroom:
        scales = headroom_scales(scales, model, exp["schedule"])
        frozen = [model.t_mlp, model.final_mod] + \
                 [blk.mod for blk in model.blocks] + \
                 ([model.y_emb] if cfg.get("n_classes") else [])
        for mod in frozen:
            mod.requires_grad_(False)
        print("headroom scales applied; conditioning tower frozen", flush=True)
    qat = QATDiT(model, scales, act=act, drop_attn=drop_attn).train()
    ema = EMA(model, cfg["ema"])  # tracks the inner (plain-DiT) params

    eval_dit = build_model(cfg).to(dev).eval()
    eval_qat = QATDiT(eval_dit, scales, act=act, drop_attn=drop_attn).eval()

    teacher = None
    if args.distill:
        import types

        def fwd_drop(self, x, c):
            s1, b1, g1, s2, b2, g2 = self.mod(c)[:, None].chunk(6, dim=-1)
            if not self._drop_attn:
                x = x + g1 * self.attn(self.norm1(x) * (1 + s1) + b1)
            x = x + g2 * self.mlp(self.norm2(x) * (1 + s2) + b2)
            return x

        tck = ckpt
        if args.teacher_ckpt:
            tck = torch.load(rfpaths.resolve(args.teacher_ckpt),
                             map_location=dev, weights_only=False)
        teacher = build_model(cfg).to(dev).eval()
        teacher.load_state_dict(
            {k.replace("_orig_mod.", ""): v for k, v in tck["ema"].items()})
        for i, blk in enumerate(teacher.blocks):  # same slim arch as student
            blk._drop_attn = i in drop_attn
            blk.forward = types.MethodType(fwd_drop, blk)
        teacher.requires_grad_(False)
        # class dropout moves to the training loop so teacher and student see
        # the SAME (possibly nulled) labels; the wrapper must not re-drop.
        class_dropout, model.class_dropout = model.class_dropout, 0.0
        m_ncls = model.n_classes
        print("distillation mode: frozen fp teacher, matching velocities",
              flush=True)

    torch.manual_seed(cfg["seed"] + 1)
    z = np.load(rfpaths.resolve(cfg["latents"]))
    latents = torch.from_numpy(z["latents"]).to(dev)
    n_val = cfg["n_val"]
    train, val = latents[:-n_val], latents[-n_val:]
    lat_mean = torch.from_numpy(z["mean"]).to(dev)
    lat_std = torch.from_numpy(z["std"]).to(dev)
    y_all = (torch.from_numpy(z["labels"]).long().to(dev)
             if cfg.get("n_classes") else None)
    y_train = y_all[:-n_val] if y_all is not None else None
    y_val = y_all[-n_val:] if y_all is not None else None
    vae_decoder = load_vae_decoder(dev, args.model,
                                   cfg.get("vae_ckpt")
                                   or exp.get("vae_checkpoint"))

    pool = None
    if args.traj_pool:
        # fp teacher trajectory states (plain + guided): the z-distribution
        # the deployed sampler actually visits, off the data-noise line.
        ws = (1.0, 4.0, 6.0, 8.0)
        pg = torch.Generator(device="cpu").manual_seed(1234)
        zs, tss, yss = [], [], []
        with torch.no_grad():
            for wv in ws:
                for i0 in range(0, args.traj_pool, 256):
                    nb = min(256, args.traj_pool - i0)
                    zb = torch.randn(nb, *latents.shape[1:], generator=pg).to(dev)
                    yb = torch.randint(0, cfg["n_classes"], (nb,),
                                       generator=pg).to(dev)
                    yn = torch.full((nb,), m_ncls, device=dev)
                    sched = uniform_schedule(8) + [0.0]
                    for t_, tn_ in zip(sched[:-1], sched[1:]):
                        tb = torch.full((nb,), t_, device=dev)
                        zs.append(zb.clone())
                        tss.append(tb.clone())
                        yss.append(yb.clone())
                        if wv == 1.0:
                            vt = teacher(zb, tb, yb)
                        else:
                            vn = teacher(zb, tb, yn)
                            vt = vn + wv * (teacher(zb, tb, yb) - vn)
                        zb = zb - (t_ - tn_) * vt
        pool = (torch.cat(zs), torch.cat(tss), torch.cat(yss))
        print(f"trajectory pool: {pool[0].shape[0]} states "
              f"({args.traj_pool} seeds x {len(ws)} w x 8 steps)", flush=True)

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr0, weight_decay=0.0)
    bs = cfg["batch_size"]

    g = torch.Generator(device="cpu").manual_seed(42)  # same fixtures as fp
    val_eps = torch.randn(len(val), *latents.shape[1:], generator=g).to(dev)
    val_t = torch.sigmoid(torch.randn(len(val), generator=g)).to(dev)
    grid_noise = torch.randn(64, *latents.shape[1:], generator=g).to(dev)
    y_grid = ((torch.arange(64, device=dev) * (cfg["n_classes"] + 1)) // 64
              if y_all is not None else None)

    @torch.no_grad()
    def evaluate(step):
        eval_dit.load_state_dict(ema.state_dict())
        vloss, dloss, n = 0.0, 0.0, 0
        for i in range(0, len(val), 512):
            x0 = val[i:i + 512].float()
            e, t = val_eps[i:i + 512], val_t[i:i + 512]
            zt = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * e
            yv = y_val[i:i + 512] if y_val is not None else None
            v = eval_qat(zt, t, yv)
            vloss += F.mse_loss(v, e - x0, reduction="sum").item()
            if teacher is not None:
                dloss += F.mse_loss(v, teacher(zt, t, yv),
                                    reduction="sum").item()
            n += x0.numel()
        vloss /= n
        dloss /= n
        for k_steps in (4, 64):
            zs = euler_sample(eval_qat, grid_noise, uniform_schedule(k_steps),
                              y_grid)
            imgs = decode_latents(vae_decoder, zs, lat_mean, lat_std)
            save_grid_gray(imgs,
                           os.path.join(out_dir, f"grid_{step:06d}_k{k_steps}.png"))
        extra = f"  val distill(fq,fp) {dloss:.6f}" if teacher is not None else ""
        print(f"step {step:6d}  val v-loss(fq) {vloss:.5f}{extra}", flush=True)
        return vloss

    vloss = evaluate(0)  # PTQ baseline: the gap QAT is here to close

    # no autocast: 127-level rounding is too coarse-grained for bf16 math
    t0 = time.time()
    run_loss = 0.0
    for step in range(1, steps + 1):
        if step <= warmup:
            lr = lr0 * step / warmup
        elif args.cosine:
            prog = (step - warmup) / max(1, steps - warmup)
            lr = lr0 * max(0.1, 0.5 * (1.0 + math.cos(math.pi * prog)))
        else:
            lr = lr0
        for gp in opt.param_groups:
            gp["lr"] = lr
        if pool is not None:  # 50% data-noise interpolants, 50% trajectory
            nd = bs // 2
            idx = torch.randint(0, len(train), (nd,), device=dev)
            x0 = train[idx].float()
            eps = torch.randn_like(x0)
            td = torch.sigmoid(torch.randn(nd, device=dev))
            zt = (1 - td[:, None, None, None]) * x0 + td[:, None, None, None] * eps
            pi = torch.randint(0, pool[0].shape[0], (bs - nd,), device=dev)
            zt = torch.cat([zt, pool[0][pi]])
            t = torch.cat([td, pool[1][pi]])
            y_b = (torch.cat([y_train[idx], pool[2][pi]])
                   if y_train is not None else None)
        else:
            idx = torch.randint(0, len(train), (bs,), device=dev)
            x0 = train[idx].float()
            eps = torch.randn_like(x0)
            t = torch.sigmoid(torch.randn(bs, device=dev))
            zt = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * eps
            y_b = y_train[idx] if y_train is not None else None
        if teacher is not None:
            # dropout applied HERE so teacher and student see the same labels
            if y_b is not None and class_dropout > 0:
                dm = torch.rand(y_b.shape, device=dev) < class_dropout
                y_b = torch.where(dm, torch.full_like(y_b, m_ncls), y_b)
            with torch.no_grad():
                target = teacher(zt, t, y_b)
        else:
            target = eps - x0
        v = qat(zt, t, y_b)
        loss = F.mse_loss(v, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        ema.update(model)
        run_loss += loss.item()

        if step % 500 == 0:
            ips = step * bs / (time.time() - t0)
            print(f"step {step:6d}  loss {run_loss/500:.5f}  {ips:.0f} img/s",
                  flush=True)
            run_loss = 0.0
        if step % 2500 == 0 or step == steps:
            vloss = evaluate(step)
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(),
                        "cfg": {**cfg, "qat": True}, "step": step,
                        "val_loss": vloss},
                       os.path.join(out_dir, "ckpt_last.pt"))

    torch.save({"model": model.state_dict(), "ema": ema.state_dict(),
                "cfg": {**cfg, "qat": True}, "step": steps, "val_loss": vloss},
               os.path.join(out_dir, "ckpt_final.pt"))
    print(f"QAT-DONE in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
