"""Collect activation ranges (99.9th percentile) from real K-step sampling
trajectories through the fp model + decoder. Every quant point in the deployed
graph gets a scale here.

Run from repo root (WSL):
  python quant/calibrate.py --model m1_gray   # paths from models/<m>/export.yaml
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import rfpaths
from train.dit.model import build_model
from train.dit.train_dit import load_vae_decoder

PCT = 99.9


class Collector:
    def __init__(self):
        self.pcts = defaultdict(list)
        self.pc = defaultdict(list)  # per-channel (last dim) percentiles

    def tap(self, name, pc=False, pc_axis=None):
        def hook(t):
            with torch.no_grad():
                a = t.detach().abs().float()
                self.pcts[name].append(
                    torch.quantile(a.flatten()[::7], PCT / 100).item())
                if pc:  # per-channel scales: fold free into the per-channel
                    #     norm gains + following weight columns (see fold.py).
                    #     pc_axis selects the channel axis (default: last) --
                    #     e.g. qk_post uses the HEAD axis for per-head scales.
                    if pc_axis is None:
                        flat = a.reshape(-1, a.shape[-1])
                    else:
                        flat = a.movedim(pc_axis, -1).reshape(
                            -1, a.shape[pc_axis])
                    self.pc[name].append(
                        torch.quantile(flat, PCT / 100, dim=0).cpu().numpy())
        return hook

    def result(self):
        # median of per-batch percentiles: robust to outlier batches
        res = {k: float(np.median(v)) for k, v in self.pcts.items()}
        for k, v in self.pc.items():
            res[k + "__pc"] = np.median(np.stack(v), axis=0)
        return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--trajectories", type=int, default=512)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--guidance-w", type=float, default=None,
                    help="sample with classifier-free guidance at this w "
                         "(two model passes per step; both are tapped, so "
                         "the ranges cover cond AND null activations)")
    ap.add_argument("--union", default=None,
                    help="existing calib .npz to take the per-key max with "
                         "(e.g. the frozen QAT base calib)")
    ap.add_argument("--hires-ckpt", default=None,
                    help="hires head checkpoint (default: export.yaml "
                         "hires_ckpt); adds the head.0 activation tap")
    args = ap.parse_args()

    dev = "cuda"
    exp = rfpaths.cfg(args.model, "export")
    ckpt_path = rfpaths.resolve(args.ckpt or exp["checkpoint"])
    out_path = rfpaths.resolve(args.out or exp["calib"])
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    cfg = ckpt["cfg"]
    assert cfg["arch"] == "dit", "calibration currently implements the DiT path"
    model = build_model(cfg).to(dev).eval()
    model.load_state_dict(
        {k.replace("_orig_mod.", ""): v for k, v in ckpt["ema"].items()})

    vae_decoder = load_vae_decoder(dev, args.model,
                                   cfg.get("vae_ckpt") or exp.get("vae_checkpoint"))
    z = np.load(rfpaths.resolve(cfg["latents"]))
    lat_mean = torch.from_numpy(z["mean"]).to(dev)
    lat_std = torch.from_numpy(z["std"]).to(dev)

    col = Collector()
    hooks = []

    def pre(mod, name, pc=False):
        hooks.append(mod.register_forward_pre_hook(
            lambda m, inp: col.tap(name, pc)(inp[0])))

    def post(mod, name, pc=False, pc_axis=None):
        hooks.append(mod.register_forward_hook(
            lambda m, inp, out: col.tap(name, pc, pc_axis)(out)))

    post(model.embed, "res")  # embed contribution to residual
    for b, blk in enumerate(model.blocks):
        post(blk, "res")                          # residual stream magnitude
        pre(blk.attn.qkv, f"att_in.{b}", pc=True)  # norm1 output
        def qkv_hook(b_):
            def hook(m, inp, out):  # returns None: must not replace output
                third = out.shape[-1] // 3
                col.tap(f"qk_pre.{b_}")(out[..., : 2 * third])
                col.tap(f"v.{b_}")(out[..., 2 * third:])
            return hook
        hooks.append(blk.attn.qkv.register_forward_hook(qkv_hook(b)))
        post(blk.attn.q_norm, f"qk_post.{b}", pc=True, pc_axis=1)  # per-head
        post(blk.attn.k_norm, f"qk_post.{b}", pc=True, pc_axis=1)
        pre(blk.attn.proj, f"att_out.{b}")
        pre(blk.mlp[0], f"fc1_in.{b}", pc=True)
        post(blk.mlp[0], f"act_in.{b}")
        post(blk.mlp[1], f"fc2_in.{b}", pc=True)
    pre(model.final, "final_in", pc=True)

    for i, layer in enumerate(vae_decoder.body):
        post(layer, f"dec.{i}")

    head = None
    hires_ckpt = args.hires_ckpt or exp.get("hires_ckpt")
    if hires_ckpt:
        from train.vae.hires_head import HiresHead, trunk
        hk = torch.load(rfpaths.resolve(hires_ckpt), map_location=dev,
                        weights_only=False)
        vcfg = rfpaths.cfg(args.model, "vae")
        head = HiresHead(c_in=vcfg["dec_plan"][-1][0],
                         c_mid=hk["c_mid"]).to(dev).eval()
        head.load_state_dict(hk["head"])
        post(head.block, "head.0")

    sched = exp["schedule"]
    ts = list(sched) + [0.0]
    g = torch.Generator(device="cpu").manual_seed(777)

    n_batches = args.trajectories // args.batch
    with torch.no_grad():
        for _ in range(n_batches):
            zc, zhw = cfg.get("latent_ch", 4), cfg.get("latent_hw", 16)
            zt = torch.randn(args.batch, zc, zhw, zhw, generator=g).to(dev)
            ncls = cfg.get("n_classes", 0)
            y = (torch.randint(0, ncls, (args.batch,), generator=g).to(dev)
                 if ncls else None)  # mixed classes: ranges must cover all
            for t, t_next in zip(ts[:-1], ts[1:]):
                col.tap("z")(zt)
                tb = torch.full((args.batch,), t, device=dev)
                if args.guidance_w is not None and y is not None:
                    v_c = model(zt, tb, y)
                    v_n = model(zt, tb, torch.full_like(y, ncls))
                    v = v_n + args.guidance_w * (v_c - v_n)
                else:
                    v = model(zt, tb, y) if y is not None else model(zt, tb)
                zt = zt - (t - t_next) * v
            col.tap("z")(zt)
            col.tap("z_dec_in")(zt)
            zr = zt * lat_std[None, :, None, None] + lat_mean[None, :, None, None]
            if head is not None:
                feat, out128 = trunk(vae_decoder, zr)
                head(feat, out128)  # dec.i hooks fire via trunk; adds head.0
            else:
                vae_decoder(zr)

    for h in hooks:
        h.remove()
    res = col.result()
    if args.union:
        base = dict(np.load(rfpaths.resolve(args.union)).items())
        # base may lack keys new to this run (e.g. head.0); never vice versa
        missing = set(base) - set(res)
        assert not missing, f"union base has keys this run lacks: {missing}"
        res = {k: np.maximum(res[k], base[k]) if k in base else res[k]
               for k in res}
    for k in sorted(res):
        v = res[k]
        if np.ndim(v):  # per-channel array
            print(f"{k:14s} pc[{len(v)}] max={float(np.max(v)):.4f}")
        else:
            print(f"{k:14s} {float(v):.4f}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, **res)
    print("saved", out_path)


if __name__ == "__main__":
    main()
