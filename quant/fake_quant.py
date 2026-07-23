"""Fake-quantization (quantize-dequantize with straight-through gradients)
mirroring the deployed int8 graph, for QAT finetuning.

QATDiT wraps a trained DiT and re-runs its forward with qdq inserted at
exactly the boundaries the device quantizes (the same names calibrate.py
taps and fold.py folds):

  weights      per-out-channel symmetric int8, scale = max|row|/127
               (identical to fold.quant_w, recomputed live from the weights)
  activations  symmetric int8, scale FROZEN at calib/127 so training
               optimizes the deployed configuration. Per-CHANNEL wherever
               the calib carries a <name>__pc array (v7 precision pack:
               att_in/fc1_in/final_in per model channel, fc2_in per hidden
               channel, qk_post per head), with fold.S_pc's dead-channel
               floor s_tensor/64; per-tensor otherwise.
  softmax p    fixed scale 1/127 (device: int softmax output)

v7 relu2 square-requant (act_sq, gated exactly like fold.py: act == relu2
and a per-channel fc2_in calib): the device squares the RAW int32 fc1
accumulator -- there is NO int8 grid between fc1 and the activation -- so
the fake-quant applies relu2 in fp and quantizes only its OUTPUT onto the
per-channel fc2_in grid. Legacy LUT models keep the act_in input grid.

drop_attn: blocks whose attention branch the export omits (v6 attn_mask)
skip that branch here too, so QAT heals the surgery.

Deliberately NOT quantized (higher precision on device): the residual
stream (int16), bias adds (int32 accumulators), the Euler z update (int16
tokens), the conditioning tower (folded exactly into per-step tables with
headroom at export), and the VAE decoder (PTQ, not finetuned here).

The wrapper shares parameters with the inner model: optimizers, EMA and
checkpoints all see plain DiT tensors, so fold.py consumes the result
unchanged.
"""
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.common.sincos import timestep_embedding


def qdq_w(W):
    """per-out-channel symmetric int8 qdq, STE. Matches fold.quant_w."""
    s = W.detach().abs().flatten(1).amax(dim=1).clamp_min(1e-12) / 127.0
    s = s.view(-1, *([1] * (W.dim() - 1)))
    q = torch.clamp(torch.round(W / s), -127, 127) * s
    return W + (q - W).detach()


def qdq_a(x, s, dim=-1):
    """symmetric int8 qdq at fixed scale s, clamp-first STE. s is a 0-d
    tensor (per-tensor) or a 1-d tensor broadcast along `dim` (per-channel
    / per-head). The clamp must carry the gradient (zero outside the
    representable range): a leaky STE keeps pushing the ~0.1% of values
    beyond the 99.9-pct calib scale further out and the finetune diverges."""
    if s.dim() == 1:
        shape = [1] * x.dim()
        shape[dim] = -1
        s = s.view(shape)
    lim = 127.0 * s
    xc = torch.minimum(torch.maximum(x, -lim), lim)
    q = torch.round(xc / s) * s
    return xc + (q - xc).detach()


def prep_scales(scales, device):
    """calib dict -> {name: scale tensor}, value/127, mirroring fold.S_pc:
    a <name>__pc array yields a per-channel vector floored at the
    per-tensor scale / 64 (dead channels); otherwise a 0-d scalar."""
    out = {}
    for k, v in scales.items():
        if k.endswith("__pc"):
            continue
        s = float(np.asarray(v)) / 127.0
        pck = k + "__pc"
        if pck in scales:
            vec = np.asarray(scales[pck], np.float64) / 127.0
            vec = np.maximum(vec, s / 64.0)
            out[k] = torch.tensor(vec, dtype=torch.float32, device=device)
        else:
            out[k] = torch.tensor(s, dtype=torch.float32, device=device)
    return out


def headroom_scales(scales, model, schedule):
    """Return a copy of the calib dict with fold.headroom()'s inflation
    applied (per-TENSOR calibs only -- the m3_long_cfg headroom-QAT
    experiment showed QAT weights tolerate the deploy-scale inflation, so
    per-channel/v7 runs train on the raw calib grids instead)."""
    if any(k.endswith("__pc") for k in scales):
        raise NotImplementedError(
            "headroom mode is per-tensor only; v7 per-channel calibs train "
            "on the raw grids (deploy-scale tolerance shown empirically)")

    out = {k: float(v) for k, v in scales.items()}

    def infl(name, gains, biases=None):
        s = max(out[name] / 127.0, float(np.abs(gains).max()) / 120.0)
        if biases is not None:
            s = max(s, float(np.abs(biases).max()) / 240.0)
        out[name] = s * 127.0

    with torch.no_grad():
        dev = next(model.parameters()).device
        t_pts = torch.tensor(schedule, dtype=torch.float32, device=dev)
        c = model.t_mlp(timestep_embedding(t_pts, model.t_dim))
        n_cond = model.n_classes + 1 if model.n_classes else 1
        cs = ([c + model.y_emb.weight[y][None] for y in range(n_cond)]
              if model.n_classes else [c])
        for b, blk in enumerate(model.blocks):
            m_all = torch.cat([blk.mod(cy) for cy in cs]).cpu().numpy()
            d = m_all.shape[-1] // 6
            m_all = m_all.reshape(-1, 6, d)
            infl(f"att_in.{b}", 1 + m_all[:, 0], m_all[:, 1])
            infl(f"fc1_in.{b}", 1 + m_all[:, 3], m_all[:, 4])
            qk = torch.cat([blk.attn.q_norm.weight,
                            blk.attn.k_norm.weight]).cpu().numpy()
            infl(f"qk_post.{b}", qk)
        fmod = torch.cat([model.final_mod(cy) for cy in cs]).cpu().numpy()
        d = fmod.shape[-1] // 2
        infl("final_in", 1 + fmod[..., :d], fmod[..., d:])
    return out


class QATDiT(nn.Module):
    """Fake-quant view of a trained DiT.

    scales: calib dict (name -> percentile value, plus optional <name>__pc
    per-channel arrays). act: the model's MLP activation ("relu2" enables
    the v7 square-requant path when the calib is per-channel, mirroring
    fold.py's gate). drop_attn: block indices whose attention branch the
    export drops (v6 attn_mask)."""

    def __init__(self, model, scales, act="relu2", drop_attn=()):
        super().__init__()
        self.m = model
        dev = next(model.parameters()).device
        self.s = prep_scales(scales, dev)
        self.drop_attn = frozenset(drop_attn)
        # per-block v7 gate, exactly fold.py's: relu2 AND per-channel fc2_in
        self.act_sq = {
            b: act == "relu2" and self.s[f"fc2_in.{b}"].dim() == 1
            for b in range(len(model.blocks))}

    def forward(self, z, t, y=None):
        m, s = self.m, self.s
        B = z.shape[0]

        x_in = qdq_a(m.patchify(z), s["z"])
        x = F.linear(x_in, qdq_w(m.embed.weight), m.embed.bias) + m.pos

        c = m.t_mlp(timestep_embedding(t, m.t_dim))
        if m.n_classes:
            if y is None:
                y = torch.full((B,), m.n_classes, dtype=torch.long,
                               device=z.device)
            if self.training and m.class_dropout > 0:
                drop = torch.rand(y.shape, device=y.device) < m.class_dropout
                y = torch.where(drop, torch.full_like(y, m.n_classes), y)
            c = c + m.y_emb(y)

        for b, blk in enumerate(m.blocks):
            s1, b1, g1, s2, b2, g2 = blk.mod(c)[:, None].chunk(6, dim=-1)

            if b not in self.drop_attn:
                at = blk.attn
                H, hd = at.heads, at.hd
                xa = qdq_a(blk.norm1(x) * (1 + s1) + b1, s[f"att_in.{b}"])
                qkv = F.linear(xa, qdq_w(at.qkv.weight), at.qkv.bias)
                N = xa.shape[1]
                q, k, v = qkv.view(B, N, 3, H, hd).permute(2, 0, 3, 1, 4)
                q = qdq_a(q, s[f"qk_pre.{b}"])
                k = qdq_a(k, s[f"qk_pre.{b}"])
                v = qdq_a(v, s[f"v.{b}"])
                # qk_post: per-HEAD grid in the v7 calib (dim=1 of [B,H,N,hd])
                q = qdq_a(at.q_norm(q), s[f"qk_post.{b}"], dim=1)
                k = qdq_a(at.k_norm(k), s[f"qk_post.{b}"], dim=1)
                p = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(hd),
                                  dim=-1)
                p = qdq_a(p, torch.tensor(1.0 / 127.0, device=z.device))
                attn = (p @ v).transpose(1, 2).reshape(B, N, -1)
                attn = qdq_a(attn, s[f"att_out.{b}"])
                x = x + g1 * F.linear(attn, qdq_w(at.proj.weight),
                                      at.proj.bias)

            xm = qdq_a(blk.norm2(x) * (1 + s2) + b2, s[f"fc1_in.{b}"])
            h = F.linear(xm, qdq_w(blk.mlp[0].weight), blk.mlp[0].bias)
            if self.act_sq[b]:
                # v7: the device squares the raw int32 accumulator -- no int8
                # grid between fc1 and the activation. Only the activation
                # OUTPUT lands on the (per-channel) fc2_in grid.
                h = blk.mlp[1](h)
                h = qdq_a(h, s[f"fc2_in.{b}"])
            else:
                h = blk.mlp[1](qdq_a(h, s[f"act_in.{b}"]))
                h = qdq_a(h, s[f"fc2_in.{b}"])
            x = x + g2 * F.linear(h, qdq_w(blk.mlp[2].weight), blk.mlp[2].bias)

        sf, bf = m.final_mod(c)[:, None].chunk(2, dim=-1)
        xf = qdq_a(m.final_norm(x) * (1 + sf) + bf, s["final_in"])
        out = F.linear(xf, qdq_w(m.final.weight), m.final.bias)
        return m.unpatchify(out)
