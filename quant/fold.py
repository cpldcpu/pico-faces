"""Fold a trained DiT + VAE-decoder checkpoint into the exact-integer model
dict (md) consumed by int_sim / export / the C engine.

All the offline algebra lives here:
- per-out-channel symmetric int8 weight quantization
- biases converted to accumulator units (real / (s_in * s_w[o]))
- requant multipliers M/s from real factors (M ~ f * 2^s, |M| in [2^29, 2^30))
- adaLN folded per step: (1+scale)->RMSNorm gains (Q8), shift->bias (Q7),
  gate->branch-final requant multipliers (can be negative)
- Euler -dt[k] and the z Q12 fixed point folded into the final linear's M_v
- timestep tower evaluated at the K schedule points and discarded
- VAE: latent de-normalization folded into conv0, BN folded into all convs,
  final conv emits uint8 pixels (the -0.5 half-level folded into its bias)

Run:  python quant/fold.py --model m1_gray   # paths from models/<m>/export.yaml
Writes artifacts/<m>/export/{md.pkl,model.bin} and the model's e2e goldens.
"""
import argparse
import os
import pickle
import sys

import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import rfpaths
from quant.int_ops import RMS_B_Q, RMS_G_Q
from train.common.sincos import timestep_embedding
from train.dit.model import build_model
from train.vae.model import build_vae

Z_Q = 4096.0  # z state is Q12: int = real * 4096 (CLT-12 noise is natively this)


def quant_w(W):
    """per-out-channel symmetric int8. W: [O, ...] -> (i8 W, f32 s_w[O])"""
    W = W.astype(np.float64)
    flat = W.reshape(W.shape[0], -1)
    s = np.abs(flat).max(axis=1) / 127.0
    s = np.where(s == 0, 1e-12, s)
    q = np.clip(np.round(flat / s[:, None]), -127, 127).astype(np.int8)
    return q.reshape(W.shape), s


def to_Ms(f):
    """real factor(s) -> (M i32, s u8) with M ~ f * 2^s."""
    f = np.atleast_1d(np.asarray(f, dtype=np.float64))
    a = np.abs(f)
    s = np.where(a > 0, 30 - np.floor(np.log2(np.maximum(a, 1e-300))), 31)
    s = np.clip(s, 1, 62).astype(np.uint8)
    M = np.round(f * 2.0 ** s.astype(np.float64))
    M = np.clip(M, -(2**31 - 1), 2**31 - 1).astype(np.int32)
    return M, s


def renorm_Ms(M, s):
    """Collapse a per-channel (M, s) requant pair to ONE shift (v8): align
    every mantissa to the entry's smallest shift. The per-channel shift
    array is ~information-free (to_Ms pushes all magnitude into it), so this
    deletes it from the blob: u8[d] -> u8 scalar per step-table entry.
    Precision: a channel loses mantissa bits only in proportion to how far
    its magnitude sits below the entry max (M' ~ 2^30 / 2^(s[c]-s0)); by the
    time that matters the channel's contribution is below the output LSB."""
    M = np.asarray(M, np.int64)
    s = np.asarray(s, np.int64)
    s0 = int(s.min())
    d = s - s0
    Mr = np.where(d > 0, (M + (np.int64(1) << np.maximum(d - 1, 0))) >> d, M)
    return Mr.astype(np.int32), np.uint8(s0)


def bias_acc(b_real, s_in, s_w):
    return np.clip(np.round(b_real / (s_in * s_w)), -(2**31 - 1), 2**31 - 1
                   ).astype(np.int32)


def rms_gain(gain_real, s_out):
    return np.clip(np.round(gain_real / s_out * (1 << RMS_G_Q)), -32767, 32767
                   ).astype(np.int16)


def rms_bias(bias_real, s_out):
    return np.clip(np.round(bias_real / s_out * (1 << RMS_B_Q)), -32767, 32767
                   ).astype(np.int16)


def act_fn(name, x):
    if name == "gelu":
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    if name == "relu2":
        return np.maximum(x, 0) ** 2
    return np.maximum(x, 0)


def fold(model_name, ckpt_path, calib_path, out_dir):
    dev = "cpu"
    ckpt = torch.load(rfpaths.resolve(ckpt_path), map_location=dev,
                      weights_only=False)
    cfg = ckpt["cfg"]
    assert cfg["arch"] == "dit"
    model = build_model(cfg)
    model.load_state_dict(
        {k.replace("_orig_mod.", ""): v for k, v in ckpt["ema"].items()})
    model.eval()
    sd = {k: v.detach().numpy().astype(np.float64)
          for k, v in model.state_dict().items()}
    calib = {k: (v.astype(np.float64) if np.ndim(v) else float(v))
             for k, v in np.load(rfpaths.resolve(calib_path)).items()}

    exp_cfg = rfpaths.cfg(model_name, "export")
    sched = exp_cfg["schedule"]
    K = len(sched)
    dts = [t - tn for t, tn in zip(sched, list(sched[1:]) + [0.0])]

    d = cfg["dit"]["dim"]
    depth = cfg["dit"]["depth"]
    heads = cfg["dit"]["heads"]
    hd = d // heads
    act = cfg["act"]

    def S(name):  # scale = percentile / 127
        return calib[name] / 127.0

    # int16 residual scale. Legacy default is p99.9/127 — an int8-style
    # convention that leaves the residual RMS on ~8 of 32767 steps.
    # res_fine_bits (export.yaml) shifts it 2^n finer; the RMSNorm readers
    # are scale-invariant so only the writers (POS/M_emb/Mproj/Mfc2) move.
    # Empirical residual peak is ~7.3x p99.9 (m3_long, 320-run sweep), so
    # n=3 keeps ~4.4x clip margin. n=0 reproduces legacy folds byte-exact.
    s_res = S("res") / (1 << int(exp_cfg.get("res_fine_bits", 0)))
    s_xin = S("z")

    # ---- evaluate the conditioning tower at the K schedule points, then
    # drop it. Conditional models: one table set per class via
    # c = t_mlp(t) + y_emb[y]; index n_classes = null class (unconditional).
    ncls = int(cfg.get("n_classes", 0))
    n_cond = ncls + 1 if ncls else 1
    with torch.no_grad():
        t_pts = torch.tensor(sched, dtype=torch.float32)
        c = model.t_mlp(timestep_embedding(t_pts, model.t_dim))  # [K, d]
        cs = ([c + model.y_emb.weight[y][None] for y in range(n_cond)]
              if ncls else [c])
        mods = [[model.blocks[b].mod(cy).numpy().astype(np.float64)
                 for b in range(depth)] for cy in cs]           # [n_cond][depth][K,6d]
        fmod = np.stack([model.final_mod(cy).numpy().astype(np.float64)
                         for cy in cs])                          # [n_cond, K, 2d]

    # v7 precision pack (relu2 models with a per-channel calib): raw-acc relu2
    # square-requant, per-head softmax scales, and a 512-entry exp LUT (1/64
    # exponent grid, halving the softmax idx quantization -- budget 0.40->0.25).
    v7 = act == "relu2" and "fc2_in.0__pc" in calib
    lut_n, lut_scale = (512, 64.0) if v7 else (256, 32.0)
    md = {
        "meta": {"n_steps": K, "dim": d, "depth": depth, "heads": heads,
                 "tokens": model.g * model.g, "zch": model.z_ch,
                 "zhw": model.z_hw, "patch": model.patch, "n_cond": n_cond},
        "EXPLUT": np.round(32767 * np.exp(-np.arange(lut_n) / lut_scale)
                           ).astype(np.uint16),
        "POS": np.clip(np.round(sd["pos"][0] / s_res), -32767, 32767).astype(np.int16),
    }

    Mz, sz = to_Ms(1.0 / (Z_Q * s_xin))
    md["M_zin"] = np.repeat(Mz, K).astype(np.int32)
    md["s_zin"] = np.repeat(sz, K).astype(np.uint8)

    We, swe = quant_w(sd["embed.weight"])
    md["W_emb"] = We
    md["b_emb"] = bias_acc(sd["embed.bias"], s_xin, swe)
    md["M_emb"], md["s_emb"] = to_Ms(s_xin * swe / s_res)

    def S_pc(name):
        """Per-channel activation scales when the calib collected them
        (att_in/fc1_in/final_in). These fold for FREE: the norm gains/biases
        are already per-channel (s_out becomes a vector) and the consumer
        weight's input columns absorb s_out[c] before quant_w -- identical
        blob format, zero engine cost. Error budget: att_in 0.93->0.11,
        fc1_in 0.76->0.13 %/step. Dead channels floored at s_tensor/64."""
        key = name + "__pc"
        if key not in calib:
            return None
        v = np.asarray(calib[key], np.float64) / 127.0
        return np.maximum(v, float(calib[name]) / 127.0 / 64.0)

    def headroom(s_cal, gains, biases=None):
        """Inflate a norm's output scale so folded Q8 gains / Q7 biases never
        clamp int16 (downstream requant multipliers absorb the inflation).
        Limits with margin: |G| <= 120 * s_out, |B| <= 240 * s_out.
        s_cal may be per-channel [d]: limits then apply per channel."""
        if np.ndim(s_cal):
            g = np.abs(gains).reshape(-1, len(s_cal)).max(axis=0)
            s = np.maximum(s_cal, g / 120.0)
            if biases is not None:
                bmax = np.abs(biases).reshape(-1, len(s_cal)).max(axis=0)
                s = np.maximum(s, bmax / 240.0)
            return s
        s = max(s_cal, float(np.abs(gains).max()) / 120.0)
        if biases is not None:
            s = max(s, float(np.abs(biases).max()) / 240.0)
        return s

    blocks = []
    step_tab = [[[None] * depth for _ in range(K)] for _ in range(n_cond)]
    for b in range(depth):
        p = f"blocks.{b}."
        # headroom must hold for every (class, step) table
        m_all = np.concatenate([mods[y][b] for y in range(n_cond)]
                               ).reshape(n_cond * K, 6, d)  # s1,b1,g1,s2,b2,g2
        s_att_pc, s_fc1_pc = S_pc(f"att_in.{b}"), S_pc(f"fc1_in.{b}")
        att_pc, fc1_pc = s_att_pc is not None, s_fc1_pc is not None
        s_att_in = headroom(s_att_pc if att_pc else S(f"att_in.{b}"),
                            1 + m_all[:, 0], m_all[:, 1])
        s_fc1_in = headroom(s_fc1_pc if fc1_pc else S(f"fc1_in.{b}"),
                            1 + m_all[:, 3], m_all[:, 4])
        s_qk_pre = S(f"qk_pre.{b}")
        gnorm = np.concatenate([sd[p + "attn.q_norm.weight"],
                                sd[p + "attn.k_norm.weight"]])
        s_qk_pc = S_pc(f"qk_post.{b}")  # per-HEAD scales (v7 calib)
        if v7 and s_qk_pc is not None:
            # per-head qk_post grids: fold into the per-head Gq/Gk rows (free)
            # and per-head M_sm (scores carry s_qk[h]^2). Budget 0.83 -> ~0.3.
            s_qk = np.maximum(s_qk_pc, np.abs(gnorm).max() / 120.0)  # [heads]
            s_qk_vec = np.repeat(s_qk, hd)  # [d], aligns with tiled gains
        else:
            s_qk = headroom(S(f"qk_post.{b}"), gnorm)
            s_qk_vec = s_qk
        s_v = S(f"v.{b}")
        s_att_out = S(f"att_out.{b}")
        s_act_in = S(f"act_in.{b}")
        # relu2 is degree-2 homogeneous (act(s*x) = s^2*act(x)): the activation
        # is computed by SQUARING the raw fc1 int32 accumulator directly --
        # no intermediate requant, no LUT, no input clip (act_in error -> 0
        # exactly) -- and the per-channel output scale lives in the requant
        # multiplier M_actq[c] (kills the per-tensor fc2_in grid; budget
        # 1.40 -> 0.36 %/step). u = relu(acc)^2 >> 12 keeps u*M inside int64
        # (acc <= K*127^2 ~ 2^21, u <= 2^30); the 2^12 folds into M_actq.
        # Mfc1/sfc1 become unused and are dropped from the blob (v7).
        s_fc2_pc = S_pc(f"fc2_in.{b}")
        act_sq = s_fc2_pc is not None and act == "relu2"
        s_fc2_in = s_fc2_pc if act_sq else S(f"fc2_in.{b}")

        # per-channel att_in: fold s_att_in[c] into the qkv input columns; the
        # requant then sees unit input scale (s_in drops out of M and bias)
        Wq_src = sd[p + "attn.qkv.weight"]  # [3d, d]
        if att_pc:
            Wq_src = Wq_src * s_att_in[None, :]
        Wq, swq = quant_w(Wq_src)
        out_scales = np.concatenate([np.full(2 * d, s_qk_pre), np.full(d, s_v)])
        Mq, sq = to_Ms((1.0 if att_pc else s_att_in) * swq / out_scales)
        qg = np.tile(sd[p + "attn.q_norm.weight"], heads)
        kg = np.tile(sd[p + "attn.k_norm.weight"], heads)
        Wp, swp = quant_w(sd[p + "attn.proj.weight"])
        Wf1_src = sd[p + "mlp.0.weight"]
        if fc1_pc:  # per-channel fc1_in folded into the fc1 input columns
            Wf1_src = Wf1_src * s_fc1_in[None, :]
        Wf1, swf1 = quant_w(Wf1_src)
        Wf2_src = sd[p + "mlp.2.weight"]
        if act_sq:  # per-channel h1 scales folded into the fc2 input columns
            Wf2_src = Wf2_src * s_fc2_in[None, :]
        Wf2, swf2 = quant_w(Wf2_src)
        M_sm, s_sm = to_Ms(np.broadcast_to(np.atleast_1d(s_qk * s_qk),
                                           (heads,)) * lut_scale / np.sqrt(hd)
                           if v7 else s_qk * s_qk * lut_scale / np.sqrt(hd))
        M_att, s_att = to_Ms(np.full(d, (s_v / 127.0) / s_att_out))
        if act_sq:
            # real value per acc count of fc1 output channel c
            s_sq_in = (1.0 if fc1_pc else s_fc1_in) * swf1
            M_actq, s_actq = to_Ms(4096.0 * s_sq_in * s_sq_in / s_fc2_in)
        else:
            Mf1, sf1 = to_Ms((1.0 if fc1_pc else s_fc1_in) * swf1 / s_act_in)
            xs = (np.arange(256) - 128) * s_act_in
            lut = np.clip(np.round(act_fn(act, xs) / s_fc2_in),
                          -127, 127).astype(np.int8)

        blocks.append({
            "Wqkv": Wq, "bqkv": bias_acc(sd[p + "attn.qkv.bias"],
                                         1.0 if att_pc else s_att_in, swq),
            "Mqkv": Mq, "sqkv": sq,
            "Gq": rms_gain(qg, s_qk_vec).reshape(heads, hd),
            "Gk": rms_gain(kg, s_qk_vec).reshape(heads, hd),
            "M_sm": (M_sm.astype(np.int32) if v7 else np.int32(M_sm[0])),
            "s_sm": (s_sm.astype(np.uint8) if v7 else np.uint8(s_sm[0])),
            "M_att": M_att, "s_att": s_att,
            "Wproj": Wp, "bproj": bias_acc(sd[p + "attn.proj.bias"], s_att_out, swp),
            "Wfc1": Wf1, "bfc1": bias_acc(sd[p + "mlp.0.bias"],
                                          1.0 if fc1_pc else s_fc1_in, swf1),
            **({"M_actq": M_actq, "s_actq": s_actq} if act_sq
               else {"Mfc1": Mf1, "sfc1": sf1, "ACT_LUT": lut}),
            "Wfc2": Wf2, "bfc2": bias_acc(sd[p + "mlp.2.bias"],
                                          1.0 if act_sq else s_fc2_in, swf2),
        })

        for y in range(n_cond):
            for k in range(K):
                m6 = mods[y][b][k].reshape(6, d)  # order: s1,b1,g1,s2,b2,g2
                s1, b1, g1, s2, b2, g2 = m6
                Mp_, sp_ = to_Ms(s_att_out * swp * g1 / s_res)
                Mf2_, sf2_ = to_Ms((1.0 if act_sq else s_fc2_in)
                                   * swf2 * g2 / s_res)
                if v7:  # v8 blob: scalar shift per entry, mantissas aligned
                    Mp_, sp_ = renorm_Ms(Mp_, sp_)
                    Mf2_, sf2_ = renorm_Ms(Mf2_, sf2_)
                step_tab[y][k][b] = {
                    "G1": rms_gain(1 + s1, s_att_in), "B1": rms_bias(b1, s_att_in),
                    "G2": rms_gain(1 + s2, s_fc1_in), "B2": rms_bias(b2, s_fc1_in),
                    "Mproj": Mp_, "sproj": sp_, "Mfc2": Mf2_, "sfc2": sf2_,
                }
    md["blocks"] = blocks
    md["step_tab"] = step_tab
    md["meta"]["act_sq"] = int("M_actq" in blocks[0])  # relu2 square-requant (v7)

    s_fin_pc = S_pc("final_in")
    fin_pc = s_fin_pc is not None
    s_final_in = headroom(s_fin_pc if fin_pc else S("final_in"),
                          1 + fmod[..., :d], fmod[..., d:])
    Wfin_src = sd["final.weight"]
    if fin_pc:  # per-channel final_in folded into W_final's input columns
        Wfin_src = Wfin_src * s_final_in[None, :]
    Wfin, swfin = quant_w(Wfin_src)
    s_fin_eff = 1.0 if fin_pc else s_final_in
    md["W_final"] = Wfin
    md["b_final"] = bias_acc(sd["final.bias"], s_fin_eff, swfin)
    md["Gf"] = [[rms_gain(1 + fmod[y, k, :d], s_final_in) for k in range(K)]
                for y in range(n_cond)]
    md["Bf"] = [[rms_bias(fmod[y, k, d:], s_final_in) for k in range(K)]
                for y in range(n_cond)]
    # dt fold is class-independent (s_final_in is shared across classes)
    md["M_v"], md["s_v"] = [], []
    for k in range(K):
        Mv, sv = to_Ms(-dts[k] * s_fin_eff * swfin * Z_Q)
        md["M_v"].append(Mv)
        md["s_v"].append(sv)

    # ---- classifier-free guidance tables (export.yaml: cfg_w list) ---------
    # Guided update z <- z - dt*(w*v_cond + (1-w)*v_null) runs the body twice
    # per step; the blend folds entirely into per-pass rescales of M_v
    # (cond pass: w * -dt, null pass: (1-w) * -dt, positive since w > 1).
    # Requires a calibration that covers guided trajectories (union calib).
    cfg_w = [float(w) for w in exp_cfg.get("cfg_w", [])]
    md["meta"]["cfg_w"] = cfg_w
    # dropped-attention blocks (their attn branch is omitted from the blob and
    # skipped by int_sim/engine; the block still runs its MLP). See the ablation
    # study -- e.g. block 10's attention was mildly harmful, so dropping it both
    # shrinks the blob and slightly improves FID.
    md["drop_attn"] = sorted(int(b) for b in exp_cfg.get("drop_attn", []))
    # v8 drops the per-pass M_v_c/M_v_n tables: dead weight since the int32-
    # difference blend (the guided path requants ONCE with the base M_v).
    # Legacy (pre-v8) folds keep them so frozen builds refold byte-exact.
    if not v7:
        md["M_v_c"], md["s_v_c"], md["M_v_n"], md["s_v_n"] = [], [], [], []
        for wv in cfg_w:
            Mc, Mn = [], []
            for k in range(K):
                Mvc, svc = to_Ms(-dts[k] * wv * s_fin_eff * swfin * Z_Q)
                Mvn, svn = to_Ms(-dts[k] * (1.0 - wv) * s_fin_eff * swfin * Z_Q)
                Mc.append((Mvc, svc))
                Mn.append((Mvn, svn))
            md["M_v_c"].append([x[0] for x in Mc])
            md["s_v_c"].append([x[1] for x in Mc])
            md["M_v_n"].append([x[0] for x in Mn])
            md["s_v_n"].append([x[1] for x in Mn])

    # ---- VAE decoder --------------------------------------------------------
    vcfg = rfpaths.cfg(model_name, "vae")
    vck = torch.load(rfpaths.resolve(cfg.get("vae_ckpt") or exp_cfg.get("vae_checkpoint")
                                     or os.path.join(vcfg["out_dir"], "vae_final.pt")),
                     map_location=dev, weights_only=False)
    vae = build_vae(vcfg)
    vae.load_state_dict(vck["model"])
    dec = vae.decoder
    vsd = {k: v.detach().numpy().astype(np.float64)
           for k, v in dec.state_dict().items()}
    lat = np.load(os.path.join(ROOT, vcfg["out_dir"], "latents.npz"))
    lmean, lstd = lat["mean"].astype(np.float64), lat["std"].astype(np.float64)

    s_zdec_in = S("z_dec_in")
    Mzd, szd = to_Ms(1.0 / (Z_Q * s_zdec_in))
    md["M_zdec"], md["s_zdec"] = np.int32(Mzd[0]), np.uint8(szd[0])

    layers = []
    s_in = s_zdec_in
    n_body = len(dec.body)
    for i in range(n_body + 1):
        if i < n_body:
            W = vsd[f"body.{i}.0.weight"]      # [O, C, 3, 3]
            bconv = vsd[f"body.{i}.0.bias"]
            gam, bet = vsd[f"body.{i}.1.weight"], vsd[f"body.{i}.1.bias"]
            mu, var = vsd[f"body.{i}.1.running_mean"], vsd[f"body.{i}.1.running_var"]
            f = gam / np.sqrt(var + 1e-5)
            W = W * f[:, None, None, None]
            b_real = (bconv - mu) * f + bet
        else:
            W = vsd["out.weight"]
            b_real = vsd["out.bias"]
        if i == 0:  # latent de-normalization folded into conv0
            W = W * lstd[None, :, None, None]
            b_real = b_real + (W / lstd[None, :, None, None] *
                               lmean[None, :, None, None]).sum(axis=(1, 2, 3))
        Wq_, sw_ = quant_w(W)
        if i < n_body:
            s_out = S(f"dec.{i}")
            M_, s_ = to_Ms(s_in * sw_ / s_out)
            b_ = bias_acc(b_real, s_in, sw_)
            layers.append({"W": np.transpose(Wq_, (0, 2, 3, 1)).copy(),
                           "b": b_, "M": M_, "s": s_,
                           "up": 1 if i in dec.up_before else 0, "relu": 1, "u8": 0})
            s_in = s_out
        else:
            # uint8 pixels: rq ~ round(v*127.5 - 0.5), then +128 in the op
            M_, s_ = to_Ms(s_in * sw_ * 127.5)
            b_ = bias_acc(b_real - 0.5 / 127.5, s_in, sw_)
            layers.append({"W": np.transpose(Wq_, (0, 2, 3, 1)).copy(),
                           "b": b_, "M": M_, "s": s_, "up": 0, "relu": 0, "u8": 1})
    md["dec"] = layers
    md["meta"]["img_ch"] = int(layers[-1]["W"].shape[0])


    os.makedirs(rfpaths.resolve(out_dir), exist_ok=True)
    with open(os.path.join(rfpaths.resolve(out_dir), "md.pkl"), "wb") as f:
        pickle.dump(md, f)
    print(f"folded model -> {out_dir}/md.pkl")
    return md


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--calib", default=None)
    args = ap.parse_args()
    exp = rfpaths.cfg(args.model, "export")
    out_dir = os.path.dirname(exp["out"])
    md = fold(args.model, args.ckpt or exp["checkpoint"],
              args.calib or exp["calib"], out_dir)

    from quant.export import write_goldens, write_model_bin, write_rf_cfg
    write_model_bin(md, rfpaths.resolve(exp["out"]))
    write_rf_cfg(md, os.path.join(rfpaths.resolve(out_dir), "rf_cfg.h"))
    # CFG models get a 4th golden: seeds 1..3 cycle the guided table sets,
    # seed 4 gates the plain pass (w_idx -1) in the same build
    n_gold = 4 if md["meta"].get("cfg_w") else 3
    write_goldens(md, exp["golden_seeds"][:n_gold],
                  rfpaths.resolve(exp["goldens"]))
