"""Rectified-flow backbones for the M2 bake-off: DiT and a small U-Net.

Both are built from the same int8-deployable vocabulary:
- RMSNorm (no mean subtraction) with adaLN scale/shift folded in at export
- adaLN-zero gates (fold into requant multipliers at export)
- activation knob: gelu | relu2 | relu (all cheap in int8)
- qk-RMSNorm bounds attention scores for integer softmax
- U-Net uses strided-conv down / nearest-neighbor up (integer-exact)

Token/patch ordering contract (DiT), mirrored by the C engine:
tokens are row-major over the 8x8 patch grid; within a token the 16 features
are ordered (c, py, px) -> index c*4 + py*2 + px.
"""
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from train.common.sincos import sincos_2d, timestep_embedding


def make_act(name):
    if name == "gelu":
        return nn.GELU(approximate="tanh")
    if name == "relu2":
        return ReLU2()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(name)


class ReLU2(nn.Module):
    def forward(self, x):
        return F.relu(x).square()


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.hd = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.q_norm = nn.RMSNorm(self.hd)
        self.k_norm = nn.RMSNorm(self.hd)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):  # (B, N, dim)
        B, N, _ = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.heads, self.hd).permute(2, 0, 3, 1, 4)
        q, k = self.q_norm(q), self.k_norm(k)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(B, N, -1))


class DiTBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio, act):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, elementwise_affine=False)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.RMSNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim), make_act(act), nn.Linear(mlp_ratio * dim, dim)
        )
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.mod[1].weight)
        nn.init.zeros_(self.mod[1].bias)

    def forward(self, x, c):  # c: (B, dim)
        s1, b1, g1, s2, b2, g2 = self.mod(c)[:, None].chunk(6, dim=-1)
        x = x + g1 * self.attn(self.norm1(x) * (1 + s1) + b1)
        x = x + g2 * self.mlp(self.norm2(x) * (1 + s2) + b2)
        return x


class DiT(nn.Module):
    def __init__(self, dim=128, depth=8, heads=4, patch=2, mlp_ratio=4,
                 act="gelu", z_ch=4, z_hw=16, t_dim=256,
                 n_classes=0, class_dropout=0.0, pos_embed="sincos"):
        super().__init__()
        self.patch, self.z_ch, self.z_hw = patch, z_ch, z_hw
        self.n_classes, self.class_dropout = n_classes, class_dropout
        self.g = z_hw // patch  # patch grid (8)
        pdim = z_ch * patch * patch  # 16
        self.embed = nn.Linear(pdim, dim)
        if pos_embed == "learned":  # ViT-style; folds to the same POS table
            self.pos = nn.Parameter(torch.empty(1, self.g * self.g, dim))
            nn.init.trunc_normal_(self.pos, std=0.02)
        else:
            self.register_buffer("pos", sincos_2d(dim, self.g, self.g)[None])
        self.t_dim = t_dim
        self.t_mlp = nn.Sequential(nn.Linear(t_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        if n_classes:  # index n_classes = null class (dropout / unconditional)
            self.y_emb = nn.Embedding(n_classes + 1, dim)
            nn.init.normal_(self.y_emb.weight, std=0.02)
        self.blocks = nn.ModuleList(
            [DiTBlock(dim, heads, mlp_ratio, act) for _ in range(depth)]
        )
        self.final_norm = nn.RMSNorm(dim, elementwise_affine=False)
        self.final_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.final = nn.Linear(dim, pdim)
        nn.init.zeros_(self.final_mod[1].weight)
        nn.init.zeros_(self.final_mod[1].bias)
        nn.init.zeros_(self.final.weight)
        nn.init.zeros_(self.final.bias)

    def patchify(self, z):  # (B,C,H,W) -> (B, g*g, C*p*p) with (c,py,px) feature order
        B, C, H, W = z.shape
        p, g = self.patch, self.g
        z = z.view(B, C, g, p, g, p).permute(0, 2, 4, 1, 3, 5)  # B,gh,gw,C,py,px
        return z.reshape(B, g * g, C * p * p)

    def unpatchify(self, x):
        B = x.shape[0]
        p, g, C = self.patch, self.g, self.z_ch
        x = x.view(B, g, g, C, p, p).permute(0, 3, 1, 4, 2, 5)
        return x.reshape(B, C, g * p, g * p)

    def forward(self, z, t, y=None):
        x = self.embed(self.patchify(z)) + self.pos
        c = self.t_mlp(timestep_embedding(t, self.t_dim))
        if self.n_classes:
            if y is None:  # null class = unconditional
                y = torch.full((z.shape[0],), self.n_classes,
                               dtype=torch.long, device=z.device)
            if self.training and self.class_dropout > 0:
                drop = torch.rand(y.shape, device=y.device) < self.class_dropout
                y = torch.where(drop, torch.full_like(y, self.n_classes), y)
            c = c + self.y_emb(y)
        for blk in self.blocks:
            x = blk(x, c)
        s, b = self.final_mod(c)[:, None].chunk(2, dim=-1)
        x = self.final(self.final_norm(x) * (1 + s) + b)
        return self.unpatchify(x)


# ---------------------------------------------------------------- U-Net variant

class ChanRMSNorm(nn.Module):
    """RMSNorm over the channel dim of (B,C,H,W), no affine (adaLN provides it)."""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(dim=1, keepdim=True) + self.eps).to(x.dtype)


class UNetResBlock(nn.Module):
    def __init__(self, c_in, c_out, cond_dim, act):
        super().__init__()
        self.norm1 = ChanRMSNorm()
        self.act1 = make_act(act)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.norm2 = ChanRMSNorm()
        self.act2 = make_act(act)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 2 * c_in + 2 * c_out + c_out))
        nn.init.zeros_(self.mod[1].weight)
        nn.init.zeros_(self.mod[1].bias)
        self.c_in, self.c_out = c_in, c_out

    def forward(self, x, c):
        m = self.mod(c)[:, :, None, None]
        s1, b1 = m[:, : self.c_in], m[:, self.c_in : 2 * self.c_in]
        s2 = m[:, 2 * self.c_in : 2 * self.c_in + self.c_out]
        b2 = m[:, 2 * self.c_in + self.c_out : 2 * self.c_in + 2 * self.c_out]
        g = m[:, 2 * self.c_in + 2 * self.c_out :]
        h = self.conv1(self.act1(self.norm1(x) * (1 + s1) + b1))
        h = self.conv2(self.act2(self.norm2(h) * (1 + s2) + b2))
        return self.skip(x) + g * h


class UNetAttn(nn.Module):
    def __init__(self, ch, cond_dim, head_dim=32):
        super().__init__()
        self.norm = ChanRMSNorm()
        self.attn = Attention(ch, max(1, ch // head_dim))
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 3 * ch))
        nn.init.zeros_(self.mod[1].weight)
        nn.init.zeros_(self.mod[1].bias)
        self.ch = ch

    def forward(self, x, c):
        B, C, H, W = x.shape
        s, b, g = self.mod(c)[:, :, None, None].chunk(3, dim=1)
        h = (self.norm(x) * (1 + s) + b).flatten(2).transpose(1, 2)  # B,HW,C
        h = self.attn(h).transpose(1, 2).view(B, C, H, W)
        return x + g * h


class UNet(nn.Module):
    def __init__(self, channels=(32, 64, 96), blocks_per_stage=2, attn_res=(8, 4),
                 act="gelu", z_ch=4, z_hw=16, t_dim=256, cond_dim=128):
        super().__init__()
        self.t_dim = t_dim
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim)
        )
        self.stem = nn.Conv2d(z_ch, channels[0], 3, padding=1)

        res = z_hw
        self.enc = nn.ModuleList()
        self.downs = nn.ModuleList()
        enc_ch = []
        for i, ch in enumerate(channels):
            stage = nn.ModuleList()
            for _ in range(blocks_per_stage):
                stage.append(UNetResBlock(ch, ch, cond_dim, act))
                if res in attn_res:
                    stage.append(UNetAttn(ch, cond_dim))
            self.enc.append(stage)
            enc_ch.append(ch)
            if i < len(channels) - 1:
                self.downs.append(nn.Conv2d(ch, channels[i + 1], 3, stride=2, padding=1))
                res //= 2

        self.mid = nn.ModuleList([
            UNetResBlock(channels[-1], channels[-1], cond_dim, act),
            UNetAttn(channels[-1], cond_dim),
            UNetResBlock(channels[-1], channels[-1], cond_dim, act),
        ])

        self.dec = nn.ModuleList()
        self.ups = nn.ModuleList()
        for i in reversed(range(len(channels))):
            ch = channels[i]
            stage = nn.ModuleList()
            in_ch = ch + (channels[i + 1] if i < len(channels) - 1 else channels[-1])
            for j in range(blocks_per_stage):
                stage.append(UNetResBlock(in_ch if j == 0 else ch, ch, cond_dim, act))
                if res in attn_res:
                    stage.append(UNetAttn(ch, cond_dim))
            self.dec.append(stage)
            if i > 0:
                self.ups.append(nn.Upsample(scale_factor=2, mode="nearest"))
                res *= 2

        self.out_norm = ChanRMSNorm()
        self.out_act = make_act(act)
        self.out = nn.Conv2d(channels[0], z_ch, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, z, t):
        c = self.t_mlp(timestep_embedding(t, self.t_dim))
        h = self.stem(z)
        skips = []
        for i, stage in enumerate(self.enc):
            for blk in stage:
                h = blk(h, c)
            skips.append(h)
            if i < len(self.downs):
                h = self.downs[i](h)
        for blk in self.mid:
            h = blk(h, c)
        for i, stage in enumerate(self.dec):
            if i > 0:
                h = self.ups[i - 1](h)
            h = torch.cat([h, skips[-1 - i]], dim=1)
            for blk in stage:
                h = blk(h, c)
        return self.out(self.out_act(self.out_norm(h)))


def build_model(cfg):
    act = cfg["act"]
    z_ch, z_hw = cfg.get("latent_ch", 4), cfg.get("latent_hw", 16)
    if cfg["arch"] == "dit":
        d = cfg["dit"]
        return DiT(d["dim"], d["depth"], d["heads"], d["patch"], d["mlp_ratio"],
                   act=act, z_ch=z_ch, z_hw=z_hw, t_dim=cfg["t_embed_dim"],
                   n_classes=cfg.get("n_classes", 0),
                   class_dropout=cfg.get("class_dropout", 0.0),
                   pos_embed=d.get("pos_embed", "sincos"))
    d = cfg["unet"]
    return UNet(tuple(d["channels"]), d["blocks_per_stage"], tuple(d["attn_res"]),
                act=act, z_ch=z_ch, z_hw=z_hw, t_dim=cfg["t_embed_dim"])


def device_params(model):
    """Params that ship to the device (excludes the timestep-conditioning tower)."""
    total = 0
    for name, p in model.named_parameters():
        if (".mod." in name or "t_mlp" in name or "final_mod" in name
                or "y_emb" in name):  # conditioning tower: folded at export
            continue
        total += p.numel()
    return total


if __name__ == "__main__":
    for arch, sub in (("dit", {}), ("unet", {})):
        cfg = {
            "arch": arch, "act": "gelu", "t_embed_dim": 256,
            "dit": {"dim": 128, "depth": 8, "heads": 4, "patch": 2, "mlp_ratio": 4},
            "unet": {"channels": [32, 64, 96], "blocks_per_stage": 2, "attn_res": [8, 4]},
        }
        m = build_model(cfg)
        n = sum(p.numel() for p in m.parameters())
        v = m(torch.randn(2, 4, 16, 16), torch.rand(2))
        print(f"{arch}: total {n/1e6:.2f}M  device {device_params(m)/1e6:.2f}M  out {tuple(v.shape)}")
