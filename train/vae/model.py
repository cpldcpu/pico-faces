"""Asymmetric VAE: a ~10M-param encoder (train-only, standard ResBlocks/GroupNorm)
and a ~112K-param quantization-friendly decoder that ships to the RP2350.

Decoder design rules (int8 deployment):
- plain Conv3x3 + BatchNorm + ReLU only (BN folds into conv weights offline,
  ReLU becomes a clamp at requantization)
- nearest-neighbor 2x upsampling (integer-exact index duplication)
- no residual connections, no GroupNorm, final conv linear
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return self.skip(x) + h


class Encoder(nn.Module):
    def __init__(self, channels=(64, 128, 256, 384), z_ch=4, img_ch=1):
        super().__init__()
        self.stem = nn.Conv2d(img_ch, channels[0], 3, padding=1)
        stages = []
        for i, c in enumerate(channels):
            stages += [ResBlock(c, c), ResBlock(c, c)]
            if i < len(channels) - 1:
                stages.append(nn.Conv2d(c, channels[i + 1], 3, stride=2, padding=1))
        self.stages = nn.Sequential(*stages)
        self.head = nn.Sequential(
            nn.GroupNorm(32, channels[-1]),
            nn.SiLU(),
            nn.Conv2d(channels[-1], 2 * z_ch, 3, padding=1),
        )

    def forward(self, x):
        h = self.head(self.stages(self.stem(x)))
        mu, logvar = h.chunk(2, dim=1)
        return mu, logvar.clamp(-30, 20)


def conv_bn_relu(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class Decoder(nn.Module):
    """Latent -> image as a flat plan of (c_out, up_before) conv_bn_relu layers
    plus a linear out conv; quant/fold.py and the C engine mirror it 1:1.
    Default plan is the m1 layout: 16x16x4 -> 128x128x1, ~112K params."""

    def __init__(self, channels=(64, 32, 16, 8), z_ch=4, img_ch=1, plan=None):
        super().__init__()
        if plan is None:  # legacy m1 layout derived from the 4-channel spec
            c0, c1, c2, c3 = channels
            plan = [(c0, 0), (c0, 0), (c0, 0), (c1, 1), (c1, 0),
                    (c2, 1), (c2, 0), (c3, 1), (c3, 0)]
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        body, c_in, up_before = [], z_ch, set()
        for i, (c_out, up) in enumerate(plan):
            if up:
                up_before.add(i)
            body.append(conv_bn_relu(c_in, c_out))
            c_in = c_out
        self.body = nn.ModuleList(body)
        self.out = nn.Conv2d(c_in, img_ch, 3, padding=1)  # linear
        self.up_before = up_before  # indices where NN-upsample precedes the conv

    def forward(self, z):
        h = z
        for i, layer in enumerate(self.body):
            if i in self.up_before:
                h = self.up(h)
            h = layer(h)
        return self.out(h)


class VAE(nn.Module):
    def __init__(self, enc_channels=(64, 128, 256, 384), dec_channels=(64, 32, 16, 8),
                 z_ch=4, img_ch=1, dec_plan=None):
        super().__init__()
        self.encoder = Encoder(enc_channels, z_ch, img_ch)
        self.decoder = Decoder(dec_channels, z_ch, img_ch, dec_plan)

    def encode(self, x):
        return self.encoder(x)

    def reparameterize(self, mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


def build_vae(vcfg):
    """Construct a VAE from a models/<name>/vae.yaml dict."""
    plan = [tuple(x) for x in vcfg["dec_plan"]] if "dec_plan" in vcfg else None
    return VAE(tuple(vcfg["enc_channels"]), tuple(vcfg["dec_channels"]),
               vcfg["latent_ch"], img_ch=vcfg.get("img_ch", 1), dec_plan=plan)


if __name__ == "__main__":
    m = VAE()
    n_enc = sum(p.numel() for p in m.encoder.parameters())
    n_dec = sum(p.numel() for p in m.decoder.parameters())
    print(f"encoder {n_enc/1e6:.2f}M  decoder {n_dec/1e3:.1f}K params")
    x = torch.randn(2, 1, 128, 128)
    r, mu, lv = m(x)
    print(r.shape, mu.shape)
