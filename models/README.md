# models/

One directory per released model; the three yamls are the single source of
truth for every stage, and all tools take `--model <name>`:

| file | consumed by | defines |
|---|---|---|
| `vae.yaml` | `train/vae/*`, `train/latents.py`, `quant/fold.py` | dataset, encoder/decoder architecture (`dec_plan` is authoritative), VAE training run |
| `dit.yaml` | `train/dit/train_dit.py`, `quant/qat.py` | latent geometry, DiT size, rectified-flow training schedule |
| `export.yaml` | `quant/calibrate.py`, `quant/qat.py`, `quant/fold.py`, `scripts/*` | which checkpoints/calib to fold, sampling schedule, guidance ws, bit widths, output paths |

Paths in the yamls are repo-root-relative. Released inputs point into
`checkpoints/<name>/` (tracked); generated outputs land in
`artifacts/<name>/` (gitignored scratch).

## The two models

- **`m3_decD_deep_full`** — the quality flagship: depth-12 DiT + decoder D
  on blob v8, shipped as a 45k distillation-QAT checkpoint folded with its
  frozen calibration. The export.yaml header documents the full recipe and
  why full attention is required (the slim variant's block-10 attention drop
  structurally damages plain-mode generation).
- **`m3_long_cfg`** — the fast build: depth-8 DiT + small decoder on blob
  v4 with a union (base ∪ guided-w8) calibration.

Both share one latent space (the m3_long_cfg VAE encoder, 16×16×8, f8): the
flagship's DiT trains on latents produced by the fast model's VAE, and its
`vae.yaml` retrains only the decoder against that frozen encoder. For a full
retrain, do `m3_long_cfg` (VAE + latents) first — see
[../train/README.md](../train/README.md).
