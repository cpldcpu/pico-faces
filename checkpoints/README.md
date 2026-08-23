# checkpoints/

The released, git-tracked artifacts of both models — everything needed to
reproduce the UF2s byte-for-byte without training (`scripts/finalize.sh
<model>`), plus the fp teacher needed to reproduce the QAT stage.

## Per-model contents

| file | what it is |
|---|---|
| `dit_qat.pt` | the shipped DiT checkpoint (int8-ready after fold) |
| `vae_final.pt` | VAE (frozen encoder + the deployed decoder) |
| `calib_cfg.npz` | **frozen** activation calibration the checkpoint was trained/folded with |
| `calib_base.npz` | pre-guidance base calibration (provenance for the union calib) |
| `latent_stats.npz` | latent per-channel mean/std (slim extract; `finalize.sh` stages it where `fold.py` expects `latents.npz`) |
| `model.bin` | reference folded blob — a re-fold must reproduce this byte-exactly |
| `rf_cfg.h` | generated engine config the blob was built with |
| `goldens/` | end-to-end goldens (`.npz` int-sim latents + `.rgb` images) for the byte-exact gate |

## Provenance

- **`m3_decD_deep_full/`** (2026-07-17): `dit_qat.pt` = 45k self-distillation
  QAT (15k flat-LR + 30k cosine, teacher = `dit_fp.pt`, 50% trajectory-pool
  batches; recipe in the model's export.yaml). `dit_fp.pt` = the base fp
  teacher (500k + 100k cosine cooldown) — kept so the QAT stage is
  reproducible without a retrain. Device gen-FID 53.8 (N=5000, K=8 w=4).
  Golden CRC `40c5e5a0` (seed 1: K=4, class 1, w=4).
- **`m3_long_cfg/`** (2026-07-12, refolded v4 on the current engine):
  `dit_qat.pt` = 300k rectified-flow + 15k QAT. Golden CRC `b32e6e63`
  (seed 1: K=4, class 1, w=4).

## Architecture: where the parameters hide

Exact counts, generated from these configs/checkpoints. Three things are
easy to miss: most of the VAE never ships (the big encoder is train-only),
the DiT's conditioning tower (~35% of its trainable params) never ships
either — it is *folded* into the step tables — and as a result about a
quarter of each `model.bin` is not weights at all.

### DiT (dim 128, 4 heads, 2×2 patches → 64 tokens × 32 features)

| component | `m3_long_cfg` (depth 8) | `m3_decD_deep_full` (depth 12) | ships? |
|---|---:|---:|---|
| patch embed (32→128) | 4,224 | 4,224 | ✔ int8 |
| attention (qkv + proj + qk-RMSNorm) | 528,896 | 793,344 | ✔ int8 |
| MLPs (fc1 128→512, fc2 512→128) | 1,053,696 | 1,580,544 | ✔ int8 |
| output head (128→32) | 4,128 | 4,128 | ✔ int8 |
| **device subtotal** | **1,590,944** | **2,382,240** | |
| adaLN per-block `mod` MLPs | 792,576 | 1,188,864 | ✘ folded |
| timestep tower `t_mlp` | 49,408 | 49,408 | ✘ folded |
| class embedding `y_emb` | 640 | 640 | ✘ folded |
| `final_mod` | 33,024 | 33,024 | ✘ folded |
| **trainable total** | **2,466,592** | **3,654,176** | |

"Folded": the sampler runs a FIXED schedule (8 steps × 5 condition sets),
so every adaLN modulation the tower could ever produce is precomputed at
export into per-(class, step, block) tables — RMSNorm gains/biases plus
requant multipliers. The tower trains, then evaporates. This also explains
`dit_qat.pt` = 29.4 MB: 3.65M params × fp32 × (model + EMA copies).

### VAE (per model; shared frozen encoder)

| component | `m3_long_cfg` | `m3_decD_deep_full` | ships? |
|---|---:|---:|---|
| encoder (128/64/32/16 px, ch 64→384) | 9,726,992 | 9,726,992 | ✘ train-only |
| decoder | 115,755 | 494,211 ("D") | ✔ int8 |

The asymmetry is the design: a heavy encoder organizes the latent so a
tiny decoder can render it. 98.8% (fast) / 95.2% (quality) of the VAE
exists only to shape the latent space during training.

### Byte budget of the shipped `model.bin`

| section | `m3_long_cfg` (2,567,828 B) | `m3_decD_deep_full` (4,016,632 B) |
|---|---:|---:|
| DiT block weights (int8) + per-channel requant | 1,656,832 | 2,482,416 |
| **step tables** (folded conditioning: 5 cond × 8 steps × depth) | 737,280 | 983,040 |
| VAE decoder (int8, BN-folded) | 117,603 | 498,411 |
| positional embedding (int8) | 16,384 | 16,384 |
| final-norm gain/bias tables | 20,480 | 20,480 |
| schedule, LUTs, scales, misc | ~19,000 | ~16,000 |

So ~29% (fast) / ~24% (quality) of the blob is the *ghost of the
conditioning tower* — precomputed modulations, not weights. That is the
price of running zero conditioning compute on device, and it scales
linearly with the number of Euler steps K (the reason a native K=16 fold
was rejected as unshippable, and why the v8 format's table renorm was
worth −127 KB).

## Rules

1. **Frozen calib:** a QAT checkpoint is folded with the SAME calibration it
   trained against. Never recalibrate after QAT.
2. **Teacher pinning:** any further distillation uses `dit_fp.pt` as the
   teacher, never a distilled student (distill-of-distill compounds errors).
3. A re-fold that does not `cmp`-match `model.bin` here means the code or
   checkpoint changed — `finalize.sh` prints which.
