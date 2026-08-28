# checkpoints/

The released, git-tracked artifacts of both models. Deployment scripts: `scripts/finalize.sh
<model>`.

## Per-model contents

| file | what it is |
|---|---|
| `dit_qat.pt` | the shipped DiT checkpoint  |
| `vae_final.pt` | VAE (frozen encoder + the deployed decoder) |
| `calib_cfg.npz` | **frozen** activation calibration the checkpoint was trained/folded with |
| `calib_base.npz` | pre-guidance base calibration |
| `latent_stats.npz` | latent per-channel mean/std  |
| `model.bin` | reference folded blob |
| `rf_cfg.h` | generated engine config the blob was built with |
| `goldens/` | end-to-end goldens (`.npz` int-sim latents + `.rgb` images) |

## Training History

- **`m3_decD_deep_full/`** (2026-07-17): `dit_qat.pt` = 60k self-distillation
  QAT (15k flat-LR + 45k cosine, teacher = `dit_fp.pt`, 50% trajectory-pool
  batches; recipe in the model's export.yaml). `dit_fp.pt` = the base fp
  teacher (500k steps, last 100k cosine cooldown) — kept so the QAT stage is
  reproducible without a retrain. Device gen-FID 53.8 (N=5000, K=8 w=4).
  Golden CRC `40c5e5a0` (seed 1: K=4, class 1, w=4). Folded adaLN step
  tables inside `model.bin`: **960 KiB** of 3.83 MiB (5 cond × 8 steps ×
  12 blocks).
- **`m3_long_cfg/`** (2026-07-12, refolded v4 on the current engine):
  `dit_qat.pt` = 300k rectified-flow + 15k QAT. Golden CRC `b32e6e63`
  (seed 1: K=4, class 1, w=4). Folded adaLN step tables inside
  `model.bin`: **720 KiB** of 2.45 MiB (5 cond × 8 steps × 8 blocks).

## Architecture

Exact counts, generated from these configs/checkpoints. 
The DiT's conditioning tower (~35% of its trainable params) 
is *folded* into the step tables.

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
requant multipliers. 

### VAE 

| component | `m3_long_cfg` | `m3_decD_deep_full` | ships? |
|---|---:|---:|---|
| encoder (128/64/32/16 px, ch 64→384) | 9,726,992 | 9,726,992 | ✘ train-only |
| decoder | 115,755 | 494,211 ("D") | ✔ int8 |

The asymmetry is intentional: A heavy encoder organizes the latent so a
tiny decoder can render it. The larger model used a deeper decoder trained
on the same frozen latents.

### Byte budget of the shipped `model.bin`

| section | `m3_long_cfg` (2,567,828 B) | `m3_decD_deep_full` (4,016,632 B) |
|---|---:|---:|
| DiT block weights (int8) + per-channel requant | 1,656,832 | 2,482,416 |
| **step tables** (folded conditioning: 5 cond × 8 steps × depth) | 737,280 | 983,040 |
| VAE decoder (int8, BN-folded) | 117,603 | 498,411 |
| positional embedding (int8) | 16,384 | 16,384 |
| final-norm gain/bias tables | 20,480 | 20,480 |
| schedule, LUTs, scales, misc | ~19,000 | ~16,000 |
