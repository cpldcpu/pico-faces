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

## Rules

1. **Frozen calib:** a QAT checkpoint is folded with the SAME calibration it
   trained against. Never recalibrate after QAT.
2. **Teacher pinning:** any further distillation uses `dit_fp.pt` as the
   teacher, never a distilled student (distill-of-distill compounds errors).
3. A re-fold that does not `cmp`-match `model.bin` here means the code or
   checkpoint changed — `finalize.sh` prints which.
