# quant/

The int8 pipeline: everything between a trained checkpoint and a verified
`model.bin`. Two halves live here:

**Training-side (GPU):**

| file | role |
|---|---|
| `calibrate.py` | activation scales from 512 real sampling trajectories (not dataset latents — z_t statistics differ per step). `--guidance-w N --union <base>` builds the union calib for CFG builds |
| `fake_quant.py` | the QAT view of the DiT: per-channel weight/activation grids, ReLU² square-requant semantics, clamp-first STE — mirrors fold.py's deployed grids exactly |
| `qat.py` | the QAT finetune. The shipping recipe is **self-distillation**: `--distill` (match the FROZEN fp teacher's velocities), `--cosine` (LR decay), `--traj-pool N` (50% of batches from fp teacher trajectory states incl. guided w), `--teacher-ckpt` (pin the teacher to the base fp checkpoint) |

**Deploy-side (CPU, deterministic):**

| file | role |
|---|---|
| `fold.py` | folds a checkpoint + frozen calib into integer tensors: BN→conv, per-(class, step) adaLN → RMSNorm gains/biases + requant multipliers, v8 scalar-shift renorm (`renorm_Ms`) |
| `export.py` | writes `model.bin` (+ `rf_cfg.h`, end-to-end goldens). Format versions: v4 = CFG builds, v8 = adds per-channel act grids, square-requant, scalar-shift step tables |
| `int_sim.py` | **the golden reference**: an exact integer numpy simulator of the engine. Same accumulation order, same one rounding rule. The C engine is verified against its output byte-for-byte |
| `int_ops.py` | the integer op primitives (requant, int rsqrt, int softmax, …) shared by int_sim |

## The rules that keep this correct

1. **Frozen calib:** fold a QAT checkpoint with the SAME calibration it
   trained against. Never recalibrate after QAT.
2. **Distill, don't data-target:** MSE-to-data QAT on a cosine-cooled
   checkpoint pulls the EMA off its optimum (measured: FID 54.2 → 55.3).
   Distilling the fp teacher is FID-safe and halves the fp↔device per-step
   error. Pin the teacher to the base fp checkpoint.
3. **Union calib for guidance:** guided activations run hotter; calibrate on
   base ∪ guided-w8 trajectories (per-key max).
4. Any change here must keep `scripts/verify_model.sh` green (int_sim ↔
   desktop C byte-exact) — and a pure refactor must reproduce
   `checkpoints/<model>/model.bin` byte-identically.

## Reproducing the released flagship QAT

```
python quant/qat.py --model m3_decD_deep_full --distill --cosine \
    --traj-pool 512 --teacher-ckpt checkpoints/m3_decD_deep_full/dit_fp.pt
python quant/fold.py --model m3_decD_deep_full
```

(15k steps per invocation; the released checkpoint is 15k flat-LR + 30k
cosine continuation. See the model's export.yaml header for the full story.)
