# engine/

Portable C99 inference engine. Integer-only datapath (no float, no malloc,
static arenas), identical accumulation order to `quant/int_sim.py` — the
same seed produces byte-identical images in numpy, on x86, and on the
RP2350.

| file | role |
|---|---|
| `include/rf_model.h` | blob layout structs + parse API |
| `include/rf_ops.h` | kernel signatures |
| `include/rf_cfg.h` | default model geometry; each fold generates a model-specific one next to its `model.bin` |
| `src/graph.c` | `model.bin` parser (format v3–v8) + generation orchestration |
| `src/dit.c` | the DiT step: attention, MLP, int RMSNorm/softmax, per-step folded conditioning, CFG blend (int32-difference — immune to high-w cancellation) |
| `src/vae_dec.c` | int8 conv decoder with fused NN-upsample addressing |
| `src/hires.c` | optional 256×256 pixel-shuffle head (present in the engine; not used by the two released models) |
| `src/kernels_ref.c` | portable reference kernels — the spec |
| `src/prng.c` | PCG32 + CLT-12 Gaussian, bit-exact across numpy/C |
| `desktop/main_golden.c` | x86 harness: `rf_golden <model.bin> <outdir> <k> <seeds…>` → `.rgb` frames |
| `desktop/main_probe.c`, `desktop/test_ops.c` | debugging/bring-up harnesses |

## Desktop build & run

`scripts/verify_model.sh <model>` compiles and runs the golden check. By
hand, against the released reference blob (no Python needed):

```
gcc -O2 -Icheckpoints/m3_decD_deep_full -Iengine/include \
    engine/desktop/main_golden.c engine/src/*.c -o rf_golden
./rf_golden checkpoints/m3_decD_deep_full/model.bin out 4 1 2 3
cmp out/eng_1.rgb checkpoints/m3_decD_deep_full/goldens/golden_1.rgb
```

The firmware compiles these same sources for the M33 plus RP2350-specific
staging/dispatch (see [../firmware/](../firmware/)). The kernels are written
so pairwise (SMLAD-style) accumulation matches the scalar order exactly
(`include/rf_ops.h` documents the invariant) — that is what makes the
byte-exact contract hold on-device.
