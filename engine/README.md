# engine/

Portable C99 inference engine. Integer-only datapath (no float, no malloc,
static arenas). Verified vs. numpy, x86 and RP2350. 

On RP2350, the engine is using CM33 intrinsics to speed up the int8 convs and matrix multiplies.

| file | role |
|---|---|
| `include/rf_model.h` | blob layout structs |
| `include/rf_ops.h` | kernel signatures, CM33 inline assembly macros |
| `include/rf_cfg.h` | default model geometry |
| `src/graph.c` | `model.bin` parser + generation orchestration |
| `src/dit.c` | the DiT step: attention, MLP, int RMSNorm/softmax, per-step folded conditioning, CFG blend |
| `src/vae_dec.c` | int8 conv decoder |
| `src/kernels_ref.c` | portable reference kernels |
| `src/prng.c` | PCG32 + CLT-12 Gaussian |
| `desktop/main_golden.c` | x86 harness: `rf_golden <model.bin> <outdir> <k> <seeds…>` → `.rgb` frames |
| `desktop/main_probe.c`, `desktop/test_ops.c` | debugging/bring-up harnesses |

## Desktop build & run

`scripts/verify_model.sh <model>` compiles and runs the golden check against the released reference blob:

```
gcc -O2 -Icheckpoints/m3_decD_deep_full -Iengine/include \
    engine/desktop/main_golden.c engine/src/*.c -o rf_golden
./rf_golden checkpoints/m3_decD_deep_full/model.bin out 4 1 2 3
cmp out/eng_1.rgb checkpoints/m3_decD_deep_full/goldens/golden_1.rgb
```

The firmware compiles these same sources for the M33 plus RP2350-specific
staging/dispatch (see [../firmware/](../firmware/)). 