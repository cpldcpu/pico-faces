# uf2/

Flashable images. Hold BOOTSEL while plugging in the Pico 2, then copy a
`.uf2` onto the `RPI-RP2` drive. Verify with the [viewer](../viewer/):

```
python viewer/view_serial.py --port COM10 --seed 1 \
    --expect checkpoints/<model>/goldens/golden_1.rgb
```

| file | model | expected CRC (seed 1: K=4, class 1, w=4) | time / image |
|---|---|---|---|
| `pico_faces_m3_decD_deep_full.uf2` | the QUALITY flagship: full-attention depth-12 DiT + decoder D, 60k distillation-QAT, blob v8. Device gen-FID 53.8 (N=5000, K=8 w=4) | `40c5e5a0` | ~10 s @ K=4 w=4, ~20 s @ K=8 w=8 |
| `pico_faces_m3_long_cfg.uf2` | the FAST build: depth-8 DiT + small decoder, blob v4. Guided K=4 ≈ plain K=8 wall-clock | `b32e6e63` | ~4.3 s @ K=4 w=4, 5.4 s @ K=8 plain |

Both: 128×128 RGB out over USB-CDC, VGA scanout on a Pimoroni VGA Demo
Base, classes 0–3 (gender × smile) + unconditional, guidance w ∈ {4, 6, 8}.

Protocol: `G <seed> [k_steps] [class] [w]` → `RFI2` header (w, h, channels,
class) + image bytes + CRC32 + timings. `w` unmatched/absent on a class
request = plain sampling. With BOTH class and w absent, the golden
convention `w_idx = seed % (n_w+1) - 1` applies, so the released goldens
reproduce over USB.

Rebuild either image from the released checkpoints with
`bash scripts/finalize.sh <model>` — see the [top-level README](../README.md).
