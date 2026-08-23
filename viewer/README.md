# viewer/

PC-side serial client for a flashed Pico 2. Requests a generation over
USB-CDC, prints the device-reported CRC and timing, saves the frame as a
PNG, and can byte-compare it against a released golden.

```
# generate and look at one face
python viewer/view_serial.py --port COM10 --seed 42 --show

# prove the board reproduces the release bit-for-bit
python viewer/view_serial.py --port COM10 --seed 1 \
    --expect checkpoints/m3_decD_deep_full/goldens/golden_1.rgb
```

(`COM10` on Windows; `/dev/ttyACM0`-style on Linux.)

## Parameters

| flag | range / values | meaning |
|---|---|---|
| `--port` | required | serial port of the flashed Pico 2 |
| `--seed` | 0 … 2³²−1 (default 1) | PRNG seed for the latent noise. The same seed produces a **bit-identical** image on every board, the desktop engine, and the numpy simulator |
| `--steps` | 8, 4, 2, 1 (default 4) | Euler integration steps. 8 = the native baked schedule (best quality); 4/2/1 stride it (faster, progressively softer). Only power-of-2 divisors of 8 are valid — the engine rescales the folded step size by an exact shift |
| `--class` | 0–3, else unconditional | see encoding below; omitted → device default `seed % 5` |
| `--cfg` | 4, 6, 8; other = plain | classifier-free guidance strength `w`. Only the baked values 4/6/8 exist (folded per-w tables); any other value falls back to plain sampling. Requires `--class`. Guided sampling runs the DiT twice per step (~2× generation time). With BOTH `--class` and `--cfg` absent, the device uses the golden convention `w_idx = seed % 4 − 1` (so released goldens reproduce over USB) |
| `--out` | path (default `out/`) | output directory; saves `seed_<N>.png` |
| `--show` | flag | after saving, upscale the frame to 512×512 (nearest-neighbor, so you see the real pixels) and open it in the system image viewer |
| `--expect` | path to `.rgb`/`.gray` | byte-compare the received image against a golden file and print BYTE-EXACT / MISMATCH |

`--cls` and `--w` are accepted as legacy aliases for `--class` / `--cfg`.

## What the class value encodes

The models are conditioned on **gender × smile**, with labels derived from
the FFHQ facial-attribute annotations
([ffhq-features-dataset](https://github.com/DCGM/ffhq-features-dataset)):

| class | meaning |
|---|---|
| 0 | female, neutral |
| 1 | female, smiling |
| 2 | male, neutral |
| 3 | male, smiling |
| 4 (or any other value / absent guidance) | unconditional — the trained null class (also used internally as the CFG negative) |

Conditioning is soft: classes steer identity/expression statistics, and
higher `--cfg` values enforce them (and overall structure) more strongly.
`w=4` is the balanced default; `w=8` is the strongest, most-typical look.

## Golden CRCs

Seed 1, K=4, class 1, w=4: `m3_decD_deep_full` → `40c5e5a0`,
`m3_long_cfg` → `b32e6e63` — the values the firmware prints and
`--expect` verifies against `checkpoints/<model>/goldens/golden_1.rgb`.

Needs `pyserial` + `pillow` (both in [requirements.txt](../requirements.txt)).
