# pico-faces

Latent rectified-flow face generators running **entirely on a Raspberry Pi
Pico 2** (RP2350, $5 microcontroller). No PC in the loop: a PRNG seeds the
latent noise, a fully-int8 diffusion transformer integrates the flow ODE in
8 Euler steps on the two Cortex-M33 cores, and an int8 VAE decoder expands
the result to a 128×128 RGB face — streamed over USB or shown live on VGA.

Everything from seed to pixels is **exact integer arithmetic** with one
specified rounding rule, so the numpy simulator, the desktop C engine, and
the microcontroller produce byte-identical images.

## The two released models

| | `m3_decD_deep_full` (quality) | `m3_long_cfg` (fast) |
|---|---|---|
| DiT | dim 128 × depth 12, 2.37M params | dim 128 × depth 8, 1.59M params |
| VAE decoder | "D", ~493K params | small, ~116K params |
| Blob format | v8 (4.02MB, ~100KB flash margin) | v4 (2.57MB) |
| Gen-FID (device, N=5000, K=8 w=4) | **53.8** (fp reference: 52.4) | — (speed build) |
| Time / image | ~10 s @ K=4 w=4 (≈20 s @ K=8 w=8) | ~4.3 s @ K=4 w=4 (5.4 s @ K=8 plain) |
| Golden CRC (seed 1: K=4, class 1, w=4) | `40c5e5a0` | `b32e6e63` |

Both are class-conditional (gender × smile: 0 f/neutral, 1 f/smile,
2 m/neutral, 3 m/smile) with classifier-free guidance, `w ∈ {4, 6, 8}`,
selectable per request at runtime. The quality model's int8 checkpoint was
healed with a 45k-step self-distillation QAT — the folded int8 model beats
its own post-training quantization on FID and sits 1.6–3 gray levels from
the fp teacher per pixel.

## Quickstart: flash and generate

1. Hold BOOTSEL while plugging in a Pico 2, copy
   [uf2/pico_faces_m3_decD_deep_full.uf2](uf2/) onto the `RPI-RP2` drive.
2. `pip install pyserial matplotlib numpy pillow`, then:

   ```
   python viewer/view_serial.py --port COM10 --seed 1 --show \
       --expect checkpoints/m3_decD_deep_full/goldens/golden_1.rgb
   ```

   The viewer requests an image (`G <seed> [k] [class] [w]`), displays it,
   and byte-compares it against the released golden (CRC `40c5e5a0`) —
   your board reproduces the release bit-for-bit or not at all.

Optional: on a Pimoroni VGA Demo Base the firmware scans out the image at
640×480 while generating.

## Recreating the UF2s

**Path A — from the released checkpoints (no GPU, ~minutes).**

```
pip install -r requirements.txt   # plus torch (CPU is fine for folding)
bash scripts/finalize.sh m3_decD_deep_full
bash scripts/finalize.sh m3_long_cfg
```

This folds the released QAT checkpoint with its **frozen** calibration into
`model.bin`, verifies the desktop C engine byte-exact against the released
goldens, confirms the blob is byte-identical to `checkpoints/<model>/model.bin`,
and builds the UF2 (needs the [Pico SDK](firmware/README.md)).

**Path B — full retrain (CUDA GPU, ~a day).** Dataset download → VAE →
decoder D → latents → DiT → calibration → distillation-QAT → path A. The
stage-by-stage commands live in [train/README.md](train/README.md) and
[quant/README.md](quant/README.md); training is seeded but GPU nondeterminism
means your checkpoints (and CRCs) will differ — the byte-exact contract then
holds for *your* fold.

## Repository map

| dir | contents |
|---|---|
| [models/](models/) | the two model definitions (`vae.yaml`, `dit.yaml`, `export.yaml`) |
| [checkpoints/](checkpoints/) | released weights, calibrations, reference blobs, goldens |
| [data/](data/) | FFHQ download + gender×smile label scripts |
| [train/](train/) | PyTorch training: VAE, decoder-only retrain, rectified-flow DiT |
| [quant/](quant/) | int8 pipeline: calibration, distillation-QAT, folding, export, exact numpy simulator |
| [engine/](engine/) | portable C99 inference engine (int-only datapath) + desktop harness |
| [firmware/](firmware/) | RP2350 firmware: dual-core dispatch, flash streaming, USB, VGA |
| [scripts/](scripts/) | the drivers: `finalize.sh`, `verify_model.sh`, `build_firmware.sh`, `train_model.sh` |
| [viewer/](viewer/) | PC-side serial viewer / golden checker |
| [uf2/](uf2/) | the two flashable images |

## Determinism contract

`quant/int_sim.py` (numpy) ↔ `engine/desktop` (x86 C) ↔ firmware (M33) must
produce byte-identical images for the same seed. Every released model ships
its end-to-end goldens in `checkpoints/<model>/goldens/`;
`scripts/verify_model.sh` enforces the contract on every re-fold. If you
change anything in `quant/` or `engine/`, this gate is the arbiter.

## Environment

- Training: Linux/WSL, PyTorch ≥ 2.4 with CUDA (developed on torch
  2.9/cu129, RTX 5090). Folding/verification: any Python 3.10+ with torch
  CPU. `requirements.txt` covers the rest.
- Firmware: Pico SDK 2.2.0, `arm-none-eabi-gcc`, cmake ≥ 3.13
  ([firmware/README.md](firmware/README.md)).
