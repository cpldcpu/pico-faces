# Pico-Faces

This project implements a **Gen AI image generator running on a $1 Microcontroller**, the RP2350, as used in the Raspberry Pi Pico 2. 

**Blog article with more details [here](here)**

---

## What does it do?

It can generate 128×128 RGB images of human faces in 10-20s each and display them on a VGA monitor or stream them over USB. The model implements a latent rectified-flow diffusion transformer (DiT), similar to what is used in diffusion models like Flux. There are two variants at 2.9 and 1.7 million parameters, 5000x times less than even a typical local diffusion model. It supports conditional generation in 5 classes (gender × smile). The model was trained on the [FFHQ dataset](https://github.com/nvlabs/ffhq-dataset).

It is more than astonishing that a model this small is able to generate complex images at all. Many MNIST toy diffusion projects use far more parameters and are barely able to generate anything coherent. Interestingly, a lot of the optimizations that helped large models and helped for this micro model. 

### Example outputs for the same seed and different classes

<div align="center">
   <img src="media/seed3_classes_k8_w6.png" alt="Example">
</div>

### Video of the device generating and displaying an image on a monitor

![media/pico_faces_monitor.mp4](media/pico_faces_monitor.mp4)

### Emergence of facial features as the number of steps increases

This indicates that the model does indeed behave like a diffusion model. Smaller diffusion models often tend to collapse to an initial bias without actual refinement in subsequent steps. This is not the case here.

<div align="center">
   <img src="media/grid_emergence_k8.png" alt="Example" width=50%>
</div>

### Parameter matrix for number of steps K and guidance strength w (CFG, classifier free guidance)

This demonstrates that the model scales as expected with both parameters.

<div align="center">
   <img src="media/kw_deepD_seed3.png" alt="Example" width=50%>
</div>

## Quickstart - run it on your own Pico 2

1. Hold BOOTSEL while resetting the RP2350 board, copy
   [uf2/pico_faces_m3_decD_deep_full.uf2](uf2/) onto the `RPI-RP2` drive.
2. `pip install pyserial matplotlib numpy pillow`, then:

   ```
   python viewer/view_serial.py --port com10 --seed 3 --steps 8 --class 4  --cfg 6 --show
   ```

Check the [viewer/README.md](viewer/README.md) for the full parameters descriptions.

Optional: on a Pimoroni VGA Demo Base the firmware displays the images on a connected VGA monitor. 

<div align="center">
   <img src="media/vga-board.jpg" alt="Board" width=40%>
</div>

### The included models

| | High quality model `m3_decD_deep_full` | Fast model `m3_long_cfg` |
|---|---|---|
| DiT | dim 128 × depth 12, 2.37M params | dim 128 × depth 8, 1.59M params |
| VAE decoder |  ~493K params | small, ~116K params |
| Blob size | 4.02MB | 2.57MB |
| Gen-FID (device, N=5000, K=8 w=4) | **53.8** (fp reference: 52.4) | — (speed build) |
| Time / image | ~10 s @ K=4 w=4 (≈20 s @ K=8 w=8) | ~4.3 s @ K=4 w=4 (5.4 s @ K=8 plain) |


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
means your checkpoints (and CRCs) will differ.

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