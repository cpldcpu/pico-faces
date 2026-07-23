# firmware/

RP2350 (Pico 2) firmware. Compiles the [engine](../engine/) sources
unchanged and adds the device-specific machinery:

| file | role |
|---|---|
| `CMakeLists.txt` | Pico SDK build; `-DMODEL_BIN=<path>` selects the blob to embed |
| `main.c` | boot (VREG 1.30 V → 300 MHz, QMI divider kept flash-legal), USB-CDC protocol, per-stage DWT timing |
| `model_blob.S` | `.incbin`s `model.bin` into `.rodata` (XIP flash) |
| `stage.c` | paced DMA weight streaming from the uncached XIP alias into the SRAM ping-pong arena — hides flash latency behind compute |
| `par.c` | `par_for()`: core-1 dispatch over disjoint row halves via the inter-core FIFO, no locks |
| `vga_noirq.c` | interrupt-free VGA scanout (default, `RF_VGA=2`): the scanline generator runs in the idle gaps of generation |
| `vga.c` | pico-extras scanvideo fallback (`RF_VGA=1`, costs ~10–15% generation time in IRQs) |

## Build

Via the driver (recommended — resolves the model's blob automatically):

```
bash scripts/build_firmware.sh m3_decD_deep_full
```

Requirements: [Pico SDK](https://github.com/raspberrypi/pico-sdk) 2.2.0
(+ [pico-extras](https://github.com/raspberrypi/pico-extras) for the
scanvideo fallback only), `arm-none-eabi-gcc`, cmake. Point
`PICO_SDK_PATH` / `PICO_EXTRAS_PATH` at your copies. The UF2 lands in
`uf2/pico_faces_<model>.uf2`.

## USB protocol

`G <seed> [k_steps] [class] [w]\n` → `RFI2` header (w, h, channels, class)
+ raw image bytes + CRC32 + timing. `w` = guidance strength (4/6/8; absent
or unmatched = plain). With both class and w absent, the golden convention
`w_idx = seed % (n_w+1) - 1` applies, so released goldens reproduce over
USB. `viewer/view_serial.py` speaks the protocol.
