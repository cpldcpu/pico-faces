# firmware/

RP2350 (Pico 2) firmware. Compiles the [engine](../engine/) sources
unchanged and adds the device-specific machinery:

| file | role |
|---|---|
| `CMakeLists.txt` | Pico SDK build; `-DMODEL_BIN=<path>` selects the blob to embed |
| `main.c` | boot (VREG 1.30 V → 300 MHz, QMI divider kept flash-legal, `clk_hstx` for the DVI path), USB-CDC protocol, per-stage DWT timing |
| `model_blob.S` | `.incbin`s `model.bin` into `.rodata` (XIP flash) |
| `stage.c` | paced DMA weight streaming from the uncached XIP alias into the SRAM ping-pong arena |
| `par.c` | `par_for()`: core-1 dispatch over disjoint row halves via the inter-core FIFO, no locks |
| `vga_noirq.c` | interrupt-free VGA scanout (default, `RF_VGA=2`) |
| `vga.c` | pico-extras scanvideo fallback (`RF_VGA=1`, costs ~10–15% generation time in IRQs) |
| `dvi_hstx.c` | interrupt-free DVI/HSTX scanout for the Fruit Jam (`RF_DVI=1`) |

## Clocks

`RF_SYS_KHZ` (default 300000) is normally free — drop it to 150000 if a board
proves unstable. **`RF_DVI=1` couples it to the video timing.** HSTX is DDR and
spends 10 TMDS bits on a pixel, so `pixel clock = clk_hstx / 5`, and the mode
chosen for the Fruit Jam (640x480 in 952x525 at 60 Hz) needs `clk_hstx` at 
exactly 150 MHz. `clk_hstx` is an integer 1..3 divide of `clk_sys` with no 
fractional part, so `clk_sys` must be 1, 2 or 3 x 150 MHz: **300000 (div 2)
or 150000 (div 1)**. Any other `RF_SYS_KHZ` fails the build — in CMake, and
again as a `#error` in `main.c` for a hand-driven cmake. `RF_DVI` and `RF_VGA`
are mutually exclusive.

## Display backends

Both scanouts implement the same four functions — `rf_vga_init`,
`rf_vga_invalidate`, `rf_vga_dither` and the engine's `rf_step_hook` override.
`main.c` and the engine do not care which is linked. Both are built the
same way: a static DMA command list that loops in hardware, with zero
interrupts and zero CPU in steady state, so the display never competes with
generation.

`dvi_hstx.c` (`-DRF_DVI=1 -DRF_VGA=0 -DPICO_BOARD=adafruit_fruit_jam`) is the
Fruit Jam one. HSTX does the TMDS encoding in hardware. Its framebuffer is the
384x384 image band only, RGB332, 4 pixels per 32-bit word: the background and
the progress bar are `TMDS_REPEAT` runs, which cost one FIFO word each rather
than a screen of pixels.

`rf_vga_invalidate()` must point the image band at the static preview rows
*before* `rf_generate()` runs, and `rf_vga_dither()` rewrites each row's
commands as well as its pixels before pointing the band back. A garbage
command reaches the expander as a command, and a desynced expander never
recovers.

## Build

```
bash scripts/build_firmware.sh m3_decD_deep_full
```

Requirements: [Pico SDK](https://github.com/raspberrypi/pico-sdk) 2.2.0
(+ [pico-extras](https://github.com/raspberrypi/pico-extras) for VGA,
`arm-none-eabi-gcc`, cmake. Point
`PICO_SDK_PATH` / `PICO_EXTRAS_PATH` at your copies. The UF2 lands in
`uf2/pico_faces_<model>.uf2`.

## USB protocol

`I\n` → one line of model geometry and measured clocks.

`G <seed> [k_steps] [class] [w]\n` → `RFI2` header (w, h, channels, class)
+ raw image bytes + CRC32 + timing. `w` = guidance strength (4/6/8; absent
or unmatched = plain).

`V\n` (DVI builds) → `dvi bringup=<n> refresh=<mHz>`; `VR\n` restarts the
scanout first.

You can use the script in `viewer/view_serial.py` to communicate with the device.
