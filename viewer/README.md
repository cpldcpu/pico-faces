# viewer/

PC-side serial client for a flashed Pico 2.

```
python viewer/view_serial.py --port COM10 --seed 1 \
    --expect checkpoints/m3_decD_deep_full/goldens/golden_1.rgb --show
```

Sends `G <seed> [k] [class] [w]`, receives the `RFI2` reply (header + raw
image + CRC32 + per-stage timings), prints the device-reported CRC, saves
the frame, and — with `--expect <file>` — byte-compares the image against a
released golden, which proves the board reproduces the build bit-for-bit
(expected CRCs: `m3_decD_deep_full` → `40c5e5a0`, `m3_long_cfg` →
`b32e6e63`, both at seed 1 / K=4 / class 1 / w=4).

Flags: `--steps` (Euler steps: 8 native, 4/2/1 strided; goldens use 4),
`--cls` (0–3 = gender × smile, else unconditional), `--w` (guidance 4/6/8;
omit for the golden convention `w_idx = seed % (n_w+1) - 1`), `--out`
(save directory), `--show` (display window).

Needs `pyserial`, `numpy`, `matplotlib`, `pillow` (all in
[requirements.txt](../requirements.txt)).
