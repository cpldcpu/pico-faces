"""PC-side viewer: request a generation over USB-CDC and display/save it.

Usage:  python viewer/view_serial.py --port COM5 --seed 1 [--show]
        python viewer/view_serial.py --port /dev/ttyACM0 --seed 1 --expect artifacts/m1_gray/goldens/e2e_trained/golden_1.gray

Speaks both protocols: "RFIM" (legacy gray) and "RFI2" (w,h,ch,class header;
gray or RGB). Class defaults on-device to seed % n_cond (golden convention);
override with --cls (m2_color: 0=cat, 1=dog, 2=wild, 3=unconditional).
"""
import argparse
import binascii
import os
import struct
import sys

import serial


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=4,
                    help="Euler steps: 8, 4, 2 or 1 (quality vs speed)")
    ap.add_argument("--cls", type=int, default=None,
                    help="class index (conditional models); default seed %% n_cond")
    ap.add_argument("--w", type=int, default=None,
                    help="guidance strength (CFG builds: 4/6/8; 0 = plain). "
                         "Requires --cls; omitted = golden convention")
    ap.add_argument("--out", default="out")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--expect", help=".gray/.rgb file to byte-compare against")
    args = ap.parse_args()

    s = serial.Serial(args.port, 115200, timeout=120)
    s.reset_input_buffer()
    cmd = f"G {args.seed} {args.steps}"
    if args.cls is not None:
        cmd += f" {args.cls}"
        if args.w is not None:
            cmd += f" {args.w}"
    s.write((cmd + "\n").encode())

    # sync on magic
    window = b""
    while window not in (b"RFIM", b"RFI2"):
        b = s.read(1)
        if not b:
            print("timeout waiting for image (generation may take a while)")
            sys.exit(1)
        window = (window + b)[-4:]

    if window == b"RFIM":  # legacy gray header
        seed, w, h = struct.unpack("<IHH", s.read(8))
        ch, cond = 1, 0
    else:
        seed, w, h, ch, cond = struct.unpack("<IHHHH", s.read(12))
    img = s.read(w * h * ch)
    crc, ms = struct.unpack("<II", s.read(8))
    local_crc = binascii.crc32(img) & 0xFFFFFFFF
    ok = "OK" if local_crc == crc else "CRC MISMATCH"
    print(f"seed {seed} cls {cond}: {w}x{h}x{ch}, {ms} ms, crc {crc:08x} ({ok})")

    os.makedirs(args.out, exist_ok=True)
    from PIL import Image

    im = Image.frombytes("L" if ch == 1 else "RGB", (w, h), img)
    path = os.path.join(args.out, f"seed_{seed}.png")
    im.save(path)
    print(f"saved {path}")

    if args.expect:
        want = open(args.expect, "rb").read()
        print("golden compare:", "BYTE-EXACT" if want == img else "MISMATCH")
    if args.show:
        im.resize((512, 512), Image.NEAREST).show()


if __name__ == "__main__":
    main()
