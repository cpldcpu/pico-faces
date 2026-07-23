#!/usr/bin/env bash
# Build the RP2350 UF2 (Linux / WSL, arm-none-eabi-gcc + cmake required).
# Usage: bash scripts/build_firmware.sh <model>          (e.g. m3_long_cfg)
#        bash scripts/build_firmware.sh <path/to/blob.bin>   (ad-hoc blob)
# Model builds publish the UF2 to uf2/pico_faces_<model>.uf2 (git-tracked).
# Set PICO_SDK_PATH / PICO_EXTRAS_PATH to your SDK locations (2.2.0 tested).
set -e
cd "$(dirname "$0")/.."
export PICO_SDK_PATH="${PICO_SDK_PATH:-/mnt/d/Pico/pico-sdk}"
export PICO_EXTRAS_PATH="${PICO_EXTRAS_PATH:-/mnt/d/Pico/pico-extras}"

ARG="${1:?usage: build_firmware.sh <model|model.bin>}"
if [ -d "models/$ARG" ]; then
  NAME="$ARG"
  BLOB="$(pwd)/artifacts/$NAME/export/model.bin"
  BUILD="build/$NAME/fw"
else
  NAME="custom"
  BLOB="$(realpath "$ARG")"
  BUILD="build/custom/fw"
fi
[ -f "$BLOB" ] || { echo "model blob not found: $BLOB"; exit 1; }

# picotool: a native Linux copy is installed ONCE at ~/.pico-tools (the WSL
# PATH's Windows picotool.exe cannot run here, and the old
# PICOTOOL_FORCE_FETCH_FROM_GIT=1 fallback re-fetches + rebuilds it for
# every fresh build dir). To (re)install:
#   cmake -S <picotool src> -B /tmp/ptb -DPICOTOOL_NO_LIBUSB=1 \
#     -DCMAKE_INSTALL_PREFIX=$HOME/.pico-tools -DCMAKE_BUILD_TYPE=Release
#   cmake --build /tmp/ptb -j && cmake --install /tmp/ptb
PICOTOOL_ARG="-Dpicotool_DIR=$HOME/.pico-tools/lib/cmake/picotool"
[ -f "$HOME/.pico-tools/lib/cmake/picotool/picotoolConfig.cmake" ] || \
  PICOTOOL_ARG="-DPICOTOOL_FORCE_FETCH_FROM_GIT=1"
cmake -S firmware -B "$BUILD" -G "Unix Makefiles" \
      -DMODEL_BIN="$BLOB" -DCMAKE_BUILD_TYPE=Release \
      "$PICOTOOL_ARG"
cmake --build "$BUILD" -j"$(nproc)"

if [ "$NAME" != "custom" ]; then
  cp "$BUILD/pico_faces.uf2" "uf2/pico_faces_$NAME.uf2"
  ls -la "uf2/pico_faces_$NAME.uf2"
else
  ls -la "$BUILD/pico_faces.uf2"
fi
