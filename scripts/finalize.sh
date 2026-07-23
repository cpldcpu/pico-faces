#!/usr/bin/env bash
# Recreate a model's flashable UF2 from the released checkpoints ("path A"):
#   stage latent stats -> fold (frozen calib) -> byte-exact desktop-engine
#   check -> compare against the released reference blob -> build the UF2.
# No GPU needed. Usage: bash scripts/finalize.sh <model>
# Python override: PY=/path/to/python bash scripts/finalize.sh <model>
set -e
cd "$(dirname "$0")/.."
NAME="${1:?usage: finalize.sh <model>}"
PY="${PY:-python3}"

# fold.py reads latent mean/std from <vae out_dir>/latents.npz; stage the
# released slim stats there unless a (re)trained latents file already exists.
VAEDIR=$($PY -c "import rfpaths; print(rfpaths.resolve(rfpaths.cfg('$NAME','vae')['out_dir']))")
mkdir -p "$VAEDIR"
[ -f "$VAEDIR/latents.npz" ] || cp "checkpoints/$NAME/latent_stats.npz" "$VAEDIR/latents.npz"

$PY quant/fold.py --model "$NAME"
bash scripts/verify_model.sh "$NAME"

if cmp -s "artifacts/$NAME/export/model.bin" "checkpoints/$NAME/model.bin"; then
  echo "model.bin: BYTE-IDENTICAL to the released reference"
else
  echo "model.bin: DIFFERS from checkpoints/$NAME/model.bin"
  echo "  (expected only after retraining or a deliberate quant/ change)"
fi

bash scripts/build_firmware.sh "$NAME"
echo FINALIZE-DONE
