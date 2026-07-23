#!/usr/bin/env bash
# Byte-exact gate for a model: desktop C engine vs the numpy int-sim goldens.
# The engine is compiled against the model's generated rf_cfg.h.
# Usage: bash scripts/verify_model.sh <model>
set -e
cd "$(dirname "$0")/.."
NAME="${1:?usage: verify_model.sh <model>}"
BLOB="artifacts/$NAME/export/model.bin"
GOLD="artifacts/$NAME/goldens/e2e_trained"
CFGDIR="artifacts/$NAME/export"
[ -f "$BLOB" ] || { echo "missing $BLOB (run quant/fold.py first)"; exit 1; }
[ -f "$CFGDIR/rf_cfg.h" ] || { echo "missing $CFGDIR/rf_cfg.h"; exit 1; }

mkdir -p "build/$NAME/e2e"
gcc -O2 -Wall -Wextra -I"$CFGDIR" -Iengine/include \
    engine/desktop/main_golden.c engine/src/graph.c engine/src/dit.c \
    engine/src/vae_dec.c engine/src/hires.c engine/src/kernels_ref.c \
    engine/src/prng.c -o "build/$NAME/rf_golden"
# goldens are .gray (1ch) or .rgb (3ch); seed set = whatever fold emitted
# (3 for plain models, 4 for CFG models: 3 guided + 1 plain-pass)
EXT=gray
[ -f "$GOLD/golden_1.rgb" ] && EXT=rgb
SEEDS=$(ls "$GOLD"/golden_*."$EXT" | sed 's/.*golden_\([0-9]*\)\..*/\1/' | sort -n)
"./build/$NAME/rf_golden" "$BLOB" "build/$NAME/e2e" 4 $SEEDS

fail=0
for s in $SEEDS; do
  if cmp -s "$GOLD/golden_${s}.${EXT}" "build/$NAME/e2e/eng_${s}.${EXT}"; then
    echo "seed $s: BYTE-EXACT"
  else
    echo "seed $s: MISMATCH"
    fail=1
  fi
done
exit $fail
