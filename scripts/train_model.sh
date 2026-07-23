#!/usr/bin/env bash
# Base training chain for a model ("path B", GPU): VAE -> latents -> DiT.
# Waits for the dataset files first, so it can be chained behind a download.
# Long jobs are best run in tmux:
#   tmux new-session -d -s train \
#     'bash scripts/train_model.sh m3_long_cfg > train.log 2>&1'
# See models/*/export.yaml + quant/README.md for the post-training stages
# (decoder D, calibration, distillation-QAT) that finish each released model.
set -e
cd "$(dirname "$0")/.."
NAME="${1:?usage: train_model.sh <model>}"
PY="${PY:-python3}"

DATA=$($PY -c "import rfpaths; print(rfpaths.resolve(rfpaths.cfg('$NAME','vae')['data']))")
LATD=$($PY -c "import rfpaths; c=__import__('rfpaths').cfg('$NAME','vae'); print(__import__('rfpaths').resolve(c.get('latents_data', c['data'])))")
until [ -f "$DATA" ] && [ -f "$LATD" ]; do sleep 30; done
echo "=== data present $(date)"

$PY train/vae/train_vae.py --model "$NAME"
$PY train/latents.py --model "$NAME"
$PY train/dit/train_dit.py --model "$NAME"
echo TRAIN-DONE
