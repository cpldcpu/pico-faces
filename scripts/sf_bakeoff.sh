#!/usr/bin/env bash
# Self-Flow bake-off (doc/self_flow_plan.md): three twin 80k runs at the
# fast model's depth-8 arch — baseline vs Dual-Timestep vs full Self-Flow.
# Same seed, same latents; compare fixed-seed eval grids + val v-loss.
#
# Needs the training latents at artifacts/m3_long_cfg/runs/vae/latents.npz
# (regenerate via train_vae.py + latents.py --model m3_long_cfg, or copy
# them from an existing run). Run in tmux:
#   tmux new-session -d -s sf 'bash scripts/sf_bakeoff.sh > sf_bakeoff.log 2>&1'
set -e
cd "$(dirname "$0")/.."
PY="${PY:-python3}"

LAT=artifacts/m3_long_cfg/runs/vae/latents.npz
[ -f "$LAT" ] || { echo "missing $LAT — train the m3_long_cfg VAE + latents first"; exit 1; }

for M in sf0_base sf0_dualt sf1_selfflow; do
  echo "=== $M  $(date) ==="
  mkdir -p "artifacts/$M/runs"
  $PY train/dit/train_dit.py --model "$M" 2>&1 | tee "artifacts/$M/runs/train.log"
done
echo SF-BAKEOFF-DONE
