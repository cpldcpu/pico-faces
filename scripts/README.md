# scripts/

The drivers. All are run from anywhere (they `cd` to the repo root) and take
a model name from [models/](../models/). Python defaults to `python3`;
override with `PY=/path/to/python`.

| script | what it does |
|---|---|
| `finalize.sh <model>` | **path A in one command**: stage latent stats → `fold.py` (frozen calib) → `verify_model.sh` → compare the fresh blob against `checkpoints/<model>/model.bin` → `build_firmware.sh`. No GPU |
| `verify_model.sh <model>` | the byte-exact gate: compiles the desktop C engine against the model's generated `rf_cfg.h` and compares its output to the int-sim goldens, seed by seed |
| `build_firmware.sh <model\|blob.bin>` | cmake + build the RP2350 UF2 with the model's blob embedded; publishes `uf2/pico_faces_<model>.uf2`. Needs the Pico SDK (`PICO_SDK_PATH`) |
| `train_model.sh <model>` | path B base chain: VAE → latents → DiT (waits for the dataset first). The remaining stages are per-model — see [train/](../train/README.md) and [quant/](../quant/README.md) |

Typical flows:

```
# recreate + verify both released UF2s from checkpoints
bash scripts/finalize.sh m3_decD_deep_full
bash scripts/finalize.sh m3_long_cfg

# ad-hoc firmware from any blob
bash scripts/build_firmware.sh path/to/model.bin
```
