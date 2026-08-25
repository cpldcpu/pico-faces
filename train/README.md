# train/

PyTorch training for the two released models (path B). Everything here runs
on the PC/GPU; nothing in this tree ships to the device — the deployable
subset of each network (small decoder, DiT blocks minus the conditioning
tower) is extracted at fold time.

| file | role |
|---|---|
| `vae/model.py` | asymmetric VAE: big encoder (train-only), small int8-friendly decoder (`dec_plan`-driven: Conv+BN+ReLU, NN-upsample) |
| `vae/train_vae.py` | VAE training (L1 + LPIPS + tiny KL) |
| `vae/train_decoder.py` | decoder-only retrain against a FROZEN encoder — how decoder D is built without disturbing the latent space |
| `latents.py` | precompute normalized fp16 latents (+ per-channel mean/std) for DiT training |
| `dit/model.py` | the backbones: DiT (RMSNorm, adaLN-zero, qk-RMSNorm, ReLU²) and a U-Net variant (bake-off loser, kept for reference) |
| `dit/train_dit.py` | rectified-flow training: v-prediction MSE, logit-normal t, EMA, optional cosine cooldown |
| `dit/sample.py` | Euler sampling, decoding, grid rendering |
| `common/` | EMA, sincos/timestep embeddings |

## Full retrain sequence

```
# 1. fast model: VAE + latents + DiT (300k)
bash scripts/train_model.sh m3_long_cfg

# 2. flagship decoder D: bigger decoder, SAME frozen encoder/latents
python train/vae/train_decoder.py --model m3_decD_deep_full \
    --enc-ckpt artifacts/m3_long_cfg/runs/vae/vae_final.pt

# 3. flagship DiT (depth 12, 500k + 100k cosine cooldown; uses the
#    latents from step 1 via models/m3_decD_deep_full/dit.yaml)
python train/dit/train_dit.py --model m3_decD_deep_full
```

Then continue with the int8 stages in [../quant/README.md](../quant/README.md)
(calibrate → QAT → fold). Rough wall-clock on an RTX 5090: VAE ~1 h, fast
DiT ~3 h, flagship DiT overnight.
