# data/

Dataset acquisition for a full retrain (path B). Not needed to recreate the
UF2s from the released checkpoints. All outputs are gitignored.

1. **`download_ffhq_rgb.py`** — FFHQ `thumbnails128x128` (NVlabs release,
   ~70k images) via the Hugging Face hub into `ffhq_rgb_128.npy`
   (uint8 `[70000, 128, 128, 3]`, ~3.4 GB, sorted `NNNNN.png` order).
2. **`make_ffhq_labels.py`** — gender × smile class labels from the
   [ffhq-features-dataset](https://github.com/DCGM/ffhq-features-dataset)
   (Microsoft Face API annotations): 0 f/neutral, 1 f/smile, 2 m/neutral,
   3 m/smile, 4 = no face detected (trains the null/unconditional tables).
   Writes `ffhq_gs_labels.npy`, aligned with the image order.

```
python data/download_ffhq_rgb.py
python data/make_ffhq_labels.py
```

FFHQ is CC-BY-NC-SA (the individual images carry their original licenses);
see the NVlabs FFHQ repository for terms.
