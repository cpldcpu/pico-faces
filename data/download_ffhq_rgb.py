"""Extract FFHQ 128x128 thumbnails as RGB uint8 (N,128,128,3) for the
m2_faces control experiment (the gray variant was extracted for m1).
Reads the zip already in the HF cache. Run: python data/download_ffhq_rgb.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "data", "hf_cache"))

import numpy as np
from PIL import Image
from tqdm import tqdm


def main():
    import io
    import re
    import zipfile

    from huggingface_hub import hf_hub_download

    out_path = os.path.join(ROOT, "data", "ffhq_rgb_128.npy")
    if os.path.exists(out_path):
        print(f"{out_path} already exists, skipping. Delete it to re-run.")
        return

    zip_path = hf_hub_download(
        repo_id="nuwandaa/ffhq128", repo_type="dataset", filename="thumbnails128x128.zip"
    )
    zf = zipfile.ZipFile(zip_path)
    members = sorted(m for m in zf.namelist() if re.fullmatch(r".*/?\d{5}\.png", m))
    n = len(members)
    assert n == 70000, f"expected 70000 images, got {n}"

    arr = np.lib.format.open_memmap(
        out_path + ".tmp", mode="w+", dtype=np.uint8, shape=(n, 128, 128, 3)
    )
    for i, name in enumerate(tqdm(members, desc="to RGB")):
        img = Image.open(io.BytesIO(zf.read(name))).convert("RGB")
        if img.size != (128, 128):
            img = img.resize((128, 128), Image.LANCZOS)
        arr[i] = np.asarray(img, dtype=np.uint8)
    arr.flush()
    del arr
    os.replace(out_path + ".tmp", out_path)
    print(f"wrote {out_path}  ({os.path.getsize(out_path) / 1e9:.2f} GB)")


if __name__ == "__main__":
    sys.exit(main())
