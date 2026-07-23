"""Build gender x smile class labels for FFHQ from the ffhq-features-dataset
(Microsoft Face API annotations, github.com/DCGM/ffhq-features-dataset).

Classes: 0 = female/no-smile, 1 = female/smile, 2 = male/no-smile,
3 = male/smile (smile threshold 0.5), 4 = no face detected / missing ->
trains the null (unconditional) table set, which exists anyway.

Writes data/ffhq_gs_labels.npy (uint8[70000], aligned with the sorted
NNNNN.png order of ffhq_rgb_128.npy). Run: python data/make_ffhq_labels.py
"""
import io
import json
import os
import sys
import urllib.request
import zipfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = "https://github.com/DCGM/ffhq-features-dataset/archive/refs/heads/master.zip"


def main():
    out = os.path.join(ROOT, "data", "ffhq_gs_labels.npy")
    if os.path.exists(out):
        print(f"{out} exists, skipping")
        return
    zp = os.path.join(ROOT, "data", "hf_cache", "ffhq_features.zip")
    if not os.path.exists(zp):
        print("downloading", URL)
        urllib.request.urlretrieve(URL, zp)
    zf = zipfile.ZipFile(zp)
    members = {os.path.basename(m)[:5]: m for m in zf.namelist()
               if m.endswith(".json") and os.path.basename(m)[:5].isdigit()}
    print(f"{len(members)} json files in archive")

    labels = np.full(70000, 4, np.uint8)
    n_missing = 0
    for i in range(70000):
        m = members.get(f"{i:05d}")
        if m is None:
            n_missing += 1
            continue
        try:
            j = json.load(io.TextIOWrapper(zf.open(m), encoding="utf-8"))
            fa = j[0]["faceAttributes"] if isinstance(j, list) else j["faceAttributes"]
            g = 0 if fa["gender"] == "female" else 2
            s = 1 if float(fa["smile"]) >= 0.5 else 0
            labels[i] = g + s
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
            n_missing += 1
    np.save(out, labels)
    names = ["female/no-smile", "female/smile", "male/no-smile", "male/smile",
             "null (no face data)"]
    for c, name in enumerate(names):
        print(f"  {c} {name:22s} {int((labels == c).sum())}")
    print(f"wrote {out}  (missing/unparsed: {n_missing})")


if __name__ == "__main__":
    sys.exit(main())
