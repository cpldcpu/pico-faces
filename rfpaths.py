"""Per-model path resolution.

A "model" is a directory models/<name>/ holding vae.yaml, dit.yaml and
export.yaml. Released inputs (checkpoints, calibrations, folded blobs,
goldens) live under checkpoints/<name>/ (git-tracked); everything generated
by training or re-folding lands under artifacts/<name>/ (gitignored). Paths
inside the yamls are repo-root-relative, so the yaml is the single source of
truth and scripts only need the model name.
"""
import os

import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))


def model_dir(model):
    d = os.path.join(ROOT, "models", model)
    if not os.path.isdir(d):
        known = sorted(os.listdir(os.path.join(ROOT, "models")))
        raise SystemExit(f"unknown model '{model}' — expected one of {known}")
    return d


def cfg(model, which):
    """Load models/<model>/<which>.yaml  (which: vae | dit | export)."""
    return yaml.safe_load(open(os.path.join(model_dir(model), which + ".yaml")))


def art(model, *parts):
    return os.path.join(ROOT, "artifacts", model, *parts)


def resolve(p):
    """Root-relative path -> absolute."""
    return p if os.path.isabs(p) else os.path.join(ROOT, p)
