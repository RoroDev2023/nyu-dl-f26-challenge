"""Does he 5-crop TTA help on the *shift*, or only on clean val?

Uses partner's tta.py views (reflect-pad 4, 5 windows, x flip, softmax averaged) and
scores every policy on the seed-0 5k holdout, clean plus shifted conditions.
Adds two conditions that target the axis B1 says actually differs (framing/scale):
  shift3   - deterministic 3px translation, reflect padded
  zoom_1.2 - centre crop 27px -> resize 32px (1.2x zoom)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path("/home/philip/uni/deep_learning/nyu-dl-f26-challenge")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from data import load_train, train_val_split  # noqa: E402
from models import count_params, load_model  # noqa: E402
from shift_analysis import CORRUPTIONS, corrupt  # noqa: E402
from tta import POLICIES, combine, view_probs  # noqa: E402

RUNS = ["B_trivial_300_sam", "B_augmix_300_sam"]
POLS = ["flip", "5crop", "cf50", "cf80"]


def shift3(im: Image.Image, dy: int = 3, dx: int = 3) -> Image.Image:
    a = np.asarray(im.convert("RGB"))
    a = np.pad(a, ((4, 4), (4, 4), (0, 0)), mode="reflect")
    return Image.fromarray(a[4 + dy:36 + dy, 4 + dx:36 + dx])


def zoom(im: Image.Image, factor: float = 1.2) -> Image.Image:
    im = im.convert("RGB")
    side = int(round(32 / factor))
    off = (32 - side) // 2
    return im.crop((off, off, off + side, off + side)).resize((32, 32), Image.BILINEAR)


def build_conditions(xva: np.ndarray) -> dict[str, np.ndarray]:
    conds = {"clean": xva}
    for name in ["noise_0.05", "jpeg_30", "combo"]:
        assert name in CORRUPTIONS
        conds[name] = np.stack([np.asarray(corrupt(Image.fromarray(im), name)) for im in xva])
    conds["shift3"] = np.stack([np.asarray(shift3(Image.fromarray(im))) for im in xva])
    conds["zoom_1.2"] = np.stack([np.asarray(zoom(Image.fromarray(im))) for im in xva])
    return conds


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x, y = load_train()
    _xtr, _ytr, xva, yva = train_val_split(x, y, val_size=5000)
    labels = np.asarray(yva)
    print(f"val {xva.shape}  device {device}", flush=True)

    conds = build_conditions(xva)
    print("conditions:", ", ".join(conds), flush=True)

    for run in RUNS:
        ckpt = ROOT / f"runs/{run}/last.pt"
        model, _ = load_model(ckpt)
        model = model.to(device).eval()
        print(f"\n=== {run}  ({count_params(model):,} params, EMA) ===", flush=True)
        rows = {}
        for cname, arr in conds.items():
            probs = view_probs(model, arr, device)
            rows[cname] = {p: float((combine(probs, p).argmax(1) == labels).mean()) for p in POLS}
        header = f"{'condition':<11}" + "".join(f"{p:>9}" for p in POLS) + f"{'5crop-flip':>12}"
        print(header, flush=True)
        for cname, r in rows.items():
            print(f"{cname:<11}" + "".join(f"{r[p]*100:9.2f}" for p in POLS)
                  + f"{(r['5crop']-r['flip'])*100:+12.2f}", flush=True)
        best = max(rows, key=lambda c: rows[c]["5crop"])
        print(f"(best 5crop condition: {best})", flush=True)


if __name__ == "__main__":
    main()
