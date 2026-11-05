"""A proxy closer to the actual shift than noise/blur/JPEG: photographic variation.

B1 says the eval is new photographs (different camera/lighting/framing), not degraded
CIFAR. So the closest label-free proxy we can build from train data is per-image
photographic jitter: brightness/contrast/saturation scaling, small zoom, small rotation.
That is exactly what AugMix manipulates, so this is the fair test of whether AugMix's
B4 win is an artefact of coding-noise corruptions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from scipy.stats import binomtest

ROOT = Path("/home/philip/uni/deep_learning/nyu-dl-f26-challenge")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from data import load_train, train_val_split  # noqa: E402
from models import load_model  # noqa: E402
from tta import combine, view_probs  # noqa: E402

RUNS = ["B_trivial_300_sam", "B_augmix_300_sam"]


def photo(im: Image.Image, rng: np.random.Generator, strong: bool) -> Image.Image:
    im = im.convert("RGB")
    if strong:
        b, c, s = rng.uniform(0.80, 1.20), rng.uniform(0.80, 1.20), rng.uniform(0.70, 1.30)
        zoom, rot = 1.15, rng.uniform(-6, 6)
    else:
        b, c, s = rng.uniform(0.92, 1.08), rng.uniform(0.92, 1.08), rng.uniform(0.88, 1.12)
        zoom, rot = 1.05, rng.uniform(-3, 3)
    im = ImageEnhance.Brightness(im).enhance(b)
    im = ImageEnhance.Contrast(im).enhance(c)
    im = ImageEnhance.Color(im).enhance(s)
    side = int(round(32 / zoom))
    off = (32 - side) // 2
    im = im.crop((off, off, off + side, off + side)).resize((32, 32), Image.BILINEAR)
    return im.rotate(rot, resample=Image.BILINEAR)


def build(xva: np.ndarray, strong: bool, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack([np.asarray(photo(Image.fromarray(im), rng, strong)) for im in xva])


def mcnemar(ok_a, ok_b):
    b = int((ok_a & ~ok_b).sum())
    c = int((~ok_a & ok_b).sum())
    p = 1.0 if b + c == 0 else binomtest(b, b + c, 0.5, alternative="two-sided").pvalue
    return b, c, p


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x, y = load_train()
    _a, _b, xva, yva = train_val_split(x, y, val_size=5000)
    labels = np.asarray(yva)
    conds = {"clean": xva,
             "photo_mild": build(xva, strong=False),
             "photo_strong": build(xva, strong=True)}

    ok = {}
    for run in RUNS:
        model, _ = load_model(ROOT / f"runs/{run}/last.pt")
        model = model.to(device).eval()
        for cname, arr in conds.items():
            probs = view_probs(model, arr, device)
            ok[(run, cname)] = combine(probs, "flip").argmax(1) == labels
            print(f"{run:22s} {cname:13s} flip {ok[(run, cname)].mean()*100:.2f}", flush=True)

    print("\n| condition | trivial | augmix | net | b/c | McNemar p |")
    print("|---|---|---|---|---|---|")
    for cname in conds:
        t, a = ok[(RUNS[0], cname)], ok[(RUNS[1], cname)]
        b, c, p = mcnemar(a, t)
        print(f"| {cname} | {t.mean()*100:.2f} | {a.mean()*100:.2f} "
              f"| {(a.mean()-t.mean())*100:+.2f} | {b}/{c} | {p:.2e} |")


if __name__ == "__main__":
    main()
