"""Paired tests behind B4 and the TTA result: are the deltas real?

Per-image correctness on the same 5000 val images, so the comparison is paired:
  - augmix vs trivial (the B4 claim)
  - 5crop vs flip TTA within each model
Reports discordant-pair counts, the net, and an exact McNemar (binomial) p-value.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.stats import binomtest

ROOT = Path("/home/philip/uni/deep_learning/nyu-dl-f26-challenge")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from data import load_train, train_val_split  # noqa: E402
from models import load_model  # noqa: E402
from shift_analysis import corrupt  # noqa: E402
from tta import combine, view_probs  # noqa: E402
from tta_shift import build_conditions  # noqa: E402

RUNS = ["B_trivial_300_sam", "B_augmix_300_sam"]
CONDS = ["clean", "combo", "jpeg_30", "noise_0.05", "shift3", "zoom_1.2"]


def mcnemar(ok_a: np.ndarray, ok_b: np.ndarray) -> tuple[int, int, float]:
    """b = a right & b wrong, c = b right & a wrong; exact two-sided p."""
    b = int((ok_a & ~ok_b).sum())
    c = int((~ok_a & ok_b).sum())
    p = 1.0 if b + c == 0 else binomtest(b, b + c, 0.5, alternative="two-sided").pvalue
    return b, c, p


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x, y = load_train()
    _a, _b, xva, yva = train_val_split(x, y, val_size=5000)
    y = np.asarray(yva)
    conds = build_conditions(xva)
    conds = {k: v for k, v in conds.items() if k in CONDS}

    ok = {}
    for run in RUNS:
        model, _ = load_model(ROOT / f"runs/{run}/last.pt")
        model = model.to(device).eval()
        ok[run] = {}
        for cname, arr in conds.items():
            probs = view_probs(model, arr, device)
            ok[run][cname] = {
                pol: combine(probs, pol).argmax(1) == y for pol in ("flip", "5crop")
            }

    print("## augmix vs trivial (paired, same images)\n")
    print("| condition | policy | trivial | augmix | net (fixes-intros) | b/c | McNemar p |")
    print("|---|---|---|---|---|---|---|")
    for cname in conds:
        for pol in ("flip", "5crop"):
            t, a = ok[RUNS[0]][cname][pol], ok[RUNS[1]][cname][pol]
            b, c, p = mcnemar(a, t)
            print(f"| {cname} | {pol} | {t.mean()*100:.2f} | {a.mean()*100:.2f} "
                  f"| {(a.mean()-t.mean())*100:+.2f} | {b}/{c} | {p:.2e} |")

    print("\n## 5crop vs flip TTA (paired, within model)\n")
    print("| run | condition | flip | 5crop | net | b/c | McNemar p |")
    print("|---|---|---|---|---|---|---|")
    for run in RUNS:
        for cname in conds:
            f, s = ok[run][cname]["flip"], ok[run][cname]["5crop"]
            b, c, p = mcnemar(s, f)
            print(f"| {run} | {cname} | {f.mean()*100:.2f} | {s.mean()*100:.2f} "
                  f"| {(s.mean()-f.mean())*100:+.2f} | {b}/{c} | {p:.2e} |")


if __name__ == "__main__":
    main()
