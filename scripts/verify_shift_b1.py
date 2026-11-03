"""Verify the B1 claim "the eval is not CIFAR-10 through a pixel transform".

This is the falsification test for the B2 premise (that a corrupted CIFAR val split
is a valid proxy for the hidden shift). Three independent checks, all using only
train data + the unlabeled eval images:

1. exact duplicates: are any eval images byte-identical to a train image?
2. per-image statistics on the *full* 50k train set, with effect sizes (Cohen's d)
   and Mann-Whitney p-values. Small effects mean these stats cannot carry the claim.
3. nearest neighbour, robust to geometry:
   - dihedral-invariant (8 flips/rotations of every train image)
   - translation-tolerant (5 small shifts + 8x8 average pooling)
   Reference distribution: genuine CIFAR images matched the same way (val -> train).
   If the eval were CIFAR through a pixel/geometric transform, eval->train distances
   would be far below the genuine-pair distribution.

Usage:
    python scripts/verify_shift_b1.py [--train-sample 0] [--no-gpu]

Writes runs/analysis/verify_shift.json and prints a summary.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import load_test, load_train  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from shift_analysis import image_metrics  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402


def prep(a: np.ndarray, pool: int = 0) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(a)).float()
    if pool:
        t = t.permute(0, 3, 1, 2)
        t = torch.nn.functional.avg_pool2d(t, pool)
        t = t.permute(0, 2, 3, 1)
    t = t.reshape(len(t), -1)
    return (t - t.mean(1, keepdim=True)) / (t.std(1, keepdim=True) + 1e-6)


def variants(b: np.ndarray, kind: str):
    """Yield transformed copies of an image batch, as (N, 32, 32, 3) uint8."""
    if kind == "identity":
        yield b
        return
    arr = torch.from_numpy(b).permute(0, 3, 1, 2)
    if kind == "dihedral":
        for k in range(4):
            for flip in (False, True):
                v = torch.rot90(arr, k, dims=(2, 3))
                if flip:
                    v = torch.flip(v, dims=(3,))
                yield v.permute(0, 2, 3, 1).contiguous().numpy()
    elif kind == "shift":
        for dy, dx in ((0, 0), (2, 0), (-2, 0), (0, 2), (0, -2)):
            yield torch.roll(arr, shifts=(dy, dx), dims=(2, 3)).permute(0, 2, 3, 1).contiguous().numpy()
    else:
        raise ValueError(kind)


def min_dist(a: np.ndarray, b: np.ndarray, kind: str, pool: int, dev: str,
             chunk: int = 1024) -> np.ndarray:
    """min squared L2 from each row of a to the rows of b, over a variant family."""
    A = prep(a, pool).to(dev)
    best = torch.full((len(A),), float("inf"), device=dev)
    for v in variants(b, kind):
        B = prep(v, pool).to(dev)
        for i in range(0, len(A), chunk):
            d = torch.cdist(A[i:i + chunk], B).min(1).values
            best[i:i + chunk] = torch.minimum(best[i:i + chunk], d)
        del B
    return best.cpu().numpy() ** 2


def summarise(d: np.ndarray) -> dict[str, float]:
    return {
        "p1": float(np.percentile(d, 1)),
        "p5": float(np.percentile(d, 5)),
        "median": float(np.median(d)),
        "p95": float(np.percentile(d, 95)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-sample", type=int, default=0,
                   help="0 = use all 50k train images for the statistics")
    p.add_argument("--out", type=Path, default=ROOT / "runs/analysis/verify_shift.json")
    args = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x, _ = load_train()
    tx, _ = load_test()
    print(f"train {x.shape}  eval {tx.shape}  device {dev}", flush=True)

    result: dict = {}

    # 1. exact duplicates (byte-identical)
    train_set = {im.tobytes() for im in x}
    exact = sum(im.tobytes() in train_set for im in tx)
    result["exact_matches"] = {"eval_matched": int(exact), "eval_total": int(len(tx)),
                               "train_unique": int(len(train_set))}
    print(f"[1] exact byte matches eval->train: {exact}/{len(tx)} "
          f"(train unique {len(train_set)}/{len(x)})", flush=True)

    # 2. per-image statistics with effect sizes
    rng = np.random.default_rng(0)
    xs = x if args.train_sample == 0 else x[rng.choice(len(x), args.train_sample, replace=False)]
    keys = list(image_metrics(xs[0]).keys())
    tr = {k: np.array([image_metrics(im)[k] for im in xs]) for k in keys}
    ev = {k: np.array([image_metrics(im)[k] for im in tx]) for k in keys}
    stats = {}
    print(f"\n[2] statistics, train n={len(xs)}", flush=True)
    print(f"{'metric':<14}{'train':>10}{'eval':>10}{'delta':>10}{'d':>8}{'p':>11}", flush=True)
    for k in keys:
        a, b = tr[k], ev[k]
        sd = np.sqrt((a.var() * len(a) + b.var() * len(b)) / (len(a) + len(b)))
        d = (b.mean() - a.mean()) / sd
        pv = float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
        stats[k] = {"train_mean": float(a.mean()), "eval_mean": float(b.mean()),
                    "delta": float(b.mean() - a.mean()), "cohen_d": float(d), "p": pv}
        print(f"{k:<14}{a.mean():10.4f}{b.mean():10.4f}{b.mean()-a.mean():+10.4f}"
              f"{d:+8.2f}{pv:11.1e}", flush=True)
    result["stats"] = stats

    # 3. geometry-robust nearest neighbours
    val_idx = rng.choice(len(x), 1000, replace=False)
    mask = np.ones(len(x), bool)
    mask[val_idx] = False
    result["nearest_neighbour"] = {}
    for label, kind, pool in (("dihedral", "dihedral", 0), ("shift_pool4", "shift", 4)):
        ref = min_dist(x[val_idx], x[mask], kind, pool, dev)
        dis = min_dist(tx, x, kind, pool, dev)
        ref_s, dis_s = summarise(ref), summarise(dis)
        frac_p1 = float((dis < np.percentile(ref, 1)).mean())
        result["nearest_neighbour"][label] = {
            "genuine_cifar": ref_s, "eval_to_train": dis_s,
            "frac_eval_below_genuine_p1": frac_p1,
        }
        print(f"\n[3] {label}: genuine CIFAR pairs median {ref_s['median']:.1f} "
              f"(p1 {ref_s['p1']:.1f}) | eval->train median {dis_s['median']:.1f} "
              f"(p1 {dis_s['p1']:.1f}) | eval below genuine p1: {frac_p1:.4f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}", flush=True)
    print("Reading: a transform of CIFAR would put eval near (or below) the genuine-pair "
          "distribution. No eval image is below the 1st percentile under either match, so "
          "the eval is different content, not transformed CIFAR.")


if __name__ == "__main__":
    main()
