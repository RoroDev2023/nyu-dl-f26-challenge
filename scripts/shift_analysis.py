"""B1/B2 shift analysis for the shared repo (seed-0 5k split).

B1: compare CIFAR-10 train vs competition-test image statistics and save a montage.
B2: corrupt the honest 5k validation split and measure accuracy drops for checkpoints.

Usage:
    python scripts/shift_analysis.py
    python scripts/shift_analysis.py --checkpoints runs/B_candidate/last.pt ...
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter
from scipy import ndimage
from torch.utils.data import DataLoader, TensorDataset

from data import eval_tf, load_test, load_train, train_val_split
from models import count_params, load_model
from train import get_device

CORRUPTIONS = [
    "noise_0.05",
    "noise_0.10",
    "blur_1.0",
    "blur_2.0",
    "jpeg_30",
    "contrast_0.7",
    "contrast_1.3",
    "color_1.3",
    "combo",
]


def image_metrics(arr: np.ndarray) -> dict[str, float]:
    x = arr.astype(np.float32) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b

    lap = ndimage.laplace(gray)
    sharpness = float(lap.var())
    noise = float(1.4826 * np.median(np.abs(lap - np.median(lap))))

    dv = np.abs(np.diff(gray, axis=1))
    dh = np.abs(np.diff(gray, axis=0))
    v_bound = dv[:, 7::8].mean()
    v_interior = np.delete(dv, np.s_[7::8], axis=1).mean()
    h_bound = dh[7::8, :].mean()
    h_interior = np.delete(dh, np.s_[7::8], axis=0).mean()
    blockiness = float(0.5 * ((v_bound - v_interior) + (h_bound - h_interior)))

    spec = np.fft.fftshift(np.abs(np.fft.fft2(gray)))
    hh, ww = gray.shape
    yy, xx = np.ogrid[:hh, :ww]
    radius = np.sqrt((yy - hh // 2) ** 2 + (xx - ww // 2) ** 2)
    hf_ratio = float(spec[radius > 0.25 * min(hh, ww)].sum() / (spec.sum() + 1e-12))

    rg, yb = r - g, 0.5 * (r + g) - b
    colorfulness = float(
        np.sqrt(rg.std() ** 2 + yb.std() ** 2)
        + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    )

    return {
        "mean_r": float(r.mean()), "mean_g": float(g.mean()), "mean_b": float(b.mean()),
        "std_r": float(r.std()), "std_g": float(g.std()), "std_b": float(b.std()),
        "brightness": float(gray.mean()), "contrast": float(gray.std()),
        "saturation": float((x.max(-1) - x.min(-1)).mean()),
        "sharpness": sharpness, "noise": noise, "blockiness": blockiness,
        "hf_ratio": hf_ratio, "colorfulness": colorfulness,
    }


def metric_table(images, name: str):
    rows = [image_metrics(np.asarray(img.convert("RGB"))) for img in images]
    df = pd.DataFrame(rows)
    return pd.DataFrame({"mean": df.mean(), "std": df.std()}).rename(
        columns=lambda c: f"{name}_{c}"
    )


def save_montage(train_x, test_x, path: Path, n: int = 8) -> None:
    fig, axes = plt.subplots(2, n, figsize=(1.35 * n, 3.2))
    rng = np.random.default_rng(0)
    for row, (arr, title) in enumerate(
        ((train_x, "CIFAR-10 train"), (test_x, "competition eval"))
    ):
        picks = rng.choice(len(arr), size=n, replace=False)
        for col, idx in enumerate(picks):
            ax = axes[row, col]
            ax.imshow(Image.fromarray(arr[idx]).resize((96, 96), Image.BILINEAR))
            ax.axis("off")
            if col == 0:
                ax.set_title(title, loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def corrupt(img: Image.Image, kind: str) -> Image.Image:
    arr = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
    rng = np.random.default_rng(zlib.crc32(kind.encode()))
    if kind.startswith("noise"):
        sigma = float(kind.split("_")[1])
        return Image.fromarray((np.clip(arr + rng.normal(0, sigma, arr.shape), 0, 1) * 255).astype(np.uint8))
    if kind.startswith("blur"):
        return img.filter(ImageFilter.GaussianBlur(float(kind.split("_")[1])))
    if kind.startswith("jpeg"):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(kind.split("_")[1]))
        return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    if kind.startswith("contrast"):
        return ImageEnhance.Contrast(img).enhance(float(kind.split("_")[1]))
    if kind.startswith("color"):
        return ImageEnhance.Color(img).enhance(float(kind.split("_")[1]))
    if kind == "combo":
        out = ImageEnhance.Contrast(img).enhance(1.15)
        out = ImageEnhance.Brightness(out).enhance(1.05)
        out = out.filter(ImageFilter.GaussianBlur(0.5))
        a = np.asarray(out).astype(np.float32) / 255.0
        a = np.clip(a + rng.normal(0, 0.02, a.shape), 0, 1)
        out = Image.fromarray((a * 255).astype(np.uint8))
        buf = io.BytesIO()
        out.save(buf, format="JPEG", quality=75)
        return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    raise ValueError(kind)


@torch.no_grad()
def accuracy(model, tensors: torch.Tensor, labels: torch.Tensor, device) -> float:
    loader = DataLoader(TensorDataset(tensors, labels), batch_size=256, shuffle=False)
    correct = total = 0
    for xb, yb in loader:
        xb = xb.to(device)
        correct += (model(xb).argmax(1).cpu() == yb).sum().item()
        total += len(yb)
    return correct / total


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoints", type=Path, nargs="*", default=[])
    p.add_argument("--out-dir", type=Path, default=ROOT / "runs/analysis")
    p.add_argument("--train-samples", type=int, default=2000)
    p.add_argument("--tta", action="store_true")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    x, y = load_train()
    test_x, _ = load_test()
    rng = np.random.default_rng(0)
    train_idx = rng.choice(len(x), size=min(args.train_samples, len(x)), replace=False)
    train_imgs = [Image.fromarray(im) for im in x[train_idx]]
    test_imgs = [Image.fromarray(im) for im in test_x]

    train_stats = metric_table(train_imgs, "train")
    test_stats = metric_table(test_imgs, "test")
    stats = pd.concat([train_stats, test_stats], axis=1)
    stats["delta"] = stats["test_mean"] - stats["train_mean"]

    report = ["# Shift analysis (seed-0 split)", ""]
    report += [
        "## B1. Eval vs CIFAR-10 train", "",
        f"- train sample: {len(train_imgs)}; eval: {len(test_imgs)}", "",
        "| metric | CIFAR train | competition eval | delta |",
        "|---|---|---|---|",
    ]
    for key in [
        "mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b",
        "brightness", "contrast", "saturation", "sharpness", "noise",
        "blockiness", "hf_ratio", "colorfulness",
    ]:
        report.append(
            f"| {key} | {stats.loc[key, 'train_mean']:.4f} "
            f"| {stats.loc[key, 'test_mean']:.4f} | {stats.loc[key, 'delta']:+.4f} |"
        )
    save_montage(x[train_idx], test_x, args.out_dir / "B_shift_montage.png")
    report += ["", f"Montage: `{args.out_dir / 'B_shift_montage.png'}`", ""]

    # B2 on the fixed seed-0 5k split
    xtr, ytr, xva, yva = train_val_split(x, y, val_size=5000)
    labels = torch.tensor(yva, dtype=torch.long)
    report += [
        "## B2. Corrupted seed-0 holdout", "",
        f"- val images: {len(xva)} (train {len(xtr)})", "",
    ]

    models = []
    device = get_device()
    for path in args.checkpoints:
        if not path.exists():
            report.append(f"Missing checkpoint `{path}`; skipped.")
            continue
        model, ckpt = load_model(path, use_ema=True)
        n = count_params(model)
        models.append((path.parent.name, model.to(device).eval(), n, ckpt))
    if not models:
        report += [
            "No checkpoints supplied; B2 skipped. Train a `B_...` run first, then "
            "re-run with `--checkpoints runs/B_.../last.pt`.",
            "",
        ]
    else:
        clean = torch.stack([eval_tf(im) for im in xva])
        corrupted = {
            name: torch.stack([eval_tf(corrupt(Image.fromarray(im), name)) for im in xva])
            for name in CORRUPTIONS
        }
        results = {}
        for name, model, n_params, ckpt in models:
            row = {"clean": accuracy(model, clean, labels, device)}
            for cname, tensors in corrupted.items():
                row[cname] = accuracy(model, tensors, labels, device)
            results[name] = row
            print(name, n_params, row, flush=True)

        names = list(results)
        clean_row = {n: results[n]["clean"] for n in names}
        report += [
            "| corruption | " + " | ".join(names) + " | mean drop |",
            "|---|" + "|".join(["---"] * (len(names) + 1)) + "|",
        ]
        for cname in ["clean"] + CORRUPTIONS:
            vals = [results[n][cname] for n in names]
            drops = [clean_row[n] - results[n][cname] for n in names]
            mean_drop = float(np.mean(drops)) if cname != "clean" else 0.0
            report.append(
                f"| {cname} | " + " | ".join(f"{v:.4f}" for v in vals)
                + f" | {mean_drop:+.4f} |"
            )
        report += ["", "No TTA; drops are relative to clean accuracy.", ""]
        (args.out_dir / "B_shift.json").write_text(json.dumps(results, indent=2))

    report += [
        "## Interpretation", "",
        "The eval images are new natural photographs (different framing, background,",
        "lighting, scale), not corrupted CIFAR-10. Synthetic corruptions are stress",
        "tests, not the shift. Prioritize natural-photo robustness (AugMix/JSD,",
        "RandAugment, CutMix, scale/translation, color/lighting), not noise/blur/JPEG.",
        "",
    ]
    (args.out_dir / "B_shift.md").write_text("\n".join(report))
    print("\n".join(report))


if __name__ == "__main__":
    main()
