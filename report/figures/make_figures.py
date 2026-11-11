"""Build every figure in the report from the archived logs/JSON in report/data.

Run from the repo root:
    ~/miniforge3/envs/ml/bin/python report/figures/make_figures.py

The report is self-contained: it only reads report/data and
report/figures/source.  Optional per-epoch logs (Track A / failed
continuations) are picked up automatically when present.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]          # report/
DATA = ROOT / "data"
LOGS = DATA / "logs"
SRC = ROOT / "figures" / "source"
OUT = ROOT / "figures"

# ---------------------------------------------------------------- style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "lines.linewidth": 1.2,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})

C_TRIV = "#3B6EA5"     # control / TrivialAug
C_AUG = "#D55E00"      # AugMix seed 0
C_AUG1 = "#E69F00"     # AugMix seed 1
C_GREY = "#6E6E6E"
C_RED = "#B2182B"


def load_log(name: str) -> pd.DataFrame | None:
    p = LOGS / name
    return pd.read_csv(p) if p.exists() else None


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- fig 1: montage
def montage() -> None:
    from PIL import Image

    im = Image.open(SRC / "shift_montage.png").convert("RGB")
    a = np.asarray(im)
    top = a[47:224, 23:1598]
    bot = a[267:445, 23:1598]

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 1.85))
    for ax, img, label in zip(
        axes, [top, bot],
        ["(a) CIFAR-10 training images (reference)",
         "(b) released unlabeled sample (not the final evaluation set)"]):
        ax.imshow(img)
        ax.set_axis_off()
        ax.text(0.0, 1.02, label, transform=ax.transAxes,
                ha="left", va="bottom", fontsize=7)
    fig.subplots_adjust(hspace=0.18)
    save(fig, "fig_shift_montage")


# ---------------------------------------------------------------- fig 2: curves
def curves() -> None:
    triv = load_log("B_trivial_300_sam.csv")
    aug1 = load_log("B_augmix_300_sam_s1.csv")
    kdt = load_log("B_kd3t_trivial.csv")
    kda = load_log("B_kd3t_augmix.csv")

    ncols = 2
    fig, axes = plt.subplots(1, ncols, figsize=(7.0, 2.05))

    # (a) B4 clean validation curves
    ax = axes[0]
    if triv is not None:
        ax.plot(triv.epoch, 100 * triv.val_acc, color=C_TRIV, lw=0.6,
                alpha=0.55, ls="--")
        ax.plot(triv.epoch, 100 * triv.val_acc_ema, color=C_TRIV, lw=1.4,
                label="TrivialAug, EMA (s0)")
    if aug1 is not None:
        ax.plot(aug1.epoch, 100 * aug1.val_acc, color=C_AUG1, lw=0.6,
                alpha=0.55, ls="--")
        ax.plot(aug1.epoch, 100 * aug1.val_acc_ema, color=C_AUG1, lw=1.4,
                label="AugMix, EMA (s1)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("clean validation accuracy (%)")
    ax.set_xlim(1, 300)
    ax.set_ylim(25, 99)
    ax.set_title("(a) B4: augmentation at 300 epochs", loc="left")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="y", lw=0.3, alpha=0.5)

    # (b) B5 KD arms (partial logs)
    ax = axes[1]
    if kdt is not None:
        ax.plot(kdt.epoch, 100 * kdt.val_acc_ema, color=C_TRIV, lw=1.4,
                label="KD + TrivialAug (s0)")
    if kda is not None:
        ax.plot(kda.epoch, 100 * kda.val_acc_ema, color=C_AUG, lw=1.4,
                label="KD + AugMix (s0)")
    if kdt is not None and kda is not None:
        cut = int(min(kdt.epoch.max(), kda.epoch.max()))
        ax.axvline(cut, color=C_GREY, lw=0.6, ls=":")
        ax.text(cut + 3, 26.5, f"common epoch {cut}", fontsize=6,
                color=C_GREY)
    ax.set_xlabel("epoch")
    ax.set_ylabel("clean validation accuracy, EMA (%)")
    ax.set_xlim(1, 180)
    ax.set_ylim(25, 99)
    ax.set_title("(b) B5: KD arms (provisional, 300 planned)", loc="left")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="y", lw=0.3, alpha=0.5)

    fig.tight_layout(w_pad=1.4)
    save(fig, "fig_val_curves")


# ---------------------------------------------------------------- fig 3: robustness
def robustness() -> None:
    d = json.loads((DATA / "shift_B4_seeds.json").read_text())
    models = [
        ("TrivialAug (s0)", "B_trivial_300_sam", C_TRIV),
        ("AugMix (s0)", "B_augmix_300_sam", C_AUG),
        ("AugMix (s1)", "B_augmix_300_sam_s1", C_AUG1),
    ]
    conditions = [
        ("clean", "clean"),
        ("combo", "combo"),
        ("noise 0.05", "noise_0.05"),
        ("noise 0.10", "noise_0.10"),
        ("JPEG 30", "jpeg_30"),
    ]
    x = np.arange(len(conditions))
    w = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(7.0, 2.15))
    for i, (label, key, color) in enumerate(models):
        vals = [100 * d[key][k] for _, k in conditions]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=label, color=color,
                      edgecolor="white", linewidth=0.4)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=5.5, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([c for c, _ in conditions])
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.legend(ncol=3, loc="upper right", frameon=False, bbox_to_anchor=(1.0, 1.02))
    ax.grid(axis="y", lw=0.3, alpha=0.5)
    # deltas on the combo group (seed 0 and seed 1)
    ax.text(1.0, 101, "+2.2 / +1.8 vs control", ha="center", fontsize=6,
            color=C_RED)
    fig.tight_layout()
    save(fig, "fig_robustness")


# ---------------------------------------------------------------- fig 4: B3 pilot
def pilot() -> None:
    d = json.loads((DATA / "shift_B3.json").read_text())
    # "extra affine+color" is not in the archived JSON (B_pilot_extra was
    # logged in STATUS.md only): clean TTA 95.46, combo 84.38.
    pts = {k: (100 * v["clean"], 100 * v["combo"]) for k, v in d.items()}
    pts["B_pilot_extra"] = (95.46, 84.38)
    rows = [
        ("TrivialAug", "B_pilot_trivial", C_GREY),
        ("RandAugment", "B_pilot_randaug", C_RED),
        ("AutoAugment-CIFAR", "B_pilot_autoaug", C_RED),
        ("geometry only", "B_pilot_geo", C_RED),
        ("extra affine+color", "B_pilot_extra", C_RED),
        ("AugMix", "B_pilot_augmix", C_AUG),
    ]
    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    offsets = {
        "TrivialAug": (7, 4, "left"),
        "RandAugment": (-7, 5, "right"),
        "AutoAugment-CIFAR": (-7, -11, "right"),
        "geometry only": (7, 5, "left"),
        "extra affine+color": (7, 4, "left"),
        "AugMix": (7, 4, "left"),
    }
    for label, key, color in rows:
        x, y = pts[key]
        ax.scatter(x, y, s=28, color=color, zorder=3, edgecolor="white",
                   linewidth=0.4)
        dx, dy, ha = offsets[label]
        ax.annotate(label, (x, y), xytext=(dx, dy),
                    textcoords="offset points", fontsize=6,
                    ha=ha, va="bottom")
    ax.set_xlabel("clean validation accuracy, TTA (%)")
    ax.set_ylabel("combo proxy accuracy (%)")
    ax.set_xlim(95.2, 96.9)
    ax.set_ylim(82.6, 88.2)
    ax.grid(lw=0.3, alpha=0.5)
    ax.set_title("150 epochs, no SAM, no KD", loc="left", fontsize=7)
    fig.tight_layout()
    save(fig, "fig_b3_pilot")


# ---------------------------------------------------------------- optional: failed curves
def failed_curves() -> None:
    names = {
        "C60_sam": ("SAM control", C_GREY),
        "C60_asam": ("ASAM", C_RED),
        "C60_swa": ("SWA", "#7B3294"),
        "C60_lskd": ("LS-KD", "#008837"),
        "C60_a045": ("KD $\\alpha$=0.45", "#E69F00"),
    }
    found = [(n, lab, c) for n, (lab, c) in names.items()
             if (LOGS / f"{n}.csv").exists()]
    if not found:
        return
    fig, ax = plt.subplots(figsize=(3.45, 2.3))
    for n, lab, c in found:
        d = pd.read_csv(LOGS / f"{n}.csv")
        ax.plot(d.epoch, 100 * d.val_acc_ema, lw=1.3, color=c, label=lab)
    ax.set_xlabel("fine-tune epoch")
    ax.set_ylabel("clean val accuracy, EMA (%)")
    ax.legend(frameon=False, fontsize=6)
    ax.grid(axis="y", lw=0.3, alpha=0.5)
    ax.set_title("60-epoch continuations from A4", loc="left", fontsize=7)
    fig.tight_layout()
    save(fig, "fig_failed_curves")


# ---------------------------------------------------------------- optional: Track A curves
def track_a_curves() -> None:
    names = {
        "A4_r3_80_600": ("A4, 600 ep", C_TRIV),
        "V2_r3_80_1200": ("V2, 1200 ep", "#7B3294"),
        "kd_r3_64_2t_600": ("KD 2 teachers", C_AUG),
    }
    found = [(n, lab, c) for n, (lab, c) in names.items()
             if (LOGS / f"{n}.csv").exists()]
    if not found:
        return
    fig, ax = plt.subplots(figsize=(3.45, 2.3))
    for n, lab, c in found:
        d = pd.read_csv(LOGS / f"{n}.csv")
        ax.plot(d.epoch, 100 * d.val_acc_ema, lw=1.3, color=c, label=lab)
    ax.set_xlabel("epoch")
    ax.set_ylabel("clean val accuracy, EMA (%)")
    ax.legend(frameon=False, fontsize=6)
    ax.grid(axis="y", lw=0.3, alpha=0.5)
    ax.set_title("Track A: long schedules and KD", loc="left", fontsize=7)
    fig.tight_layout()
    save(fig, "fig_track_a_curves")


if __name__ == "__main__":
    montage()
    curves()
    robustness()
    pilot()
    failed_curves()
    track_a_curves()
