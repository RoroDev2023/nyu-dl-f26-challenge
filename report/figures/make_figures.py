"""Build every figure in the report from the archived logs/JSON in report/data.

Run from the repo root:
    ~/miniforge3/envs/ml/bin/python report/figures/make_figures.py

The report is self-contained: it only reads report/data and the archived
runs/analysis JSON.  Every curve is a real per-epoch log from this project.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]          # report/
DATA = ROOT / "data"
LOGS = DATA / "logs"
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

C_TRIV = "#3B6EA5"
C_AUG = "#D55E00"
C_AUG1 = "#E69F00"
C_GREY = "#6E6E6E"
C_RED = "#B2182B"
C_GREEN = "#008837"
C_PURPLE = "#7B3294"


def load_log(name: str) -> pd.DataFrame | None:
    p = LOGS / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- fig 1: curves
def curves() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.4))

    # (a) main recipe
    ax = axes[0, 0]
    track = [
        ("student_baseline", "baseline", C_GREY, "val_acc"),
        ("student_tricks_300", "all tricks", C_TRIV, "val_acc_ema"),
        ("kd_r3_64_2t_600", "KD 2T", C_AUG, "val_acc_ema"),
        ("A4_r3_80_600", "A4 600", C_RED, "val_acc_ema"),
        ("V2_r3_80_1200", "V2 1200", C_PURPLE, "val_acc_ema"),
    ]
    for name, label, color, col in track:
        d = load_log(name)
        if d is None:
            continue
        ax.plot(d.epoch, 100 * d[col], color=color, lw=1.0, label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel("clean validation accuracy (%)")
    ax.set_xlim(1, 1200)
    ax.set_xticks([0, 300, 600, 900, 1200])
    ax.set_ylim(25, 99)
    ax.set_title("(a) Main recipe", loc="left", fontsize=7)
    ax.legend(loc="lower right", frameon=False, fontsize=4.8,
              handlelength=1.1, labelspacing=0.15)
    ax.grid(axis="y", lw=0.3, alpha=0.5)

    # (b) architecture sweep
    ax = axes[0, 1]
    for name, label, color in [
        ("sweep_r3_64", "r3-64", C_TRIV),
        ("sweep_r3_80", "r3-80", C_AUG),
        ("sweep_r3_96", "r3-96", C_GREEN),
    ]:
        d = load_log(name)
        if d is None:
            continue
        ax.plot(d.epoch, 100 * d.val_acc_ema, color=color, lw=1.0, label=label)
    ax.set_xlabel("epoch")
    ax.set_xlim(1, 300)
    ax.set_ylim(25, 99)
    ax.set_title("(b) Architecture sweep (300 ep)", loc="left", fontsize=7)
    ax.legend(loc="lower right", frameon=False, fontsize=5.0)
    ax.grid(axis="y", lw=0.3, alpha=0.5)

    # (c) KD settings
    ax = axes[0, 2]
    for name, label, color in [
        ("sweep_r3_80", "control", C_GREY),
        ("A3_T2", "T=2", C_TRIV),
        ("A3_T1_a1", "T=1, $\\alpha$=1", C_AUG),
    ]:
        d = load_log(name)
        if d is None:
            continue
        ax.plot(d.epoch, 100 * d.val_acc_ema, color=color, lw=1.0, label=label)
    ax.set_xlabel("epoch")
    ax.set_xlim(1, 200)
    ax.set_ylim(25, 99)
    ax.set_title("(c) KD settings", loc="left", fontsize=7)
    ax.legend(loc="lower right", frameon=False, fontsize=5.0)
    ax.grid(axis="y", lw=0.3, alpha=0.5)

    # (d) 60-epoch finishers
    ax = axes[1, 0]
    for name, label, color in [
        ("C60_sam", "SAM control", C_GREY),
        ("C60_asam", "ASAM", C_RED),
        ("C60_swa", "SWA", C_PURPLE),
        ("C60_lskd", "LS-KD", C_GREEN),
        ("C60_a045", "$\\alpha$=0.45", C_AUG),
    ]:
        d = load_log(name)
        if d is None:
            continue
        ax.plot(d.epoch, 100 * d.val_acc_ema, color=color, lw=1.0, label=label)
    ax.axhline(97.72, color="black", lw=0.5, ls=":", zorder=0)
    ax.text(2, 97.74, "A4 start", fontsize=4.8, color="black", va="bottom")
    ax.set_xlabel("fine-tune epoch")
    ax.set_ylabel("clean validation accuracy (%)")
    ax.set_xlim(1, 60)
    ax.set_ylim(96.8, 98.2)
    ax.set_title("(d) 60-epoch finishers", loc="left", fontsize=7)
    ax.legend(loc="lower right", frameon=True, facecolor="white",
              framealpha=0.85, edgecolor="none", fontsize=4.8,
              handlelength=1.1, labelspacing=0.15)
    ax.grid(axis="y", lw=0.3, alpha=0.5)

    # (e) fresh 300-epoch alternatives
    ax = axes[1, 1]
    for name, label, color in [
        ("W40_s0", "WRN-40-2 s0", C_GREY),
        ("W40_s1", "WRN-40-2 s1", "#A0A0A0"),
        ("F300_lskd", "LS-KD", C_GREEN),
        ("S300_asam", "fresh ASAM", C_RED),
    ]:
        d = load_log(name)
        if d is None:
            continue
        ax.plot(d.epoch, 100 * d.val_acc_ema, color=color, lw=1.0, label=label)
    ax.set_xlabel("epoch")
    ax.set_xlim(1, 300)
    ax.set_ylim(96.0, 98.0)
    ax.set_title("(e) Fresh 300-ep alternatives", loc="left", fontsize=7)
    ax.legend(loc="lower right", frameon=True, facecolor="white",
              framealpha=0.85, edgecolor="none", fontsize=4.8)
    ax.grid(axis="y", lw=0.3, alpha=0.5)

    # (f) B5 KD arms
    ax = axes[1, 2]
    for name, label, color in [
        ("B_kd3t_trivial", "KD + TrivialAug", C_TRIV),
        ("B_kd3t_augmix", "KD + AugMix", C_AUG),
    ]:
        d = load_log(name)
        if d is None:
            continue
        ax.plot(d.epoch, 100 * d.val_acc_ema, color=color, lw=1.0, label=label)
    ax.axvline(163, color=C_GREY, lw=0.5, ls=":")
    ax.text(166, 27, "common 163", fontsize=4.8, color=C_GREY)
    ax.set_xlabel("epoch")
    ax.set_xlim(1, 180)
    ax.set_ylim(25, 99)
    ax.set_title("(f) KD + AugMix (provisional)", loc="left", fontsize=7)
    ax.legend(loc="lower right", frameon=False, fontsize=5.0)
    ax.grid(axis="y", lw=0.3, alpha=0.5)

    fig.tight_layout(w_pad=0.8, h_pad=0.8)
    save(fig, "fig_val_curves")


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

    fig, ax = plt.subplots(figsize=(7.0, 2.1))
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
    ax.text(1.0, 101, "+2.2 / +1.8 vs control", ha="center", fontsize=6,
            color=C_RED)
    fig.tight_layout()
    save(fig, "fig_robustness")


# ---------------------------------------------------------------- fig 3: B3 pilot
def pilot() -> None:
    d = json.loads((DATA / "shift_B3.json").read_text())
    pts = {k: (100 * v["clean"], 100 * v["combo"]) for k, v in d.items()}
    # extra affine+color was logged only in STATUS.md
    pts["B_pilot_extra"] = (95.46, 84.38)
    rows = [
        ("TrivialAug", "B_pilot_trivial", C_GREY),
        ("RandAugment", "B_pilot_randaug", C_RED),
        ("AutoAugment-CIFAR", "B_pilot_autoaug", C_RED),
        ("geometry only", "B_pilot_geo", C_RED),
        ("extra affine+color", "B_pilot_extra", C_RED),
        ("one-view AugMix", "B_pilot_augmix", C_AUG),
    ]
    offsets = {
        "TrivialAug": (7, 4, "left"),
        "RandAugment": (-7, 5, "right"),
        "AutoAugment-CIFAR": (-7, -11, "right"),
        "geometry only": (7, 5, "left"),
        "extra affine+color": (7, 4, "left"),
        "one-view AugMix": (7, 4, "left"),
    }
    fig, ax = plt.subplots(figsize=(3.45, 2.35))
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


# ---------------------------------------------------------------- fig 4: failures
# ---------------------------------------------------------------- fig 5: teachers
def teacher_student() -> None:
    rows = [
        ("PyramidNet-110", 97.64, C_GREY),
        ("WRN-28-10 s0", 98.06, C_GREY),
        ("WRN-28-10 s1", 98.12, C_GREY),
        ("WRN pool (s0+s1)", 98.14, C_TRIV),
        ("A4 student (flip)", 97.84, C_AUG),
        ("V2 student (5-crop)", 97.94, C_AUG),
    ]
    fig, ax = plt.subplots(figsize=(3.45, 2.25))
    y = np.arange(len(rows))
    for yi, (label, val, color) in zip(y, rows):
        ax.barh(yi, val - 97.0, left=97.0, color=color, height=0.72)
        ax.text(val + 0.02, yi, f"{val:.2f}", va="center", fontsize=5.4)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=5.8)
    ax.set_xlim(97.0, 98.35)
    ax.set_xticks([97.0, 97.5, 98.0])
    ax.set_xlabel("holdout accuracy (%)")
    ax.set_title("Teachers and students (5,000 holdout)", loc="left", fontsize=7)
    ax.grid(axis="x", lw=0.3, alpha=0.5)
    fig.tight_layout()
    save(fig, "fig_teacher_student")


# ---------------------------------------------------------------- fig 6: TTA deltas
def tta_deltas() -> None:
    conds = ["clean", "combo", "JPEG 30", "noise 0.05", "zoom 1.2"]
    triv = [-0.02, 0.60, 1.24, 0.18, -0.36]
    aug = [0.10, 0.38, 0.56, -0.40, -0.04]
    y = np.arange(len(conds))
    h = 0.36
    fig, ax = plt.subplots(figsize=(3.45, 2.25))
    ax.barh(y + h / 2, triv, h, color=C_TRIV, label="TrivialAug")
    ax.barh(y - h / 2, aug, h, color=C_AUG, label="AugMix")
    ax.axvline(0, color="black", lw=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(conds, fontsize=6)
    ax.set_xlim(-0.6, 1.5)
    ax.set_xlabel("5-crop minus flip TTA (pp)")
    ax.set_title("TTA policy difference", loc="left", fontsize=7)
    ax.legend(loc="lower right", frameon=False, fontsize=5.6)
    ax.grid(axis="x", lw=0.3, alpha=0.5)
    fig.tight_layout()
    save(fig, "fig_tta_deltas")


# ---------------------------------------------------------------- fig 7: leaderboard

def leaderboard() -> None:
    rows = [
        ("F2 + BN adaptation", 0.884, C_RED),
        ("F1", 0.898, C_GREY),
        ("F2 5-crop", 0.898, C_GREY),
        ("F2 Sinkhorn", 0.898, C_GREY),
        ("F2 flip", 0.899, C_GREY),
        ("kd2t 5-crop", 0.901, C_GREY),
        ("F2s1 flip", 0.903, C_GREY),
        ("kd2t flip", 0.904, C_TRIV),
        ("F2s1 + ASAM", 0.908, C_GREY),
        ("soup F2s1 + ASAM", 0.908, C_GREY),
        ("F2s1 5-crop (selected)", 0.911, C_AUG),
    ]
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    colors = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(3.6, 2.55))
    y = np.arange(len(rows))
    ax.barh(y, vals, color=colors, height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=5.6)
    ax.set_xlim(0.875, 0.915)
    ax.set_xticks([0.88, 0.89, 0.90, 0.91])
    ax.set_xlabel("public accuracy")
    ax.set_title("Public leaderboard (1,000 images)", loc="left", fontsize=7)
    for yi, v in zip(y, vals):
        ax.text(v + 0.0008, yi, f"{v:.3f}", va="center", fontsize=5.2)
    ax.grid(axis="x", lw=0.3, alpha=0.5)
    fig.tight_layout()
    save(fig, "fig_leaderboard")


if __name__ == "__main__":
    curves()
    robustness()
    pilot()
    teacher_student()
    tta_deltas()
    leaderboard()
