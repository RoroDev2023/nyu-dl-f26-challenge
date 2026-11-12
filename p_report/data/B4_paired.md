# B4 paired statistics + a better-matched shift proxy (2026-11-05)

Setup: seed-0 5k holdout, EMA weights, hflip TTA unless stated, exact McNemar
(binomial on discordant pairs, two-sided). All comparisons are paired per image.
Scripts: `scripts/paired_stats.py`, `scripts/photo_proxy.py`, `scripts/tta_shift.py`.

## augmix vs trivial (300 ep + SAM, the B4 claim)

| condition | policy | trivial | augmix | net | fixes/intros | McNemar p |
|---|---|---|---|---|---|---|
| clean | flip | 97.22 | 97.06 | −0.16 | 41/49 | 0.46 |
| clean | 5crop | 97.20 | 97.16 | −0.04 | 40/42 | 0.91 |
| combo | flip | 87.18 | 89.46 | **+2.28** | 243/129 | 3.6e-09 |
| combo | 5crop | 87.78 | 89.84 | +2.06 | 227/124 | 4.2e-08 |
| jpeg_30 | flip | 73.52 | 78.82 | **+5.30** | 514/249 | 4.6e-22 |
| noise_0.05 | flip | 81.46 | 85.60 | **+4.14** | 342/135 | 9.0e-22 |
| shift3 (±3px) | flip | 96.86 | 96.92 | +0.06 | 53/50 | 0.84 |
| zoom_1.2 | flip | 94.82 | 95.36 | +0.54 | 107/80 | 0.057 |
| zoom_1.2 | 5crop | 94.46 | 95.32 | +0.86 | 109/66 | 1.4e-03 |
| photo_mild | flip | 94.80 | 95.90 | **+1.10** | 104/49 | 1.0e-05 |
| photo_strong | flip | 92.76 | 94.14 | **+1.38** | 155/86 | 1.0e-05 |

`photo_*` = per-image brightness/contrast/saturation scaling + small zoom + small
rotation (mild: brightness/contrast ±8%, sat ±12%, zoom 1.05, rot ±3°; strong: ±20%,
±30%, 1.15, ±6°). This is the closest label-free proxy to "new photographs" we can
build from train data, since it mimics camera/lighting variation rather than coding
noise.

**Verdict.** AugMix is clean-neutral (p=0.46) and its gain is real on every
image-quality axis: +2.28 combo, +4.14 noise, +5.30 JPEG, and **+1.1 to +1.4 on
photographic jitter, p≈1e-5**. Geometry alone (translation, 1.2× zoom) is where it does
nothing — consistent with the shift's difficulty being content/lighting, not framing.
So B4 is not an artefact of the noise/blur/JPEG corruptions in the proxy.

## 5crop vs flip TTA (partner's 0.911 procedure)

| run | condition | flip | 5crop | net | fixes/intros | p |
|---|---|---|---|---|---|---|
| trivial | clean | 97.22 | 97.20 | −0.02 | 5/6 | 1.0 |
| trivial | jpeg_30 | 73.52 | 74.76 | +1.24 | 109/47 | 7.5e-07 |
| trivial | combo | 87.18 | 87.78 | +0.60 | 46/16 | 1.8e-04 |
| trivial | noise_0.05 | 81.46 | 81.64 | +0.18 | 48/39 | 0.39 |
| trivial | zoom_1.2 | 94.82 | 94.46 | −0.36 | 15/33 | 0.013 |
| augmix | clean | 97.06 | 97.16 | +0.10 | 11/6 | 0.33 |
| augmix | jpeg_30 | 78.82 | 79.38 | +0.56 | 75/47 | 0.014 |
| augmix | combo | 89.46 | 89.84 | +0.38 | 40/21 | 0.020 |
| augmix | noise_0.05 | 85.60 | 85.20 | −0.40 | 21/41 | 0.015 |
| augmix | zoom_1.2 | 95.36 | 95.32 | −0.04 | 17/19 | 0.87 |

**Verdict.** Multi-view TTA is a wash: tiny gains on JPEG/combo (≤0.6pp), losses on
noise/zoom, nothing on clean or geometry. The hflip-only TTA in `train.py` captures
essentially all of it, so the 0.911-vs-0.904 public difference should not be
attributed to TTA (it is inside the ±1.1pp public noise). Cost is 10x inference;
not worth it for the final model.

Implementation check: the `flip` column reproduces `train.py`'s own
`val_acc_ema_tta` exactly (97.22 / 97.06), so this is the same TTA path the
submission uses, not a reimplementation.

## Second AugMix seed (seed 1) — seed variance

`B_augmix_300_sam_s1`, identical recipe, `--seed 1`. EMA weights, no TTA:

| condition | trivial (s0) | augmix s0 | augmix s1 | augmix mean | seed spread |
|---|---|---|---|---|---|
| clean | 97.08 | 97.10 | 96.90 | 97.00 | 0.20 |
| combo | 86.84 | 89.02 | 88.64 | 88.83 | 0.38 |
| noise_0.05 | 80.92 | 85.22 | 83.86 | 84.54 | 1.36 |
| noise_0.10 | 40.10 | 52.98 | 51.30 | 52.14 | 1.68 |
| jpeg_30 | 72.88 | 78.04 | 76.98 | 77.51 | 1.06 |
| blur_1.0 | 79.88 | 79.34 | 80.04 | 79.69 | 0.70 |
| color_1.3 | 96.80 | 97.02 | 96.78 | 96.90 | 0.24 |

`results.json`: s0 val 97.06 / TTA 97.02 / EMA 97.08 / EMA+TTA 97.04;
s1 96.90 / 97.08 / 96.88 / 97.08.

**B4 holds across seeds.** Both seeds beat the `trivial` control on combo (+2.18 and
+1.80), the mean gain is +1.99, and clean stays neutral (mean 97.00 vs 97.08 — the
0.2pp seed spread is the same size as the clean difference). Noise is the least stable
condition (1.36pp between seeds), so do not read small noise deltas as real.

Report: `runs/analysis/B4_seeds/B_shift.md`.

## Is a 2-seed ensemble even worth the rules question?

Softmax average of the two AugMix seeds vs the better single, paired:

| condition | s0 | s1 | ensemble | ens − best single | b/c | p |
|---|---|---|---|---|---|---|
| clean | 97.06 | 97.08 | 97.04 | −0.04 | 21/23 | 0.88 |
| combo | 89.46 | 89.02 | 89.80 | +0.34 | 69/52 | 0.15 |
| photo_strong | 94.14 | 94.22 | 94.66 | +0.44 | 49/27 | 0.02 |

**No.** +0.3 to +0.4pp, not significant on combo, and inside the ±1.1pp public-LB
noise. Combined with the official ruleset's silence (`proj1/overview.md`: the 5M budget
is stated for "a modified ResNet", no ensemble clause either way), the ensemble
question is not worth staff time: the expected gain is smaller than the measurement
error, and the per-model reading is the only one the text supports.
