# Shared repo status — 2026-11-03

## TL;DR

- **This is the shared team repo** (`b_shift` branch). Old code/artifacts are
  archived in `../proj1/our_code_backup/`.
- **Team best: 0.904 public** (teammate, 2-teacher KD, 97.76 val). Our best from the
  old repo: 0.864 (96.50 val ensemble, now not eligible).
- README rules: the final submission is **one ≤5M network — no ensembles**; teachers
  may be any size (`--no-budget`). The 0.864 ensemble is reference only.
- Eval is **new natural photographs** (CIFAR-10.1-like), not corrupted CIFAR. See
  `runs/analysis/shift.md`; prioritize natural-photo robustness, not noise/blur/JPEG.
- **B3 done: AugMix wins** the shifted-val sweep (+3.10pp, clean unchanged).
  **B4 done: 300 epochs + SAM + AugMix = 97.06 val** (97.08 EMA, 97.04 EMA+TTA),
  shifted proxy 89.02 vs 84.32 control — our best single ≤5M model so far.
- Running now: `B_trivial_300_sam`, the same recipe with `--autoaug trivial` (the
  team's "all tricks"), so B4's gain can be separated from "300 epochs + SAM".
- **B5 (KD + AugMix) is blocked**: no compatible teacher checkpoints yet. Old
  `../proj1` checkpoints do not load in this repo (no `config`/`ema`, other arch names).

## Results (seed-0 5k split, this repo unless noted)

| Model | Val % | Public | Notes |
|---|---|---|---|
| teammate "all tricks" | 97.36 | ~0.90 | 5M student, 300 ep + SAM |
| teammate 2-teacher KD | **97.76** | **0.904** | current team best, 600 ep |
| **B_augmix_300_sam** | **97.06** (97.04 EMA+TTA) | — | 4.92M, all tricks + AugMix |
| B_pilot_augmix | 96.52 (TTA) | — | 150 ep, no SAM, AugMix only |
| B_pilot_trivial | 96.48 (TTA) | — | 150 ep, no SAM, control |
| B_pilot_geo | 95.98 (TTA) | — | mild affine, rejected |
| B_pilot_randaug | 96.50 (TTA) | — | rejected |
| B_pilot_autoaug | 96.62 (TTA) | — | autoaug_cifar, rejected |
| B_pilot_extra | 95.46 (TTA) | — | strong affine+color, rejected |
| old repo: w40-robust + WRN-22-4 | 96.50 | 0.864 | ensemble, not eligible |
| old repo: single-teacher KD | 96.24 | — | WRN-40-4 teacher |
| old repo: WRN-40-4 teacher | 96.60 | — | 8.98M, not submittable |

Val numbers are not directly comparable across repos; this repo uses the seed-0 5k
split. `B_augmix_300_sam` is 0.3 below the teammate's "all tricks" on clean val but
is the more robust of the two on every shifted proxy we measure.

## Shift analysis (B1/B2)

- `runs/analysis/shift.md`, `shift_montage.png`, `corruption_gallery.png`.
- 0/2,000 exact CIFAR-10 matches; brighter, higher contrast/sharpness, lower
  blockiness (not JPEG); visually new photos with different framing/background/scale.
- Synthetic `combo` corruption matches the *size* of the drop (~9.1pp), not the
  mechanism. Do not optimize primarily on noise/blur/JPEG.

## B3 pilot (150 epochs, no SAM, no KD, seed 0)

| Run | Clean TTA | Combo shifted |
|---|---|---|
| control: `trivial` | 96.48 | 84.32 |
| `--geo-aug` (mild affine) | 95.98 | 84.96 |
| `--autoaug randaug` | 96.50 | 83.70 |
| `--autoaug autoaug_cifar` | 96.62 | 83.14 |
| `--extra-aug` (strong affine+color) | 95.46 | 84.38 |
| **`--autoaug augmix`** | **96.52** | **87.42** |

- **AugMix is the clear B3 winner**: clean TTA unchanged (+0.04), shifted proxy
  **+3.10pp**; also much better on noise/blur/JPEG.
- Rejected: `--extra-aug`, `randaug`, `autoaug_cifar`. Geometry-only trades 0.50
  clean for +0.64 shifted; not adopted under the clean-drop ≤0.3 rule.
- `--autoaug augmix` is one-view AugMix (no JSD), implemented in `data.py`.

## B4 result: 300 epochs + SAM + AugMix

`B_augmix_300_sam` (4,918,602 params, 300 ep, all tricks, `--autoaug augmix`) vs the
150-epoch `trivial` control. EMA weights, **no TTA**, seed-0 5k holdout:

| corruption | B_pilot_trivial (150 ep) | B_augmix_300_sam | delta |
|---|---|---|---|
| clean | 96.28 | **97.10** | +0.82 |
| combo (shift proxy) | 84.32 | **89.02** | **+4.70** |
| noise 0.05 | 78.20 | 85.22 | +7.02 |
| noise 0.10 | 40.78 | 52.98 | +12.20 |
| blur 1.0 | 74.08 | 79.34 | +5.26 |
| blur 2.0 | 16.78 | 14.70 | −2.08 |
| jpeg 30 | 67.58 | 78.04 | +10.46 |
| contrast 0.7 / 1.3 | 95.94 / 95.94 | 96.92 / 96.76 | +0.98 / +0.82 |
| color 1.3 | 96.16 | 97.02 | +0.86 |

- Shift gap on the proxy narrows from 11.96pp to 8.08pp; clean acc also improves.
- Only regression is heavy blur (2.0) — irrelevant to the eval, which is *sharper*
  than CIFAR. Do not tune for it.
- `results.json`: `val_acc 0.9706`, `val_acc_ema 0.9708`, `val_acc_ema_tta 0.9704`.
- Report: `runs/analysis/B4/B_shift.md` (script output, not tracked — numbers are
  copied above because `runs/` is gitignored).

## Repo state

- Branch: `b_shift`; B3 flag + docs committed. Working tree clean apart from `runs/`.
- Data: `data` symlink → `../proj1/data`, including the 147MB `train_cache.npz`
  generated from the official CIFAR-10 batches (matches the README's `hpc_data.zip`).
- `opencode.json` has the local Kaggle MCP token; excluded via `.git/info/exclude`.
- `runs/` is gitignored: keep results in this file / README, not only on disk.
- Modal: app `cifar-shared`, function `run_task(task, args)`, volumes `proj1-data`
  (`/data`) and `proj1-runs` (`/runs`). A10G, ~19 s/epoch for a 5M student at 300 ep.
  `scripts/shift_analysis.py --tta` is parsed but unused — its tables are always
  no-TTA.

## Next steps

1. `B_trivial_300_sam` (running) — is the B4 gain AugMix or just "300 ep + SAM"?
   Evaluate it with `scripts/shift_analysis.py --checkpoints` on clean + combo.
2. B5 (KD + AugMix): blocked on teacher checkpoints; ask Track A for
   `teacher_wrn_s0.pt` / `teacher_pyr_s2.pt` in this repo's checkpoint format.
   Do not train our own 36.5M teacher on Modal — the team's A100 budget already
   covers A5.
3. Submission: none used today. `B_augmix_300_sam` is the first candidate worth a
   submission, once the trivial control confirms AugMix is the reason.

## Open questions for the teammate

- Teacher checkpoints (`teacher_wrn_s0/s1`, `teacher_pyr_s2`) in this repo's
  `config`+`ema` format — B5 needs them.
- Which branch/commits are current for Track A?
- Does the measured clean-val gap (our 97.06 vs their 97.36 "all tricks") come from
  their 300-ep recipe, or from the architecture/seed?
