# Shared repo status — 2026-11-02

## TL;DR

- **This is now the shared team repo** (`b_shift` branch). Old code/artifacts are
  archived in `../proj1/our_code_backup/`.
- Data pipeline works: `data/train_cache.npz` (50k) + `data/test_images` (2k);
  `train.py` → `scripts/summarize.py` → `predict.py` smoke test passed
  (`runs/B_smoke`).
- **Team best: 0.904 public** (teammate, 2-teacher KD, 97.76 val). Our best from the
  old repo: **0.864** (96.50 val ensemble).
- Eval is **new natural photographs** (CIFAR-10.1-like), not corrupted CIFAR. See
  `runs/analysis/shift.md`; prioritize natural-photo robustness, not noise/blur/JPEG.
- Next: B3 natural-photo augmentation ablations on `r3_64_3-5-3`; no submissions
  used today.

## Results

| Model | Val | Public | Notes |
|---|---|---|---|
| teammate "all tricks" | 97.36 | ~0.90 | 5M student |
| teammate 2-teacher KD | **97.76** | **0.904** | current team best |
| old repo: w40-robust + WRN-22-4 | 96.50 | 0.864 | old code, incompatible checkpoints |
| old repo: single-teacher KD | 96.24 | — | WRN-40-4 teacher |
| old repo: WRN-22-4 AugMix | 96.00 | — | |
| old repo: WRN-40-4 teacher | 96.60 | — | 8.98M, not submittable |

Val numbers are not directly comparable across repos/splits; this repo uses the
seed-0 5k split.

## Shift analysis (B1/B2)

- `runs/analysis/shift.md`, `shift_montage.png`, `corruption_gallery.png`.
- 0/2,000 exact CIFAR-10 matches; brighter, higher contrast/sharpness, lower
  blockiness (not JPEG); visually new photos with different framing/background/scale.
- Synthetic `combo` corruption matches the *size* of the drop (~9.1pp), not the
  mechanism. Do not optimize primarily on noise/blur/JPEG.

## Repo state

- Branch: `b_shift` (docs untracked; commit when reviewed).
- Data: `data` symlink → `../proj1/data`, including the 154MB
  `train_cache.npz` generated from the official CIFAR-10 batches.
- `opencode.json` has the local Kaggle MCP token; excluded via `.git/info/exclude`.
- `runs/B_smoke` is the 2-batch smoke test (not a real model).
- No HPC or Modal jobs running.

## Next steps

1. Review/commit `AGENTS.md`, `PLAN.md`, `STATUS.md` on `b_shift`.
2. Port B1/B2 to this repo's `data.py`/`models.py` and seed-0 split.
3. B3 natural-photo ablations on `r3_64_3-5-3`; log runs as `B_...`.
4. B4 merge winning flag; B5 KD check with Track A's 3 teachers.
5. Keep submissions for clear val/shifted-val wins; final pick 2 single ≤5M models.

## Open questions for the teammate

- Teacher 5M rule; teacher checkpoints and the exact "all tricks" flags.
- Which branch/commits are current for Track A?
- Is `summarize.py` the canonical results table, and are run names `A_...`/`B_...`?
