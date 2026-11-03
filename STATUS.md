# Shared repo status — 2026-11-02

## TL;DR

- **This is now the shared team repo** (`b_shift` branch). Old code/artifacts are
  archived in `../proj1/our_code_backup/`.
- Data pipeline works: `data/train_cache.npz` (50k) + `data/test_images` (2k);
  `train.py` → `scripts/summarize.py` → `predict.py` smoke test passed
  (`runs/B_smoke`).
- **Team best: 0.904 public** (teammate, 2-teacher KD, 97.76 val). Our best from the
  old repo: **0.864** (96.50 val ensemble).
- README rules: the final submission is **one ≤5M network — no ensembles**; teachers
  may be any size (`--no-budget`). The 0.864 ensemble is reference only, not a valid
  final model.
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
seed-0 5k split. The old 0.864 row is an ensemble and is **not eligible as the final
model** under the README rule (one ≤5M network, no ensembles).

## Shift analysis (B1/B2)

- `runs/analysis/shift.md`, `shift_montage.png`, `corruption_gallery.png`.
- 0/2,000 exact CIFAR-10 matches; brighter, higher contrast/sharpness, lower
  blockiness (not JPEG); visually new photos with different framing/background/scale.
- Synthetic `combo` corruption matches the *size* of the drop (~9.1pp), not the
  mechanism. Do not optimize primarily on noise/blur/JPEG.

## B3 results (2026-11-02, 150 epochs, no SAM, seed 0)

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
- Full table: `runs/analysis/B_shift.md`.
- Running: **300 epochs + SAM + AugMix** (`B_augmix_300_sam`) as the B4 candidate.

## Repo state

- Branch: `b_shift`; README pulled and merged (`b3cbec6`). Docs updated after the
  README; commit when reviewed.
- Data: `data` symlink → `../proj1/data`, including the 147MB
  `train_cache.npz` generated from the official CIFAR-10 batches. This matches the
  README's `hpc_data.zip` cache; `train_labels.csv` is absent but not needed while
  the cache exists.
- `opencode.json` has the local Kaggle MCP token; excluded via `.git/info/exclude`.
- `runs/B_smoke` is the 2-batch smoke test (not a real model).
- No HPC or Modal jobs running.

## Next steps

1. Review/commit `AGENTS.md`, `PLAN.md`, `STATUS.md` on `b_shift`.
2. Port B1/B2 to this repo's `data.py`/`models.py` and seed-0 split.
3. B3 complete: **AugMix wins** (+3.1 shifted, clean unchanged); `--autoaug augmix` is in the repo.
4. B4: evaluate the 300-epoch + SAM AugMix run; if clean is ~97.3 and shifted is
   strong, add it to the README results table and consider it for the final single
   model (no ensemble).
5. B5: KD with AugMix once Track A's teacher checkpoints arrive.
6. Keep submissions for clear val/shifted-val wins; final pick is one ≤5M network.

## Open questions for the teammate

- Teacher checkpoints and the exact "all tricks" flags for the 97.76 run.
- Which branch/commits are current for Track A?
- Is `summarize.py` the canonical results table, and are run names `A_...`/`B_...`?
- Does our `train_cache.npz` match `hpc_data.zip`?
