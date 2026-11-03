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
- **B3 done: AugMix wins** the shifted-val sweep (pilot: +3.10pp, clean unchanged).
- **B4 done and signed off.** At 300 ep + SAM, AugMix is clean-neutral (EMA 97.08 for
  both) and **+2.18pp on the shifted proxy** (89.02 vs 86.84 for the team's `trivial`
  recipe). The larger jump seen in the pilot came from 300 ep + SAM, not AugMix — do
  augmentation ablations at the schedule you intend to ship.
- Nothing is running. **B4 → A5 (robust teachers) → B5 (KD)** unblocked: A5 should
  train its teachers with `--autoaug augmix`.
- **B5 (KD + AugMix) is downstream of A5, which is downstream of B4.** No teacher
  checkpoints exist yet, and old `../proj1` ones do not load (no `config`/`ema`,
  different arch names).

## Results (seed-0 5k split, this repo unless noted)

| Model | Val % | Public | Notes |
|---|---|---|---|
| teammate "all tricks" | 97.36 | ~0.90 | 5M student, 300 ep + SAM |
| teammate 2-teacher KD | **97.76** | **0.904** | current team best, 600 ep |
| **B_augmix_300_sam** | 97.08 (97.04 EMA+TTA) | — | 4.92M, all tricks + AugMix |
| **B_trivial_300_sam** | **97.08** (97.22 EMA+TTA) | — | 4.92M, team recipe control |
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
split. Our two 300-ep runs are clean-identical (97.08 EMA both) and 0.3 below the
teammate's "all tricks"; AugMix's gain is entirely on the shifted proxy (+2.18pp).

## Shift analysis (B1/B2)

- `runs/analysis/shift.md`, `shift_montage.png`, `corruption_gallery.png`.
- Verifier: `scripts/verify_shift_b1.py` → `runs/analysis/verify_shift.json`.
- 0/2,000 exact CIFAR-10 matches; visually new photos with different
  framing/background/scale (the montage is decisive: full-frame dogs, a bus, a horse
  and rider — not CIFAR's small objects in context).
- Synthetic `combo` corruption matches the *size* of the drop (~9.1pp), not the
  mechanism. Do not optimize primarily on noise/blur/JPEG.

### B1 evidence check (2026-11-03)

Rewritten after re-deriving it from scratch; the original reading leaned on statistics
that do not support the weight they carried.

- **Photometric deltas are real but tiny** (50k train vs 2k eval, Cohen's d):
  sharpness +0.0053 (d 0.27), contrast +0.0101 (0.17), brightness +0.0173 (0.14),
  blockiness −0.0009 (−0.07), hf_ratio +0.0029 (0.06), colorfulness n.s. (p=0.07).
  New photographs from the same classes would differ this much from composition
  alone, so "the stats show the eval is cleaner / not re-JPEG'd" is **not** evidence.
- **The nearest-neighbour tests are the evidence.** eval vs all 50k train, against a
  genuine-CIFAR reference (held-out val vs train) matched identically:

  | matching | genuine pairs (median / p1) | eval→train (median / p1) | eval below genuine p1 |
  |---|---|---|---|
  | dihedral (8 symmetries) | 1852 / 313 | 1982 / 675 | 0.0% |
  | 5 shifts + 8×8 pool | 73.5 / 9.3 | 79.9 / 22.4 | 0.0% |

  Eval images are at least as far from every CIFAR image as unrelated CIFAR images
  are. A brightness/contrast transform, re-JPEG, flip/rotation or shift-and-crop of a
  CIFAR image would collapse that residual toward zero; none does. So the eval is
  **different content, not transformed CIFAR-10** — which also closes the "same
  photos, different save pipeline" possibility.
- **Consequence for B2:** the corruption branch is closed, and matching the *size* of
  the accuracy drop is not evidence of mechanism. The inherited line "the corruption
  with a drop closest to the Kaggle gap is the best proxy" (old `shift.md`) is wrong.
  Corruption numbers stay as a stress test / regression guard only; augmentation
  choices are made on clean val plus the natural-photo prior, and validated by the LB.
- Residual risk: the NN test cannot rule out eval images derived from the CIFAR-10
  **test** split (we do not have it) or a non-rigid transform. Low, but unverified.

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

## B4 result: 300 epochs + SAM, AugMix vs trivial (both 300 ep, seed 0)

Both runs are `r3_64_3-5-3` (4,918,602 params), 300 ep, all tricks + SAM + EMA 0.999.
`results.json`:

| run | `--autoaug` | val | TTA | EMA | EMA+TTA |
|---|---|---|---|---|---|
| B_augmix_300_sam | `augmix` | 97.06 | 97.02 | 97.08 | 97.04 |
| B_trivial_300_sam | `trivial` (team recipe) | 96.98 | **97.20** | 97.08 | **97.22** |

Shifted proxy, EMA weights, **no TTA**, seed-0 5k holdout (`runs/analysis/B4_300`):

| corruption | trivial | augmix | augmix − trivial |
|---|---|---|---|
| clean | 97.08 | 97.10 | +0.02 |
| combo (shift proxy) | 86.84 | **89.02** | **+2.18** |
| noise 0.05 | 80.92 | 85.22 | +4.30 |
| noise 0.10 | 40.10 | 52.98 | **+12.88** |
| jpeg 30 | 72.88 | 78.04 | +5.16 |
| blur 1.0 | 79.88 | 79.34 | −0.54 |
| blur 2.0 | 15.84 | 14.70 | −1.14 |
| contrast 0.7 | 97.00 | 96.92 | −0.08 |
| contrast 1.3 | 96.70 | 96.76 | +0.06 |
| color 1.3 | 96.80 | 97.02 | +0.22 |

**Verdict: B4 stands, but smaller than the pilot implied.** On clean val the two are
identical (EMA 97.08 both; TTA favours `trivial` by 0.18, i.e. ~9 of 5000 images).
The genuine AugMix contribution is **+2.18pp on the natural-photo proxy** and a much
larger margin under noise/JPEG (+12.9pp at noise 0.10, +5.2pp at JPEG 30).

**Correction to the earlier claim:** the 150-ep → 300-ep comparison (+4.70 combo,
+0.82 clean) mostly measured **300 epochs + SAM**, not AugMix — the 300-ep `trivial`
control also jumped (84.32 → 86.84 combo, 96.28 → 97.08 clean). Only the two 300-ep
runs above isolate the augmentation. Lesson for future ablations: compare augmentation
at the schedule you intend to ship, not at a cheap one.

TTA note: TTA adds +0.14 to `trivial` but −0.04 to `augmix`, the same pattern Track A
saw on their KD student. Treat TTA as noise-level for these models.

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

Dependency chain: **B4 (done) → A5 (robust teachers) → B5 (KD + AugMix)**. A5 is
unblocked: it should train its teachers with `--autoaug augmix` and share them in this
repo's `config`+`ema` format. Do not train a 36.5M teacher on Modal to shortcut that —
it duplicates A5 on paid credit and yields one teacher instead of three.

1. **Hand off to Track A.** Tell Rahim B4 is signed off: flag merged (`--autoaug
   augmix`), clean-neutral, +2.18pp shifted, +12.9pp at noise 0.10. Ask for the teacher
   checkpoints once A5 finishes.
2. **Seed variance (ours, next spend ~$2).** Our number is a single seed. A second
   AugMix seed at 300 ep (~1.75 h) tells us whether 97.08 / 89.02 is the model or a
   lucky draw, and gives a second candidate for our 2 submission slots. Recommended
   before any submission, since the final model is a single ≤5M network.
3. **B5** once A5's teachers land: 3-teacher KD on this AugMix recipe vs the same
   without AugMix, 300 ep, scored on clean + shifted.
4. **Submission:** none used today. Candidates are `B_augmix_300_sam` (robustness:
   +2.18 combo) and `B_trivial_300_sam` (clean EMA+TTA 97.22 vs 97.04). Pick by val +
   shifted val, state both in the submission message.

## Open questions for the teammate

- Teacher checkpoints (`teacher_wrn_s0/s1`, `teacher_pyr_s2`) in this repo's
  `config`+`ema` format — B5 needs them; A5 should use `--autoaug augmix`.
- Which branch/commits are current for Track A?
- Their "all tricks" is 97.36 clean vs our 96.98-97.08 for the same recipe: is that
  the architecture, the seed, or something in their recipe we are missing?
- Our 300-ep control (97.22 EMA+TTA) is worth re-checking against their 97.36 — if
  the gap is real and not seed luck, something in their "all tricks" is unaccounted for.
