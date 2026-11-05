# Shared repo status — 2026-11-05

## TL;DR

- **This is the shared team repo** (`b_shift` branch). Old code/artifacts are
  archived in `../proj1/our_code_backup/`.
- **Team best: 0.911 public** (`KEEP/F2s1.pt`, 1200 ep, full data, 5-crop TTA),
  holdout equivalent A4 97.84 / 97.92 (5-crop). Our old-repo best: 0.864.
- **Deadline: midnight 2026-11-06.** `EXPERIMENTS.md` is the run-by-run backbone.
- On the rules: the official text (`../proj1/overview.md`) says the 5M budget applies
  to "a modified ResNet" and **never mentions ensembles** — the "no ensembles" line is
  the team README's convention, not staff's. We do not rely on ensembles anyway: a
  two-seed AugMix ensemble measured +0.34 on the composite proxy (p=0.15), inside the
  ±1.1pp public noise.
- Eval is **new natural photographs**, not corrupted CIFAR — verified by a
  nearest-neighbour test (`scripts/verify_shift_b1.py`), not by the photometric stats.
- **B4 done and signed off**, now with paired statistics: AugMix is clean-neutral
  (97.08 EMA both arms, p=0.46) and **+2.18pp on the composite proxy** (p=3.6e-9),
  +4.1 noise, +5.3 JPEG, +1.1-1.4 on photographic jitter (p≈1e-5). Stable across two
  seeds (+2.18, +1.80).
- **B5 is running now** (unblocked: Rahim shared the 3 teachers, all with `val_size=5000`).
  Two arms, `r3_80_2-3-3`, 300 ep, 3 teachers, SAM+EMA, KL α=0.9 T=4: `--autoaug trivial`
  vs `--autoaug augmix`, seed 0. ~59 s/epoch → ~4.9 h each, in parallel.
- Reproduced his best submission from our checkout: `scripts/tta_eval.py submit` gives
  **0 differences** against `KEEP/submission_F2s1_tta5.csv`.

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

- Branch: `b_shift`, now merged with Rahim's `main` (EXPERIMENTS.md, TTA module, LS-KD,
  ASAM fix, cyclic/SWA, Modal launcher). Push needed so Track A can use `--autoaug augmix`.
- Data: `data` symlink → `../proj1/data`, including the 147MB `train_cache.npz`.
- **Shared big files** (gitignored, from Rahim's Drive zip, repo root):
  `KEEP/` (F2s1.pt = 0.911 model + its csv — do not overwrite), `ckpts/` (A4.pt 97.92
  with 5-crop, kd2t.pt 0.904), `teachers/` (wrn_s0 98.06, wrn_s1 98.12, pyr_s2 97.64),
  `submissions/` (every submitted csv).
- Environment check, reproduced from his checkout:
  `python scripts/tta_eval.py cache KEEP/F2s1.pt probs/F2s1_test.npz --split test` then
  `python scripts/tta_eval.py submit probs/F2s1_test.npz out.csv --check KEEP/submission_F2s1_tta5.csv`
  → **0 differences** (data, code and TTA are identical to his).
- `runs/` is gitignored: keep results in this file / README, not only on disk.
- Modal: app `cifar-shared`, function `run_task(task, args)`, volumes `proj1-data`
  (`/data`, teachers uploaded to `/data/teachers`) and `proj1-runs` (`/runs`). A10G:
  ~19-21 s/epoch for a 5M student, ~59 s/epoch for 3-teacher KD.
  `scripts/shift_analysis.py --tta` is parsed but unused — its tables are always
  no-TTA.

## B5 (running, 2026-11-05)

Rahim's 3 teachers landed and all have `val_size=5000`, so there is no teacher-on-val
leak. Two matched arms, student `r3_80_2-3-3` (4.88M, his best sweep shape, 97.62 at
300 ep), 300 epochs, 3 teachers (`wrn_s0` + `wrn_s1` + `pyr_s2`), KL alpha=0.9 T=4,
TrivialAug/Cutout16/mixup/CutMix, SAM rho=0.05, EMA 0.999, seed 0:

| run | change vs control | status |
|---|---|---|
| `B_kd3t_trivial` | control: `--autoaug trivial` | running |
| `B_kd3t_augmix` | `--autoaug augmix` | running |

~4.9 h each, run in parallel; a listener downloads both and evaluates clean + the
proxies with paired McNemar, using `ckpts/A4.pt` (600 ep, trivial) as reference.

## Next steps (deadline: midnight 2026-11-06)

1. **B5 result -> hand off.** If AugMix holds on top of KD, A6 (full data, 1200 ep,
   `--val-size 0`) should use `--autoaug augmix`; the flag is on `b_shift`. His
   acceptance rule for A4-level claims: beat the matched control *and* A4 on the full
   val under 5-crop, paired, raw+EMA agreeing, then repeat from another seed.
2. **Submission slot (ours).** Current in-repo candidates are `B_augmix_300_sam`
   (clean 97.08 EMA, combo 89.02) and `B_trivial_300_sam` (97.08 EMA / 97.22 TTA,
   combo 86.84); the KD arms should beat both if the recipe transfers. State val +
   shifted val in the submission message.
3. **Report.** Method/setup draft in `REPORT_DRAFT.md`; Rahim owns results+discussion.
4. **Rules**: no ensemble, no eval-label use; the 5M-vs-teacher reading is still an
   open question for staff and is flagged in the report draft.

## Open questions for the teammate

- Does the 5M budget constrain teachers, or only the submitted student? (Flagged in
  `REPORT_DRAFT.md` section 1; our teachers are 28.5M/36.5M.)
- Does A4 (600 ep, `trivial`) count as the matched control for a 300-epoch AugMix arm,
  or does he want a 600-epoch AugMix arm too?
- Timing for A6: if it starts before B5 finishes, does he train with `trivial` and
  re-run, or wait ~5 h for our answer?
- `opencode.json` holds the local Kaggle MCP token and is excluded via
  `.git/info/exclude`; never commit or print it.
