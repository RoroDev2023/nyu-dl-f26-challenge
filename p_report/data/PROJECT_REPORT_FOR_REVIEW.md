# CIFAR-10 ResNet (≤5M params) Kaggle challenge — full project report for outside review

**Purpose of this document.** We want independent opinions on how to raise our **hidden-test (Kaggle)
score**. Current best public score: **0.911** (13th place). **0.916 (+0.5 pt) would put us around 6th.**
Deadline: **Nov 6, 2026, 23:59 ET** (a few hours from when this was written). Everything we have tried,
with numbers, is below — please do not re-propose something already tested unless you explain why our
test was flawed. Numbers are measured; anything labeled "hypothesis" is not established.

---

## 1. The competition and its rules

- NYU Tandon Deep Learning F26, Kaggle "cifar-10-res-net-architecture-challenge-dl".
- Train data: CIFAR-10 train set given as 50,000 PNGs + labels CSV. Hidden test: **2000 images**
  (ids 0–1999), submission columns `id,label`. Public/private split proportion: **unknown**.
- Model must be **ResNet-style**, **≤ 5M trainable parameters for the complete inference system**
  (an ensemble counts all members together), trained **from scratch**, **only the provided training
  data**, **no pretrained weights**.
- The professor allowed test-time BN adaptation and class-balanced prediction, if disclosed.
- Two final submissions are selected on Kaggle.

## 2. Data, validation and evaluation protocol

- Fixed class-balanced **5000-image validation holdout** (`train_val_split(val_size=5000, seed=0)`),
  held out by **every student and every teacher** (verified in an audit; no leakage).
- "Full-data" models train on all 50,000 images (`--val-size 0`) and have no validation number.
- **TTA used everywhere (`tta.py`)**: reflect-pad 4, windows shifted by (0,0), (±2,0), (0,±2), plus
  the same 5 on the horizontally flipped image = 10 views, **softmax probabilities averaged**.
  Called "5-crop" below. "flip" = plain image + its flip only.
- The 0.911 submission was **reproduced exactly (0 differing labels)** from its checkpoint with this
  shared code; validation and submission use the identical function.
- Paired comparisons: per-image correctness on the same 5000 images, exact two-sided sign test on
  discordant pairs. 0.1 pt = 5 images. Differences under ~8–10 images are usually noise.
- Val is reused many times → nominal p-values are exploratory.

## 3. Code (repo: github.com/RoroDev2023/nyu-dl-f26-challenge, private)

- `train.py`: SGD Nesterov momentum .9, LR warmup + cosine per step (or cyclic for SWA),
  weight decay 5e-4 (not on BN/bias), label smoothing .1 (only when no teachers), bf16 autocast on
  A100/H100, EMA of weights (.999), SAM / ASAM (BN running stats restored after SAM's 2nd pass),
  mixup + CutMix (turned off for the last `--mix-off-epochs`), TrivialAugment, Cutout,
  resume-from-checkpoint (refuses changed args), `--init-from` (start from weights),
  `--gpu-aug` (augmentation on GPU), `--drop-path`, `--dropout`, `--se`.
- **KD**: teachers see **exactly the student's augmented and mixed batch** (consistent teaching).
  Loss = (1−α)·CE + α·KL(T)·T², α .9, T 4; ensemble target = mean of teacher softmaxes at T.
  Options: logit-standardized KD (`--kd-mode lskd`), different α/T, AugMix + JSD consistency
  (`--jsd-weight`, added Nov 6).
- `models.py`: ResNet BasicBlock (3×3 stem, no max-pool, zero-init last BN), WideResNet,
  PyramidNet + ShakeDrop.
- `scripts/tta_eval.py` (cached per-view probabilities, policies, paired compare, ensembles,
  submission), `scripts/swa_build.py`, `scripts/soup.py`, `scripts/shift_eval.py`.

## 4. Best model and recipe

**Architecture `r3_80_2-3-3`**: 3 stages, channels 80/160/256, blocks 2/3/3, **4,880,474 params**.

**Recipe**: LR 0.1 cosine, warmup 5, batch 128, wd 5e-4, TrivialAugmentWide, Cutout 16,
mixup α .2 + CutMix α 1.0 (off for the last 15 ep at 300 ep / 50 ep at 1200 ep), SAM ρ .05,
EMA .999, KD α .9 T 4 from 3 teachers.

**Teachers** (val, all held out the same 5000 images):
| Teacher | Params | Val |
|---|---|---|
| WRN-28-10 seed 0 | 36.5M | 98.06 |
| WRN-28-10 seed 1 | 36.5M | 98.12 |
| PyramidNet-110 (α 270) + ShakeDrop | 28.5M | 97.64 |
Best teacher pool: the two WRNs, 98.14. Pools that added our students scored lower
(98.06 → 97.94); students alone 97.78.

**Best submission: F2s1** = this recipe, 1200 epochs, full data, seed 1 → public **0.903 (flip)**,
**0.911 (5-crop)**.

## 5. Main results timeline (val = 5000 holdout, EMA weights)

| Run | Setup | Val (flip) | Val (5-crop) | Public LB |
|---|---|---|---|---|
| baseline | r3_64_3-5-3, plain | 94.76 (no TTA), 95.42 flip | | |
| tricks, no KD | + TA, Cutout, mixup/CutMix, SAM, EMA | 97.36 | | |
| kd2t | r3_64, 2 teachers, 600 ep | 97.74 | 97.74 | 0.904 (flip), 0.901 (5-crop) |
| kd 3 teachers | r3_64, 600 ep | 97.76 | | |
| arch sweep 300 ep | r3_80_2-3-3 / r3_96_2-2-2 / r3_64_3-5-3 | 97.62 / 97.42 / 97.38 | | |
| A3 | r3_80 300 ep, KD T 2 / T 1 α 1 | 97.62 / 97.50 | | |
| **A4** | r3_80, 600 ep | 97.84 | **97.92** | |
| V2 | r3_80, 1200 ep | 97.82 | 97.94 | |
| F1 / F2 | r3_80, 1200 ep, full data | | | 0.898 / 0.899 (F2 5-crop 0.898) |
| **F2s1** | same, seed 1 | | | **0.903 flip / 0.911 5-crop** |

- Doubling epochs 600 → 1200 changed val by 1 image (A4 97.92 vs V2 97.94).
- F2 and F2s1 (only the seed differs) disagree on **61 / 2000** test labels and differ by
  0.4–1.3 pt on the public LB → public scores carry roughly ±0.5 pt of seed/subset noise.

## 6. Post-processing and inference experiments

| Experiment | Result |
|---|---|
| Test-time BN adaptation (F2) | val 97.34 (worse); public **0.884** (worse) |
| Sinkhorn class balancing (F2) | public 0.898 (no gain) |
| BN recalibration on clean train images | hurts: A4 97.84 → 97.08, V2 97.82 → 97.20 |
| TTA weighting: center+flip at 50% / 80% vs equal 5-crop (tune/confirm halves, A4 and V2) | no gain (A4 +0/+1, V2 −1/−2 images on the confirm half) → equal 5-crop kept |
| 5-crop vs flip on public | F2s1 +0.8, F2 −0.1, kd2t −0.3 (noisy) |

## 7. Diagnostics: the validation → leaderboard gap

**Observation**: val ≈ 97.9% but public ≈ 91%. Facts:
- Audit: no train/val leakage, pipeline parity exact (val via submission path = 97.82),
  per-class val accuracy unremarkable, submission id mapping verified.
- Test images vs train: brighter (mean .513/.498/.461 vs .491/.482/.447), higher std,
  ~7% sharper, lower saturation.
- Student confidence: .850 on test vs .896 on val.
  Student–teacher-ensemble agreement: **96.9% on test vs 99.24% on val**.
- Simulating test pixel statistics on val (brightness/contrast/saturation matching) costs only
  **0.1–0.3 pt** → simple pixel-statistics shift does not explain a ~7-point gap.
- **Hypothesis only**: content-level shift similar to CIFAR-10.1-style re-collected test sets
  (Recht et al. 2018 report 4–10 pt drops for CIFAR-10 models on a new test set). Not proven.

## 8. Late-stage continuations ("finishers") from converged checkpoints

All from A4 unless noted; 5-crop EMA unless noted; "vs control" is paired on the same images.

**40-ep finish, LR .003 (FIN pair)**: with TA+Cutout 97.76 flip; crop+flip only 97.76 flip.
No change vs A4 (97.84 flip).

**60-ep continuations, LR .005 cosine, TA + Cutout, no mixing, 3 teachers:**
| Run | Change vs control | 5-crop | vs control (EMA / raw) | vs A4 | Val NLL |
|---|---|---|---|---|---|
| A4 (untouched) | | 97.92 | | | 0.154 |
| C60_sam | control, SAM ρ .05 | 97.82 | | −5 (p .27) | 0.154 |
| C60_asam | ASAM ρ .5, η .01 (authors' formulation) | 98.00 | +9 (p .049) / +11 (p .013) | +4 (p .50) | 0.153 |
| C60_a045 | KD α .45 | 97.96 | +7 / +4 | +2 | **0.129** |
| C60_lskd | logit-standardized KD, T 2, 0.1 CE + 9 KL | 97.76 | −3 / −6 | −8 | 0.086 |
| C60_swa | SWA: 6×10-ep cycles .01→.001, avg cycles 2–6, BN recomputed | 97.80 | −1 / −1 | −6 | 0.153 |

**ASAM confirm from a second start point (kd2t, r3_64, 97.74)**: SAM-cont 97.76, ASAM-cont 97.78
(+1 EMA / +4 raw vs control, +2 vs untouched). Reading: ASAM beats a SAM *continuation*
consistently, but no continuation beats the untouched start model.

**ASAM finish applied to F2s1 (full data)**: changes 33/2000 test labels; **public 0.908** (vs 0.911).

**80-ep AugMix + JSD consistency finish** (Nov 6; Hendrycks et al. 2020 reference ops, severity 3,
width 3, JSD weight 12 ramped over 10 ep, CE + KD on the clean view; LR .01, warmup 5, cosine;
no mixing/Cutout; matched control without the AugMix branch). Flip TTA, plus a frozen shifted-val
test (`scripts/shift_eval.py`):
| Model | clean | bright +25/255 | contrast ×1.2 | gamma .8 | 3×3 blur | 2 px shift | shifted mean |
|---|---|---|---|---|---|---|---|
| A4 | **97.84** | 97.82 | 97.78 | 97.82 | 95.66 | 97.82 | **97.38** |
| J80_ctrl | 97.44 | 97.42 | 97.56 | 97.40 | 94.14 | 97.28 | 96.76 |
| J80_jsd | 96.88 | 96.82 | 96.88 | 96.86 | **96.30** | 96.78 | 96.73 |
JSD buys blur robustness but costs ~1 pt everywhere else; the 80-ep finish itself (control) also
loses ~20 clean images vs A4.

**Pattern**: across ~10 finisher experiments, **no late continuation has beaten its untouched
starting checkpoint**; finishes without the full mixing/Cutout recipe lose accuracy.

## 9. Fresh-training experiments (from scratch, matched controls)

**Regularizer / KD sweep (Nov 6), 300 ep, seed 0, identical recipe + `--gpu-aug`, one change each,
5-crop EMA, paired vs control:**
| Run | Change | 5-crop | vs control |
|---|---|---|---|
| V300_ctrl | none | 97.66 (raw 97.62) | |
| V300_t2 | KD from the 2 WRN teachers only (best pool) | 97.66 | ±0 |
| V300_a045 | KD α .45 | 97.44 (raw 97.42) | −11 |
| V300_dp1 | stochastic depth 0.1 | 97.30 (raw 97.38) | **−18 (p .005)**, NLL .173 vs .160 |

**Fresh 300-ep ASAM vs SAM** (same args as the arch-sweep r3_80 run, SAM → ASAM ρ .5 η .01):
97.64 flip EMA / 97.66 raw vs 97.62 → +1 image. Not a gain.

**Fresh 300-ep logit-standardized KD**: 97.62 5-crop (97.50 flip) vs 97.62 ordinary KD → no gain;
much lower val NLL (0.092 vs ~0.15) — a calibration effect, not accuracy.

**Budget split into two WRN-40-2 (2.24M each, 4.49M total), 300 ep, seeds 0/1**: 97.10 / 97.06
alone, **97.26 averaged** (error Jaccard overlap .61) vs one 4.88M r3_80 at ~97.6 → loses.

## 10. Weight soups

- Soup of kd2t + its ASAM finish (BN recomputed on the train split with training aug): 97.72 vs
  97.74 / 97.78 (−1 / −3), lower NLL. No gain.
- Soup of F2s1 + its ASAM finish: **public 0.908** (vs 0.911).

## 11. All public leaderboard scores so far

| Submission | Public |
|---|---|
| **F2s1, 5-crop** | **0.911** |
| kd2t (flip) | 0.904 |
| F2s1 flip | 0.903 |
| kd2t 5-crop | 0.901 |
| F2 (flip) / F2 5-crop / F2 Sinkhorn | 0.899 / 0.898 / 0.898 |
| F1 | 0.898 |
| F2s1 + ASAM finish | 0.908 |
| soup F2s1 + ASAM finish | 0.908 |
| F2 + test-time BN adaptation + balancing | 0.884 |
| F2s1 with center-weighted TTA (cf80) | pending at time of writing |

## 12. Compute and time available

- Remaining time: a few hours (deadline 23:59 ET Nov 6).
- Rented GPUs (Vast.ai), affordable: 2× A100 or 2× H100 machines, ~$1.2–3.6/h.
  Measured speed for r3_80 + 3 teachers + SAM: ~27 s/epoch on A100 (CPU aug), ~18–19 s/epoch on
  H100 (`--gpu-aug`); a 600-ep full-data run ≈ 3.3 h on H100; 1200 ep ≈ 9 h on A100.
  AugMix-JSD runs ≈ 37 s/epoch on A100.
- Setup per rented machine ≈ 20–30 min (upload a 630 MB bundle).
- Laptop (Apple MPS) for inference: ~1 min per 5000-image TTA view.

## 13. Untested ideas

- Decoupled KD (DKD) — needs a no-mixing control (standard DKD assumes a single target class).
- Structural pruning of a teacher down to ~4.8M (teachers are 5.9×–7.6× over budget).
- Dropout (flag exists, never used), SE blocks (one run, never scored), AugMix as part of
  full training (only tested as a finisher), other widths/depths at 600–1200 epochs.
- Training on more diverse inputs for KD fidelity, longer full-data training with new seeds
  (seed variance on public ≈ ±0.5), any method using test images for training (we believe
  this is not allowed by the rules).

## 14. What we would like from you

1. **Diagnosis**: given the 97.9% val vs ~91% public gap and the measurements in §7, what is the
   most likely cause, and what cheap test would distinguish the hypotheses?
2. **Up to 3 concrete experiments** most likely to raise the **hidden-test** score by ≥ 0.5 pt
   within a few GPU-hours, each with: exact change vs our recipe, matched control, expected effect
   size with evidence (and whether that evidence comes from CIFAR-10 or another regime), GPU-hours,
   and a keep/drop rule.
3. Whether any idea in §13 deserves priority, and why.
4. Anything in our protocol (§2) that could be misleading us.

Please respect the rules in §1 (≤5M params total at inference, from scratch, provided training data
only), do not propose things already tested in §6–§10 without saying why our test was flawed, and
mark clearly what is evidence vs speculation.
