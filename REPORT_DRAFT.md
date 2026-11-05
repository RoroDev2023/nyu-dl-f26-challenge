# Report draft — method and setup sections

Working draft for the course report. Numbers come from `EXPERIMENTS.md` (Track A) and
`runs/analysis/` (Track B); nothing here is a conclusion we have not measured. Sections
still to write: results, discussion, per-figure captions.

## 1. Task and constraints

CIFAR-10 classification (10 classes, 32×32 RGB) with a hard budget of **5M trainable
parameters**, trained **from scratch on the permitted CIFAR-10 training data only**.
No pretrained weights and no external datasets; data augmentation of the permitted
training images is allowed; all preprocessing and training must be reproducible
(`overview.md`).

We read the budget as applying to the submitted network. Two consequences matter for the
method: (a) every evaluated student is ≤5M (4.88M-4.92M), and (b) the knowledge-
distillation teachers are 28.5M-36.5M, which is only defensible if the budget constrains
the submitted model rather than every model trained along the way. That reading is
unresolved in the official text and is flagged as such in the discussion.

The evaluation set is provided without labels and may not be used for training,
fine-tuning, or "unauthorized model selection". We only computed unlabeled summary
statistics from it (Section 6) and never trained on it or selected models by it.

## 2. Data and splits

- 50,000 labeled training images. Fixed stratified **5,000-image holdout** (split seed 0),
  used by every student and teacher in this work. Teachers must hold out the same images
  as the student, otherwise teacher-trained-on-val information leaks into the student
  through the soft targets; we verified from the teacher checkpoint metadata that all
  three teachers used `val_size=5000`.
- Final models are retrained on all 50,000 images (`val_size=0`) once the recipe is
  fixed, and that model is what we submit — its validation number is not comparable, so
  selection uses the holdout model.
- Augmentation is the only source of extra variation (no pseudo-labels, no external
  data). The unlabeled evaluation images are used only for descriptive statistics.

## 3. Architectures

| Role | Preset | Params | Notes |
|---|---|---|---|
| Student (baseline) | `r3_64_3-5-3` | 4.92M | 3 stages, 64/128/256 channels, 3/5/3 blocks, pre-activation bottleneck ResNet |
| Student (selected) | `r3_80_2-3-3` | 4.88M | widest first stage, 2/3/3 blocks; best of a 5-shape sweep at 300 epochs |
| Teacher | `wrn_28_10` | 36.5M | WideResNet-28-10, seeds 0 and 1 |
| Teacher | `pyramid_110_270` | 28.5M | PyramidNet-110-270 |

Parameter counts are asserted in code: `train.py`/`predict.py` refuse any student over
5,000,000 parameters (`--no-budget` exists only for teacher runs).

## 4. Training recipe

Optimization: SGD with Nesterov momentum, base LR 0.1 (batch 128), weight decay 5e-4
excluded from BatchNorm affine parameters, 5-epoch warmup followed by cosine decay,
label smoothing 0.1, bf16 autocast, EMA of the weights (decay 0.999) reported separately
from the raw weights.

Regularization and augmentation ("all tricks"): TrivialAugment, Cutout 16, mixup
(α=0.2) or CutMix (α=1.0) on each batch with probability 1.0, mixing disabled for the
last 15 epochs, and Sharpness-Aware Minimization (SAM, ρ=0.05; each step costs two
forward/backward passes). ASAM follows the authors' formulation (perturbation
`T²g/‖Tg‖` with `T=|w|+η` applied to weight tensors only).

Schedules used: 300-epoch runs for ablations, 600 epochs for the main students,
1200 epochs for the final full-data model. Longer schedules were worth ~0.2-0.5 points
of validation accuracy and were the single largest "trick" in this project.

## 5. Knowledge distillation

Students are trained with a KL divergence to the average of 2-3 teacher distributions,
temperature T=4, loss `0.9 · KL(T) · T² + 0.1 · CE`, where the CE term uses the hard
labels with mixup/CutMix-consistent soft targets. Teachers see the same augmented and
mixed batches as the student. Label smoothing is disabled while distilling.

We also implemented logit-standardized KD (LS-KD, Sun et al. 2024): each teacher's logit
vector is z-scored before softening, the ensemble is averaged in that space, and the
student's logits are standardized identically; the loss becomes
`0.1 · CE + 9 · KL(T) · T²`. With 3 teachers this replaces the α weighting and was
evaluated as a matched A/B against the ordinary KD loss.

Distillation gains: 97.36 (all tricks, no teachers) → 97.74-97.76 (2-3 teachers,
600 epochs), with a 4.88M student reaching 97.84 at 600 epochs and 97.92 under 5-crop
TTA — the best holdout number in the project, and better than the 28.5M teacher it
distils from (97.64).

## 6. The distribution shift (Track B)

The hidden evaluation set is **not** corrupted CIFAR-10. We verified this three ways:

1. **No duplicates**: 0 of 2,000 evaluation images are byte-identical to any of the
   50,000 training images, and all training images are unique.
2. **Photometric differences are real but negligible**: on all 50k training images
   versus the 2,000 evaluation images, sharpness (+0.0053, Cohen's d 0.27), contrast
   (d 0.17), brightness (d 0.14) and blockiness (d −0.07) differ; saturation and
   colourfulness do not. Effects this small are what a different set of photographs
   produces from composition alone, so they are reported as observations, not evidence.
3. **Nearest-neighbour test** (`scripts/verify_shift_b1.py`): matching each evaluation
   image against all 50k training images under all 8 dihedral symmetries, and separately
   under 5 small translations with 8×8 average pooling, the evaluation set sits at or
   above the distance distribution of *genuine* CIFAR pairs (dihedral: median 1982 vs
   1852; translation-tolerant: 79.9 vs 73.5), and **0 of 2,000** fall below the 1st
   percentile. A brightness/contrast transform, re-JPEG, flip/rotation or shift-and-crop
   of a CIFAR image would collapse that residual towards zero; none does. The evaluation
   images are different content — new photographs, with different framing, backgrounds
   and object scale.

Consequence for method: a corrupted validation split is a stress test, not a model of
the shift. We therefore report clean accuracy *and* shifted accuracy on label-free
proxies built from the training data, and we do not select models by a proxy whose
correspondence to the hidden set is unproven. The knock-on finding is that matching the
*size* of the observed public drop (≈9-10 points) is not evidence about its mechanism —
an earlier version of this analysis made that inference from a corruption that happened
to produce a similar drop, and we retracted it.

Augmentation experiment (B3/B4, 300 epochs, matched control): adding AugMix (one view,
severity 3, no JSD term) leaves clean accuracy unchanged (EMA 97.08 for both arms) while
improving every image-quality proxy — +2.18 on the composite corruption proxy
(89.02 vs 86.84, McNemar p = 3.6e-9), +4.1 at Gaussian noise, +5.3 at JPEG-30, and
+1.1 to +1.4 on a photographic-jitter proxy (brightness/contrast/saturation/zoom/
rotation) that is closer to "same classes, different camera". Geometry alone
(±3px translation, 1.2× zoom) is where AugMix does nothing, which supports the reading
that the difficulty is content and lighting rather than framing.

## 7. Evaluation protocol

- **Holdout**: 5,000 images, seed 0, never trained on; EMA weights unless stated.
- **TTA**: horizontal flip (2 views) as the default. A 10-view procedure (reflect-pad 4,
  centre plus four 2-pixel shifts, each with and without flip, softmax averaged) was
  also implemented and scored. In paired tests it added nothing beyond the flip on clean
  data or on geometry, and at most ~0.5-1.2 points on JPEG/corruption proxies, at 10×
  inference cost; it is not used for the final model.
- **Paired statistics**: every model comparison is a McNemar test on per-image
  correctness over the same 5,000 images, not a difference of two accuracies. We report
  discordant-pair counts and exact binomial p-values. We treat a ~0.15-point difference
  as noise (the seed-to-seed spread on this holdout is 0.2 points on clean, 0.4 on the
  composite proxy).
- **Shifted proxies** (from training data only): Gaussian noise, Gaussian blur, JPEG,
  contrast, colour, a composite ("combo"), plus photographic jitter and geometric
  jitter as described in Section 6.
- **Public leaderboard**: 1,000 images, so roughly ±1.1 points of noise at 90% accuracy.
  Never used to choose between models that are close.

## 8. Reproducibility and compute

- Every run writes `args.json`, per-epoch `log.csv` (including separate CE and KD loss
  components), `results.json` (raw/EMA, with and without TTA) and a resumable `last.pt`
  (model, EMA, optimizer, scheduler, RNG state). Resuming refuses to continue a folder
  whose saved settings differ from the current flags.
- Split seed, augmentation seeds and model seeds are explicit; no run depends on
  unrecorded state.
- Compute: HPC A100 (main runs) and cloud A10/A10G (ablations). A 300-epoch student at
  4.9M parameters takes ~19-21 s/epoch on an A10-class GPU; 3-teacher distillation at
  300 epochs takes ~59 s/epoch.

## 9. Rules compliance

- No pretrained weights, no external data, no test-set labels; teachers trained from
  scratch on the permitted training split.
- All submitted students are ≤5M trainable parameters (4.88M / 4.92M).
- Ensemble prediction is **not** used: the official text does not address ensembles, but
  the 5M budget is stated for "a modified ResNet", and a two-model ensemble of the kind
  we measured (+0.34 points on the composite proxy, p = 0.15) is inside public-LB noise
  and not worth an admissibility argument.
