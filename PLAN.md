# Team plan and work trajectory

Updated 2026-11-02. This is the durable memory for the collaboration. `AGENTS.md`
has the commands; `STATUS.md` has the current numbers.

## Context

- Kaggle competition: CIFAR-10, ≤5M trainable params, trained from scratch on the
  official 50k train split.
- The hidden eval is **distribution-shifted**: ~2,000 new natural photographs
  (CIFAR-10.1-like), not corrupted CIFAR-10. Clean 5k validation is only a proxy;
  the observed Kaggle drop is ~7–10pp.
- **This repo is now the shared codebase.** Our old code/artifacts are archived in
  `../proj1/our_code_backup/`; old checkpoints are incompatible with
  `models.load_model` (this repo stores `config` + `model`/`ema`).
- Team split:
  - **Track A (teammate, HPC A100):** distillation + architecture. Current best:
    97.76 clean val → **0.904 public** with a 5M student distilled from 2 teachers
    (28M PyramidNet + another teacher). "All tricks" baseline is 97.36.
  - **Track B (us):** characterize the shift and find augmentations that survive it.

## What we established (B1/B2)

Report: `runs/analysis/shift.md`; montage: `runs/analysis/shift_montage.png`;
corruption gallery: `runs/analysis/corruption_gallery.png`.

- 0/2,000 exact CIFAR-10 matches. Eval is brighter, higher contrast/sharpness,
  lower blockiness (not JPEG); visually new photos with different framing,
  backgrounds, and object scale.
- Synthetic corruptions are stress tests, not the shift. `combo` matches the size
  of the drop (~9.1pp) but not the mechanism.
- Natural-photo robustness is the lever: AugMix+JSD, RandAugment, CutMix/MixUp,
  scale/translation/rotation, color/lighting jitter, longer schedules, ensembles.
  Noise/blur/JPEG augmentation is not the priority.
- Old-repo result: best single-teacher KD student 96.24 val; best 5M ensemble
  96.50 val / **0.864 public**. Teammate's 2-teacher KD: 97.76 val / 0.904.

## Track B next steps (ours)

1. **Port B1/B2 to this repo.** B1 is data-only (`data.load_train`, `load_test`).
   B2 needs fresh checkpoints from this repo (old checkpoints won't load). Build a
   shifted val from the seed-0 5k split with natural-photo corruptions and a
   `B_...` run table.
2. **B3 — natural-photo augmentation ablations** on `r3_64_3-5-3`, no KD, 150–200
   epochs so runs are cheap. Compare:
   - `--autoaug randaug` vs `trivial` vs `autoaug_cifar`
   - `--cutmix-alpha 1.0 --mixup-alpha 0.4 --mix-prob 1.0` vs `0.5`
   - `--cutout 8` and combinations
   - new transforms (scale/translation/rotation/color) added as flags
   Select by clean val + shifted val, with a clean-val drop ≤0.3.
3. **B4 — merge the winner as a flag** in this repo and tell Track A.
4. **B5 — one 3-teacher KD run with the winning augmentation** vs without.

## Track A (teammate, for context)

- A1 architecture sweep (5 shapes × 300 epochs × 3 teachers), pick 1–2.
- A2 2 vs 3 teachers.
- A3 KD settings: temp 1/2/4, alpha 0.7/0.9/1.0, mix-prob 0.5 vs 1.0.
- A4 600 vs 1000 epochs.
- A5 retrain robust teachers once Track B finds the augmentation.
- A6 final: best config, `--val-size 0` (all 50k), 2 seeds, EMA, submit.

## Shared rules

- One codebase; pull before launching runs; new tricks go in as flags on a branch.
- Always use the seed-0 5k split; teachers and students must hold out the same
  images.
- Run names start with `A_` / `B_`.
- Submissions: 5/day total (2 ours, 2 theirs, 1 spare). Submit only when clean or
  shifted val clearly improved; state val in the submission message.
- Final pick = 2 submissions selected on Kaggle, both single ≤5M models, chosen by
  val + shifted val, not public LB.
- Keep logs and checkpoints on disk; staff may ask for them.

## Open questions for the teammate

- Was the 5M limit enforced on teachers, or only on the submitted student?
- Which teacher checkpoints and "all tricks" flags produced the 97.36/97.76 runs?
- What exactly is the `summarize.py` schema/flow they use, and which branch?
- What augmentations are in "all tricks" (autoaug, mix, cutout, SAM, EMA)?
