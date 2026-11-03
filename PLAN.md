# Team plan and work trajectory

Updated 2026-11-03. This is the durable memory for the collaboration. `AGENTS.md`
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
- README rules (now confirmed): the final submission is **one ≤5M network — no
  ensembles**; teachers may be any size and must be trained from scratch on the
  same train data (`--no-budget`). No test-time BN adaptation or pseudo-labels.
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

## Team recipe (from README, 2026-11-02)

- All tricks: `--autoaug trivial --cutout 16 --mixup-alpha 0.2 --cutmix-alpha 1.0 --mix-off-epochs 15 --sam --ema-decay 0.999`.
- Teachers (400 epochs, same aug + `--mix-off-epochs 20`): `wrn_28_10 --no-budget` seeds 0/1 (with SAM), `pyramid_110_270` seed 2 (no SAM).
- KD (600 epochs): `r3_64_3-5-3`, all tricks, `--teachers wrn_s0 pyramid_s2 --kd-alpha 0.9 --kd-temp 4`.
- Results: baseline 94.76, all tricks 97.36, 2-teacher KD **97.76 / 0.904 public**; teachers 98.14/98.08/97.66.
- A100 timings/epoch: baseline 11s, tricks+SAM 19s, KD 2T ~30s, WRN-28-10 29s, PyramidNet 45s.

## Track B next steps (ours)

Updated 2026-11-03: B1–B4 are done; B5 is blocked on teacher checkpoints.

1. ~~Port B1/B2 to this repo.~~ **Done:** `scripts/shift_analysis.py` (B1 stats +
   montage, B2 corrupted seed-0 5k holdout). Report `runs/analysis/shift.md`.
2. ~~B3 — natural-photo augmentation ablations.~~ **Done (150 ep, no SAM):** AugMix
   (`--autoaug augmix`, one view) wins: clean TTA 96.52 vs 96.48 control, combo
   shifted 87.42 vs 84.32 (+3.10). Geometry-only trades 0.50 clean for +0.64
   shifted; RandAugment, AutoAugment-CIFAR and `--extra-aug` all lost.
3. ~~B4 — merge the winner as a flag.~~ **Done:** `--autoaug augmix` (plus
   `--geo-aug`, `--extra-aug`) on `b_shift`. The 300-epoch + SAM recipe
   (`B_augmix_300_sam`) gives **97.06 val / 97.04 EMA+TTA, 89.02 combo** vs 84.32
   for the control — clean +0.8, shift gap 11.96pp → 8.08pp. 4.92M params.
   Control run `B_trivial_300_sam` (team "all tricks", no AugMix) is running to
   separate the AugMix gain from the longer schedule.
4. **B5 — one 3-teacher KD run with the winning augmentation** vs without.
   **Blocked:** needs `teacher_wrn_s0/s1` + `teacher_pyr_s2` in this repo's
   `config`+`ema` format. Do not train a 36.5M teacher on Modal — A5 covers that on
   the A100s. Meanwhile the cheap proxy is KD from whichever student is best, but a
   teacher at ~97 val cannot teach a 97.06 student much.

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
- Final submission = **one ≤5M network, no ensembles**. Keep 2 candidate single
  models, chosen by val + shifted val, not public LB.
- Keep logs and checkpoints on disk; staff may ask for them.

## Open questions for the teammate

- Which teacher checkpoints and "all tricks" flags produced the 97.36/97.76 runs?
- What exactly is the `summarize.py` schema/flow they use, and which branch?
- What augmentations are in "all tricks" (autoaug, mix, cutout, SAM, EMA)?
- Can they share the current `hpc_data.zip` / confirm our `train_cache.npz` matches?
