# Data provenance

The report is self-contained: everything here was copied from the shared repo's
archived runs/analysis on 2026-11-11.

| File | Source | Notes |
|---|---|---|
| `logs/B_trivial_300_sam.csv` | `runs/B_trivial_300_sam/log.csv` | 300 epochs, seed 0 |
| `logs/B_augmix_300_sam_s1.csv` | `runs/B_augmix_300_sam_s1/log.csv` | 300 epochs, seed 1; the seed-0 log was not preserved |
| `logs/B_kd3t_trivial.csv` | `runs/B_kd3t_trivial/log.csv` | stops at epoch 163/300 |
| `logs/B_kd3t_augmix.csv` | `runs/B_kd3t_augmix/log.csv` | stops at epoch 178/300 |
| `shift_B4_300.json` | `runs/analysis/B4_300/B_shift.json` | B4 clean/shifted accuracies, no TTA |
| `shift_B4_seeds.json` | `runs/analysis/B4_seeds/B_shift.json` | B4 two-seed comparison, no TTA |
| `shift_B3.json` | `runs/analysis/B_shift.json` | 150-epoch augmentation pilot |
| `verify_shift.json` | `runs/analysis/verify_shift.json` | photometric stats + NN percentiles for the released sample |
| `B4_paired.md` | `runs/analysis/B4_paired.md` | paired McNemar tests, photographic proxies |
| `B5_provisional.md` | `runs/analysis/B5_provisional.md` | KD arms at epoch 200/300 |

`figures/source/shift_montage.png` and `corruption_gallery.png` are the archived
`runs/analysis` images. `make_figures.py` crops the montage's baked-in labels and
replaces them with labels that say the sample is not the final evaluation set.

## Final result

Team-provided (Philip, 2026-11-11): private score 0.880, 13th place. Selected
submissions: `F2s1_tta5` (0.911 public) and `kd2t` (0.904 public). The private
score is the better of the two selected submissions; the report does not
attribute it to one model.

## Missing / optional

Not in this checkout, but small enough to add to git if Rahim can:

- `runs/B_augmix_300_sam/log.csv` (completes the B4 seed-0 curve)
- completed `runs/B_kd3t_{trivial,augmix}/log.csv` and `results.json`
- `runs/A4_r3_80_600/log.csv`, `runs/V2_r3_80_1200/log.csv`,
  `runs/kd_r3_64_2t_600/log.csv`
- `runs/C60_*/log.csv`, `runs/F300_lskd/log.csv`, `runs/W40_s0/log.csv`,
  `runs/W40_s1/log.csv`

Drop them into `logs/` with the names listed in the report README and rerun
`figures/make_figures.py`.
