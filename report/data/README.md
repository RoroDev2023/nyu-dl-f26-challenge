# Data provenance

The report is self-contained: every number comes from a log or a paired-test
table copied from the shared repo on 2026-11-11. There is no analysis of the
2,000 unlabeled sample images.

| File | Source | Notes |
|---|---|---|
| `logs/B_trivial_300_sam.csv` | `runs/B_trivial_300_sam/log.csv` | 300 epochs, seed 0 |
| `logs/B_augmix_300_sam_s1.csv` | `runs/B_augmix_300_sam_s1/log.csv` | 300 epochs, seed 1; seed-0 log not preserved |
| `logs/B_kd3t_trivial.csv` | `runs/B_kd3t_trivial/log.csv` | stops at epoch 163/300 |
| `logs/B_kd3t_augmix.csv` | `runs/B_kd3t_augmix/log.csv` | stops at epoch 178/300 |
| `logs/student_baseline.csv` | `runs/student_baseline/log.csv` | 200 ep, raw val only |
| `logs/student_tricks_300.csv` | `runs/student_tricks_300/log.csv` | 300 ep |
| `logs/kd_r3_64_2t_600.csv` | `runs/kd_r3_64_2t_600/log.csv` | 600 ep |
| `logs/A4_r3_80_600.csv` | `runs/A4_r3_80_600/log.csv` | 600 ep |
| `logs/V2_r3_80_1200.csv` | `runs/V2_r3_80_1200/log.csv` | 1,200 ep |
| `logs/sweep_r3_64.csv`, `sweep_r3_80.csv`, `sweep_r3_96.csv` | `runs/A_sweep_kd3t_300_*/log.csv` | architecture sweep, 300 ep |
| `logs/A3_T2.csv`, `A3_T1_a1.csv` | `runs/A3_r3_80_*/log.csv` | KD-temperature/weight sweep |
| `logs/C60_sam.csv`, `C60_asam.csv`, `C60_swa.csv`, `C60_lskd.csv`, `C60_a045.csv` | `runs/C60_*/log.csv` | 60-epoch continuations from A4 |
| `logs/W40_s0.csv`, `W40_s1.csv` | `runs/W40_*/log.csv` | WRN-40-2 pair, 300 ep |
| `logs/F300_lskd.csv` | `runs/F300_lskd/log.csv` | fresh LS-KD, 300 ep |
| `logs/S300_asam.csv` | `runs/S300_asam/log.csv` | fresh ASAM, stopped at 268 ep |
| `EXPERIMENTS_main.md` | `origin/main:EXPERIMENTS.md` (commit `e240728`) | Rahim's latest run log |
| `PROJECT_REPORT_FOR_REVIEW.md` | `origin/main:PROJECT_REPORT_FOR_REVIEW.md` | pre-deadline memo; source for J80/V300/S300 results |
| `shift_B4_300.json` | `runs/analysis/B4_300/B_shift.json` | B4 clean/shifted accuracies, no TTA |
| `shift_B4_seeds.json` | `runs/analysis/B4_seeds/B_shift.json` | B4 two-seed comparison, no TTA |
| `shift_B3.json` | `runs/analysis/B_shift.json` | 150-epoch augmentation pilot |
| `B4_paired.md` | `runs/analysis/B4_paired.md` | paired McNemar tests, photographic proxies, TTA deltas |
| `B5_provisional.md` | `runs/analysis/B5_provisional.md` | KD arms at epoch 200/300 |

The Track A, C60, W40, F300, and S300 logs were added from Rahim's `main` branch
(commit `e240728`). The J80/V300 runs were not pushed as logs, so their numbers
come from `EXPERIMENTS_main.md` and `PROJECT_REPORT_FOR_REVIEW.md`.

## Final result

Team-provided (Philip, 2026-11-11): private score 0.880, 13th place. Selected
submissions: `F2s1_tta5` (0.911 public) and `kd2t` (0.904 public). The private
score is the better of the two selected submissions; the report does not
attribute it to one model.

## Missing / optional

- `runs/B_augmix_300_sam/log.csv` (completes the B4 seed-0 curve)
- completed `runs/B_kd3t_{trivial,augmix}/log.csv` and `results.json`

Drop them into `logs/` with the names above and rerun
`figures/make_figures.py`.
