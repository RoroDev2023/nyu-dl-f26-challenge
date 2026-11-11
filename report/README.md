# Report

AAAI 2026 two-column report for the CIFAR-10 5M-parameter challenge.
Team **P.R.**: Philip Vetsas (pmv264) and Rahim Askarov (ra3716).

Current draft: **8 pages**, non-anonymous, compiled with the official AAAI 2026
camera-ready style (`aaai2026.sty`, `aaai2026.bst`). The `\nocopyright` command is
used because this is a course report, not an AAAI proceedings paper.

## Build

```bash
cd report
./build.sh
```

Requires `latexmk`, `pdflatex`, and `bibtex` (TeX Live). The compiled PDF is
`main.pdf`.

## Layout

| Path | Contents |
|---|---|
| `main.tex` | Preamble, title block, abstract, section inputs, bibliography |
| `sections/` | One file per section |
| `references.bib` | Bibliography (16 entries, all cited) |
| `figures/` | Generated PDFs/PNGs plus `make_figures.py` and source images |
| `data/` | Archived logs and JSON used by the figures; provenance in `data/README.md` |
| `aaai2026.sty`, `aaai2026.bst` | Official AAAI 2026 style files |

## Figures

Regenerate all figures with the `ml` environment:

```bash
~/miniforge3/envs/ml/bin/python figures/make_figures.py
```

The script reads only `data/` and `figures/source/`; it does not need the
competition data or any checkpoint. The Track A logs (`student_baseline`,
`student_tricks_300`, `kd_r3_64_2t_600`, `A4_r3_80_600`, `V2_r3_80_1200`) and the
C60 finisher logs are now included and drive Figure 2(a) and 2(c). Still missing:
`B_augmix_300_sam` (the B4 seed-0 curve), the completed B5 logs, and the optional
`F300_lskd`/`W40_*` logs.

## Data provenance and the corrected shift story

Numbers come from two places:

1. **Track A** (accuracy, KD, architecture, continuations): `origin/main`'s
   `EXPERIMENTS.md` at commit `e240728` (copied to `data/EXPERIMENTS_main.md`),
   plus the per-epoch logs copied from the same commit.
2. **Track B** (augmentation, proxies): `STATUS.md`, `runs/analysis/B4_paired.md`,
   `runs/analysis/B5_provisional.md`, and the JSON files copied into `data/`.

The new 2026-11-06 material (fresh ASAM, V300 sweep, AugMix+JSD finisher, and the
F2s1+ASAM submission) is sourced from `data/EXPERIMENTS_main.md` and
`data/PROJECT_REPORT_FOR_REVIEW.md`; the corresponding curves come from the C60
logs. The J80/V300 run logs themselves were not in the push, so those numbers are
table/text only.

**Important correction.** The 2,000 unlabeled images distributed with the starter
data were **not** the final evaluation images. Earlier repo documents describe
them as "the eval" and conclude that the hidden set is new natural photographs;
the report does not repeat that claim. Section 5 uses the sample only as an
exploratory proxy, and the caption of Figure 4 says so explicitly. The final
leaderboard gap is treated as unexplained; the report says the released sample
cannot adjudicate its cause.

Final result (team-provided, 2026-11-11): private score **0.880**, **13th place**;
selected submissions `F2s1_tta5` (0.911 public) and `kd2t` (0.904 public). The
private score is the team's best of the two selected submissions, and the report
does not attribute it to a single model.

## Source of the numbers

- `data/logs/B_trivial_300_sam.csv`, `B_augmix_300_sam_s1.csv`: B4 validation
  curves (the seed-0 AugMix log was not preserved).
- `data/logs/B_kd3t_trivial.csv`, `B_kd3t_augmix.csv`: provisional B5 curves
  (163 and 178 of 300 epochs).
- `data/shift_B4_300.json`, `shift_B4_seeds.json`: B4 shifted accuracies.
- `data/shift_B3.json`: 150-epoch augmentation pilot (the `extra affine+color`
  point was logged only in `STATUS.md`, so it is hardcoded in the script).
- `data/verify_shift.json`: photometric and nearest-neighbor statistics for the
  released sample.
- `data/B4_paired.md`, `B5_provisional.md`: paired McNemar tests and the
  photographic proxies.

## Status / open items

- B5 is provisional (200/300 epochs; logs stop at 163/178). It is labeled as
  such everywhere.
- The B4 seed-0 curve and the completed B5 logs are still missing; Figure 2(b)
  uses seed 1 (caption says so) and B5 is marked provisional.
- J80/V300 logs were not in Rahim's push, so those results are table/text only.
- No checkpoints or competition data are needed to rebuild the current draft.
