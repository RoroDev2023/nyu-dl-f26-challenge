# Report

AAAI 2026 two-column report for the CIFAR-10 5M-parameter challenge.
Team **P.R.**: Philip Vetsas (pmv264) and Rahim Askarov (ra3716).

Current draft: **8 pages**, non-anonymous, compiled with the official AAAI 2026
camera-ready style (`aaai2026.sty`). The `\nocopyright` command is used because
this is a course report, not an AAAI proceedings paper. There is intentionally no
bibliography: the report is a record of our own runs, not a literature review.

## Build

```bash
cd report
./build.sh
```

Requires `latexmk` and `pdflatex` (TeX Live). The compiled PDF is `main.pdf`.

## Layout

| Path | Contents |
|---|---|
| `main.tex` | Preamble, title block, abstract, section inputs |
| `sections/` | One file per section |
| `figures/` | Generated PDFs/PNGs plus `make_figures.py` |
| `data/` | Archived logs and JSON used by the figures; provenance in `data/README.md` |
| `aaai2026.sty` | Official AAAI 2026 style file |

## Figures

Regenerate all figures with the `ml` environment:

```bash
~/miniforge3/envs/ml/bin/python figures/make_figures.py
```

The script reads only `data/logs/` and the archived `data/*.json`. It does not
need the competition data or any checkpoint. The report figures are:

1. teacher/student holdout bars
2. 150-epoch augmentation pilot scatter
3. six-panel validation curves: (a) main recipe, (b) architecture sweep,
   (c) KD settings, (d) 60-epoch finishers, (e) fresh 300-epoch alternatives,
   (f) provisional KD+AugMix
4. B4 clean/shifted bars and AugMix-minus-control deltas
5. 5-crop-minus-flip TTA deltas
6. public leaderboard submissions

The B4 figure uses both seed-0 and seed-1 bars from `shift_B4_seeds.json`; the
B5 panel is provisional.

## Data provenance

Numbers come from two places:

1. **Track A** (accuracy, KD, architecture, continuations): `origin/main`'s
   `EXPERIMENTS.md` at commit `e240728` (copied to `data/EXPERIMENTS_main.md`),
   plus the per-epoch logs copied from the same commit.
2. **Track B** (augmentation, proxies): `runs/analysis/B4_paired.md`,
   `runs/analysis/B5_provisional.md`, and the JSON files copied into `data/`.

The late 2026-11-06 material (fresh ASAM, V300 sweep, AugMix+JSD finisher, and the
F2s1+ASAM submission) is sourced from `data/EXPERIMENTS_main.md` and
`data/PROJECT_REPORT_FOR_REVIEW.md`; the J80/V300 logs themselves were not in the
push, so those numbers are table/text only.

The report deliberately does **not** include an analysis of the 2,000 unlabeled
sample images. Those images were not the final evaluation images and the earlier
shift narrative is not used.

Final result (team-provided, 2026-11-11): private score **0.880**, **13th of 81**;
selected submissions `F2s1_tta5` (0.911 public) and `kd2t` (0.904 public). The
private score is the team's best of the two selected submissions, and the report
does not attribute it to a single model.

## Source of the numbers

- `data/logs/B_trivial_300_sam.csv`, `B_augmix_300_sam_s1.csv`: B4 seed logs
  (retained for provenance; the B4 bars use `shift_B4_seeds.json`).
- `data/logs/B_kd3t_trivial.csv`, `B_kd3t_augmix.csv`: provisional B5 curves.
- `data/logs/{student_baseline,student_tricks_300,kd_r3_64_2t_600,
  A4_r3_80_600,V2_r3_80_1200}.csv`: main recipe curves.
- `data/logs/{sweep_r3_64,sweep_r3_80,sweep_r3_96,A3_T2,A3_T1_a1}.csv`:
  architecture and KD sweeps.
- `data/logs/C60_*.csv`: 60-epoch finishers.
- `data/logs/{W40_s0,W40_s1,F300_lskd,S300_asam}.csv`: failed fresh alternatives.
- `data/shift_B4_seeds.json`: B4 shifted accuracies.
- `data/shift_B3.json`: 150-epoch augmentation pilot (the `extra affine+color`
  point was logged only in `STATUS.md`, so it is hardcoded in the script).
- `data/B4_paired.md`, `B5_provisional.md`: paired McNemar tests and the
  photographic proxies.

## Status / open items

- B5 is provisional (200/300 epochs; logs stop at 163/178).
- The completed B5 logs are still missing.
- J80/V300 logs were not in Rahim's push, so those results are table/text only.
- No checkpoints or competition data are needed to rebuild the current draft.
