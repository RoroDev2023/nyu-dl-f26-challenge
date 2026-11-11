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
competition data or any checkpoint. Optional per-epoch logs dropped into
`data/logs/` are picked up automatically:

- `A4_r3_80_600.csv`, `V2_r3_80_1200.csv`, `kd_r3_64_2t_600.csv`
  → `fig_track_a_curves`
- `C60_sam.csv`, `C60_asam.csv`, `C60_swa.csv`, `C60_lskd.csv`, `C60_a045.csv`
  → `fig_failed_curves`

These are not included in the current draft because the logs were not in the
checkout. If Rahim adds them, rerun the script and add the figures to the
corresponding sections.

## Data provenance and the corrected shift story

Numbers come from two places:

1. **Track A** (accuracy, KD, architecture, continuations): the merged
   `EXPERIMENTS.md` on `b_shift` (includes `origin/main` commit `b4f77ac`).
2. **Track B** (augmentation, proxies): `STATUS.md`, `runs/analysis/B4_paired.md`,
   `runs/analysis/B5_provisional.md`, and the JSON files copied into `data/`.

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
- Track A and failed-continuation validation curves are missing; the report uses
  tables for those results. Add the logs and regenerate to strengthen Section 6.
- No checkpoints or competition data are needed to rebuild the current draft.
