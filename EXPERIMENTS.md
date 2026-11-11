# Experiments log

Status of every run. Val = the fixed 5000-image holdout (`val_size=5000`, split seed 0,
same for every student and teacher). EMA weights unless noted. "5-crop" = the TTA in
`tta.py` (reflect-pad 4, plus-shaped 2px shifts x hflip, softmax averaged, 10 views) —
the exact procedure of the 0.911 submission (reproduced with 0 differences on 2026-11-05).

Big files (checkpoints, submissions) are not in git — see the shared Drive folder.
The 0.911 model + csv live in `KEEP/` and must not be overwritten.

## Results that ran

| Run | Config | Val flip | Val 5-crop | Public LB | Notes |
|---|---|---|---|---|---|
| baseline | r3_64_3-5-3, plain | 94.76 | — | — | flip TTA 95.42 |
| tricks, no KD | r3_64 + TrivialAug, Cutout16, mixup/CutMix, SAM, EMA | 97.36 | — | — | |
| kd2t | r3_64, 2 teachers, 600 ep | 97.74 | — | 0.904 (0.901 5-crop) | |
| kd 3 teachers | r3_64, 3 teachers, 600 ep | 97.76 | — | — | |
| arch sweep, 300 ep | r3_80_2-3-3 / r3_96_2-2-2 / r3_64_3-5-3 | 97.62 / 97.42 / 97.38 | — | — | r3_80 picked |
| A3 | r3_80 300 ep, KD T=2 / T=1 + alpha 1 | 97.62 / 97.50 | — | — | |
| A4 | r3_80, 600 ep, 3 teachers | 97.84 | 97.92 | — | |
| V2 | r3_80, 1200 ep | 97.82 | 97.94 | — | |
| F1 / F2 | r3_80 full data | — | — | 0.898 / 0.899 | |
| **F2s1** | r3_80, 1200 ep, full data, seed 1 | — | — | **0.911 (5-crop)** / 0.903 flip | `KEEP/` |
| FIN_ctrl | A4 + 40 ep, LR .003, TA + Cutout | 97.76 | 97.82* | — | *old corner-crop TTA, re-score |
| FIN_light | A4 + 40 ep, crop + flip only | 97.76 | 97.82* | — | same as ctrl, no gain |
| Train-split BN recalibration | A4 / V2, clean train images | 97.08 / 97.20 | — | — | hurts |
| Test-time BN adaptation | F2 | 97.34 (val) | — | 0.884 | hurts |
| Sinkhorn class balance | F2 | — | — | 0.898 | no gain |
| TTA weighting | center+flip at 50% / 80% vs equal 5-crop, tune/confirm halves, A4 + V2 | — | — | — | no gain beyond noise; keep equal 5-crop |

Teachers (val): wrn_28_10 s0 98.06, s1 98.12, pyramid_110_270 97.64. Best teacher pool
wrn s0 + s1 98.14. Teachers are 36.5M / 28.5M params (7.6x / 5.9x over a 4.8M target).

## Ran on 2026-11-05 (5-crop, full 5000 val, paired with a sign test)

TTA check: equal-weight 5-crop reproduced the 0.911 csv from KEEP/F2s1.pt with 0 differences.
Old `fin_eval` used corner crops, which explains the 97.82 vs 97.92 A4 gap. Center/flip
weighting (50% / 80%) vs equal 5-crop, picked on a tune half and checked on a confirm half:
no gain on A4 (+0/+1) or V2 (-1/-2) -> equal 5-crop kept.

60-epoch continuations from A4 (seed 0, LR .005 cosine, TA + Cutout16, no mixing,
3 teachers, EMA .999, `--gpu-aug`). Only the listed thing changes vs C60_sam.

| Run | Change | 5-crop (EMA) | vs C60_sam EMA / raw | vs A4 | Val NLL |
|---|---|---|---|---|---|
| A4 (untouched) | start point | 97.92 | | | 0.154 |
| C60_sam | control: SAM rho .05 | 97.82 | | -5 (p .27) | 0.154 |
| C60_asam | ASAM rho .5, eta .01 (authors' formulation) | 98.00 | +9 (p .049) / +11 (p .013) | +4 (p .50) | 0.153 |
| C60_a045 | ordinary KD alpha .45 | 97.96 | +7 (p .14) / +4 (p .39) | +2 (p .84) | 0.129 |
| C60_lskd | logit-standardized KD, T 2, 0.1 CE + 9 KL T^2 | 97.76 | -3 / -6 | -8 (p .13) | 0.086 |
| C60_swa | SWA: 6 x 10-ep cycles .01 -> .001, avg cycles 2-6, BN re-estimated | 97.80 | -1 / -1 | -6 | 0.153 |
| C60_swa (averaged BN buffers) | diagnostic | 97.78 | | -7 | 0.156 |

ASAM confirm from a second start point (kd2t, r3_64, val 97.74):

| Run | 5-crop (EMA) | vs C60_sam_kd2t EMA / raw | vs kd2t untouched |
|---|---|---|---|
| C60_sam_kd2t | 97.76 | | +1 |
| C60_asam_kd2t | 97.78 | +1 / +4 (p .34) | +2 (p .83) |

Pooled over both start points: ASAM vs SAM continuation raw +15 (p .006), EMA +10 (p .09);
ASAM vs the untouched start point +6 of 10,000 images (p .44). Reading: ASAM is better than
SAM for this late continuation, but the continuation itself does not improve on the starting
checkpoint. The same ASAM finish applied to F2s1 (F60_asam_F2s1) changes 33/2000 test labels;
it was NOT submitted, because the confirm did not show a gain over the untouched model.

Fresh 300-epoch runs (recipe of the A3 / arch-sweep runs: mix-off 15, TA, Cutout16,
mixup .2, CutMix 1.0, SAM, EMA, 3 teachers):

| Run | Params | Val flip (EMA) | Val 5-crop (EMA) | Compared with | Val NLL |
|---|---|---|---|---|---|
| F300_lskd (LS-KD, T 2, 0.1 CE + 9 KL T^2) | 4.88M | 97.50 (raw 97.62) | 97.62 | sweep r3_80 300 ep, ordinary KD: 97.62 flip (not paired, checkpoint on HPC) | 0.092 |
| W40_s0 (WRN-40-2) | 2.24M | 97.10 | 97.10 | | 0.178 |
| W40_s1 (WRN-40-2) | 2.24M | 97.08 | 97.06 | | 0.179 |
| W40 ensemble (equal prob average) | 4.49M total | | 97.26 | one 4.88M r3_80 at 300 ep (~97.6) | 0.176 |

WRN pair: errors 145 / 147, both wrong 111, Jaccard overlap 0.61. Averaging gains +0.16 over
one WRN, not enough to make up for each model being ~0.5 below one 4.88M ResNet. With this
recipe (tuned for r3_80, not for WRN-40-2) and one seed pair, splitting the budget into two
models lost to one model. LS-KD fresh: no accuracy gain over ordinary KD at 300 ep; much lower
val NLL (both continuation and fresh), i.e. a calibration effect, not an accuracy one.

## Untested

DKD, teacher pruning (structural, to ~4.8M; teachers are 5.9x / 7.6x over), AugMix,
dropout and stochastic depth (flags exist, every student ran with 0), SE blocks (A3_se never
scored), r4 / r3_40 sweep archs scored only at 300 ep, V3/F3 (r3_96_2-2-3, cancelled at boot),
ASAM / SWA / LS-KD as fresh 1200-ep full-data runs (only continuations / 300 ep were run).

## Regularization actually used

Weight decay 5e-4 (not on BN / biases), crop + flip, TrivialAugment, Cutout16, mixup + CutMix
(off for the last epochs), label smoothing 0.1 (only without KD), KD soft targets, EMA.
SAM / ASAM / SWA are reported as optimization methods.

## Rules we use

- A method counts only if it beats its matched control and A4 on the full val under 5-crop
  in a paired comparison, with raw and EMA agreeing; then it is repeated from another start
  point or seed before it is trusted.
- Every new submission sits next to the 0.911 one; nothing replaces `KEEP/`.
- The teacher-student gap and the train/test differences we measured (test images brighter,
  lower confidence, lower student-teacher agreement) are observations. Explanations for them
  are hypotheses, not established facts about the hidden test set.
