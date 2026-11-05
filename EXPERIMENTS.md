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

## Running / queued (2026-11-05)

All continuations: from A4 EMA, seed 0, 60 ep, TA + Cutout16, no mixing, 3 teachers,
EMA .999, `--gpu-aug`, warmup 0. Only the listed thing changes.

| Run | Change vs C60_sam | Status |
|---|---|---|
| C60_sam | control: SAM rho .05, LR .005 cosine | running |
| C60_asam | ASAM rho .5, eta .01 (authors' formulation) | running |
| C60_swa | 6 x 10-ep cycles LR .01 -> .001, average raw weights of cycles 2-6, BN re-estimated on train split with TA + Cutout | running |
| C60_lskd | logit-standardized KD, T 2, loss 0.1 CE + 9 KL T^2 | queued |
| C60_a045 | ordinary KD alpha .45 | queued |
| F300_lskd | fresh 300 ep, LS-KD, vs arch-sweep r3_80 300-ep baseline (97.62) | not started |
| W40_s0 + W40_s1 | 2 x WRN-40-2 (2.24M each, 4.49M together), fresh 300 ep, ensemble | not started |

## Untested

DKD, teacher pruning (structural, to ~4.8M), AugMix, SE blocks (A3_se never scored),
r4 / r3_40 sweep archs, V3/F3 (r3_96_2-2-3, cancelled at boot).

## Rules we use

- A method counts only if it beats its matched control and A4 on the full val under 5-crop
  in a paired comparison, with raw and EMA agreeing; then it is repeated from another start
  point or seed before it is trusted.
- Every new submission sits next to the 0.911 one; nothing replaces `KEEP/`.
- The teacher-student gap and the train/test differences we measured (test images brighter,
  lower confidence, lower student-teacher agreement) are observations. Explanations for them
  are hypotheses, not established facts about the hidden test set.
