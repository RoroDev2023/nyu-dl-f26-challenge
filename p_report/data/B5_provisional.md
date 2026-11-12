# B5 provisional: KD arms at epoch 200/300 (2026-11-06)

Both arms identical except `--autoaug`: r3_80_2-3-3, 300 ep, 3 teachers
(wrn_s0 + wrn_s1 + pyr_s2), KL a=0.9 T=4, TrivialAug-equivalent control, SAM, EMA,
seed 0. Interrupted twice (Modal billing limit, then laptop) and resumed from the
epoch-160 and epoch-200 checkpoints; the pairs below are at the same epoch.

EMA weights, flip TTA, seed-0 5k holdout.

| condition | trivial | augmix | A4 (600 ep, trivial) |
|---|---|---|---|
| clean | 96.98 | 96.86 | 97.84 |
| combo | 87.06 | 89.26 | 89.20 |
| photo_strong | 91.14 | 93.06 | 94.78 |
| shift3 | 96.06 | 96.42 | 97.68 |

Paired McNemar (same 5000 images):

| comparison | condition | net | fixes/intros | p |
|---|---|---|---|---|
| augmix vs trivial | clean | -0.12 | 33/39 | 0.56 |
| augmix vs trivial | combo | **+2.20** | 182/72 | 3.7e-12 |
| augmix vs trivial | photo_strong | **+1.92** | 147/51 | 5.6e-12 |
| augmix vs trivial | shift3 | +0.36 | 53/35 | 0.069 |
| augmix vs A4 | clean | -0.98 | 10/59 | 1.4e-09 |
| augmix vs A4 | combo | +0.06 | 126/123 | 0.90 |
| augmix vs A4 | photo_strong | -1.72 | 43/129 | 3.5e-11 |

Reading: AugMix's effect survives distillation at the same size as in the no-KD B4
comparison (+2.2 combo, +1.9 photo, clean neutral), so it is an augmentation effect,
not a KD interaction. At 2/3 of the schedule the augmix arm is already level with
their 600-epoch A4 on the composite proxy, still ~1 point behind on clean.
Provisional: single seed, 200/300 epochs, two interruptions.
