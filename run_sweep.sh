#!/bin/bash
# runs the 5 sweep students side by side on one GPU; rerun the same script to resume
cd ~/project1/code
source ~/venvs/dl-hw2/bin/activate
mkdir -p logs
for a in r3_80_2-3-3 r3_96_2-2-2 r3_64_3-5-3; do
  nohup python train.py --arch $a --epochs 300 --workers 2 \
    --autoaug trivial --cutout 16 --mixup-alpha 0.2 --cutmix-alpha 1.0 --mix-off-epochs 15 \
    --sam --ema-decay 0.999 --save-every 5 \
    --teachers runs/teacher_wrn_s0/last.pt runs/teacher_pyr_s2/last.pt runs/teacher_wrn_s1/last.pt \
    --out runs/A_sweep_kd3t_300_$a >> logs/A_sw_$a-gpu.out 2>&1 &
done
