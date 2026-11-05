#!/bin/bash
# A3: 3 KD/arch variants of r3_80_2-3-3 side by side; rerun to resume
cd ~/project1/code
source ~/venvs/dl-hw2/bin/activate
mkdir -p logs
COMMON="--arch r3_80_2-3-3 --epochs 300 --workers 3 --autoaug trivial --cutout 16 --mixup-alpha 0.2 --cutmix-alpha 1.0 --mix-off-epochs 15 --sam --ema-decay 0.999 --save-every 5 --teachers runs/teacher_wrn_s0/last.pt runs/teacher_pyr_s2/last.pt runs/teacher_wrn_s1/last.pt"
nohup python train.py $COMMON --kd-temp 2 --out runs/A3_r3_80_T2 >> logs/A3_T2.out 2>&1 &
nohup python train.py $COMMON --kd-temp 1 --kd-alpha 1.0 --out runs/A3_r3_80_T1_a1 >> logs/A3_T1_a1.out 2>&1 &
nohup python train.py $COMMON --se --out runs/A3_r3_80_se >> logs/A3_se.out 2>&1 &
