#!/bin/bash
# Run on the rented 4-GPU machine, from the unpacked bundle folder. One run per GPU, each
# resumable: rerunning this script resumes every run from its last.pt.
#   STAGE=1 bash scripts/vast_run.sh     # 4 matched 300-ep val runs on GPUs 0-3
#   STAGE=1 RUNS=dp1 bash scripts/vast_run.sh   # one run on a 1-GPU machine
#   STAGE=2 RUNS="dp1 t2" bash scripts/vast_run.sh   # full-data 600-ep versions of winners
set -u
mkdir -p logs runs
T3="teachers/teacher_wrn_s0.pt teachers/teacher_pyr_s2.pt teachers/teacher_wrn_s1.pt"
T2="teachers/teacher_wrn_s0.pt teachers/teacher_wrn_s1.pt"
BASE="--arch r3_80_2-3-3 --autoaug trivial --cutout 16 --mixup-alpha 0.2 --cutmix-alpha 1.0 --sam --ema-decay 0.999 --save-every 5 --gpu-aug"

declare -A EXTRA=( [ctrl]="--teachers $T3" [dp1]="--drop-path 0.1 --teachers $T3"
                   [t2]="--teachers $T2" [a045]="--kd-alpha 0.45 --teachers $T3" )

if [ "${STAGE:-1}" = "1" ]; then
  RUNS_LIST="${RUNS:-ctrl dp1 t2 a045}"; COMMON="$BASE --epochs 300 --mix-off-epochs 15 --seed 0"; PFX=V300
else
  RUNS_LIST="$RUNS"; COMMON="$BASE --epochs 600 --mix-off-epochs 30 --seed 1 --val-size 0"; PFX=F600
fi

gpu=0
for r in $RUNS_LIST; do
  name=${PFX}_$r
  echo "GPU $gpu: $name"
  CUDA_VISIBLE_DEVICES=$gpu nohup python -u train.py $COMMON ${EXTRA[$r]} --out runs/$name \
      > logs/$name.log 2>&1 &
  gpu=$((gpu + 1))
done
sleep 5; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
