#!/bin/bash
# One-time upload of the data and teachers to the Modal volume "resnet5m-data".
# Run from anywhere, with the Modal venv active (source .venv-1/bin/activate).
# data/train_images is not uploaded: train.py only needs data/train_cache.npz.
set -euo pipefail
cd "$(dirname "$0")/.."

VOL=resnet5m-data
modal volume list --json | grep -q "\"$VOL\"" || modal volume create $VOL
modal volume list --json | grep -q "\"resnet5m-runs\"" || modal volume create resnet5m-runs

for f in train_cache.npz train_labels.csv test.csv sample_submission.csv; do
    modal volume put -f $VOL "data/$f" "/data/$f"
done
modal volume put -f $VOL data/test_images /data/test_images
for f in teachers/*.pt; do
    modal volume put -f $VOL "$f" "/teachers/$(basename "$f")"
done
modal volume ls $VOL /data
modal volume ls $VOL /teachers

# download a finished run (or one in progress) to ./runs/<run-name>:
#   modal volume get resnet5m-runs <run-name> ./runs/
