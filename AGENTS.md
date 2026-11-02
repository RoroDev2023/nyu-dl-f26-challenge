# AGENTS.md

Shared team repo for the CIFAR-10 **≤5M parameter** challenge. The hidden eval is
**distribution-shifted** (~2,000 new natural photographs, CIFAR-10.1-like), so the
clean 5k holdout is only a proxy. Team plan and trajectory: `PLAN.md`; current
numbers: `STATUS.md`.

## Current trajectory (2026-11-02)

- **Track A** (teammate, HPC A100) owns KD/architecture and leads at 97.76 clean
  val / **0.904 public** with a 2-teacher KD student. **Track B** (us) owns shift
  robustness; this repo is now the shared codebase.
- The eval is **new natural photos**, not corrupted CIFAR-10. See
  `runs/analysis/shift.md`. Prioritize natural-photo robustness (AugMix/RandAugment,
  CutMix/MixUp, scale/translation/rotation, color/lighting) over noise/blur/JPEG.
- Team conventions: seed-0 5k split, run names `A_...`/`B_...`, 5 submissions/day
  (2 ours, 2 theirs, 1 spare), final pick = 2 single ≤5M models by val + shifted
  val, not public LB.
- The 5M rule for **teachers** is unresolved; `--no-budget` exists for them.

## Environment

- **HPC**: `./submit.sh run.sbatch ...` (Slurm A100; `run.sbatch` activates
  `~/venvs/dl-hw2` and runs `train.py "$@"`). Create `logs/` once.
- **Local**: use `~/miniforge3/envs/ml/bin/python` (torch 2.10) or
  `~/venvs/dl-hw2` if present. Run from the repo root; imports are flat.
- **Data**: `data/train_cache.npz` (50k `x`/`y`), `data/test.csv`,
  `data/test_images/`, `data/sample_submission.csv`; `data/` is gitignored. If the
  cache is missing, `data.load_train()` builds it from `train_labels.csv` +
  `train_images/` (~1 min). Never train or select on the eval images.
- `opencode.json` holds the local Kaggle MCP token and is excluded via
  `.git/info/exclude`; never commit or print it.

## Commands

- Smoke: `python train.py --epochs 1 --limit-batches 2 --out runs/B_smoke --ema-decay 0.999`
- Baseline: `python train.py --arch r3_64_3-5-3 --epochs 200 --out runs/A_baseline --ema-decay 0.999`
- All tricks: add `--autoaug randaug --cutmix-alpha 1.0 --mixup-alpha 0.4 --mix-prob 1.0 --epochs 300`
- KD: add `--teachers t1.pt t2.pt --kd-alpha 0.9 --kd-temp 4`. Teachers must use the
  same `--val-size`; label smoothing is auto-disabled when teachers are present.
- Teacher: `--arch pyramid_110_270 --no-budget` (28.5M; **not submittable**).
- Full data: `--val-size 0` (no validation; final runs).
- Predict: `python predict.py runs/<run>/last.pt --out submission.csv` (EMA by
  default, horizontal-flip TTA, asserts ≤5M).
- Summarize: `python scripts/summarize.py --runs runs`
- HPC submit: `./submit.sh run.sbatch --arch ... --out runs/A_...`

## Conventions

- The val split is fixed at **seed 0** in `data.py`; never change it. Teachers and
  students must hold out the same images.
- `last.pt` is the resumable checkpoint (model/EMA/optimizer/scheduler/RNG);
  `results.json` holds final val/EMA/TTA; `log.csv` holds per-epoch rows.
- Checkpoint format is `config` + `model`/`ema`; **our old `../proj1` checkpoints are
  incompatible** with `models.load_model`.
- Name runs `A_...` or `B_...`; `scripts/summarize.py` sorts them apart.
- The 5M assert is in `train.py`/`predict.py`; `--no-budget` is only for teachers.
- Our archived code and older analysis live in `../proj1/our_code_backup/`; the
  shift report is copied to `runs/analysis/`.

## Submission

- `export KAGGLE_API_TOKEN=$(tr -d '[:space:]' < ~/.kaggle/kgat.txt)`
- `kaggle competitions submit -c cifar-10-res-net-architecture-challenge-dl -f submission.csv -m "run val=..."`
- 5/day for the team; public score is 1k images (~±1.1pp). Final picks by val +
  shifted val, not public LB.
