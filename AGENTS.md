# AGENTS.md

Shared team repo for the CIFAR-10 **≤5M parameter** challenge. The hidden eval is
**distribution-shifted** (~2,000 new natural photographs, CIFAR-10.1-like), so the
clean 5k holdout is only a proxy. Team plan and trajectory: `PLAN.md`; current
numbers: `STATUS.md`.

## Current trajectory (2026-11-03)

- **Track A** (teammate, HPC A100) owns KD/architecture and leads at 97.76 clean
  val / **0.904 public** with a 2-teacher KD student. **Track B** (us) owns shift
  robustness; this repo is now the shared codebase.
- The eval is **new natural photos**, not corrupted CIFAR-10. See
  `runs/analysis/shift.md`. Prioritize natural-photo robustness (AugMix/RandAugment,
  CutMix/MixUp, scale/translation/rotation, color/lighting) over noise/blur/JPEG.
- **Our best single ≤5M model: `B_augmix_300_sam`** — 97.06 val (97.04 EMA+TTA),
  shifted proxy 89.02 vs 84.32 for the control, 300 ep + SAM + `--autoaug augmix`.
  AugMix was the B3 sweep winner; `--autoaug trivial` control is `B_trivial_300_sam`.
- **B5 (KD on top of AugMix) is downstream of A5, which is downstream of B4**:
  A5 (robust teachers) has not started and cannot until B4 is signed off, and B5
  needs A5's teachers. Old `../proj1` checkpoints are incompatible, and training a
  36.5M teacher on Modal would duplicate A5 — so B5 is a Nov 4+ item.
- Team conventions: seed-0 5k split, `A_...`/`B_...` run names, 5 submissions/day
  (2 ours, 2 theirs, 1 spare). The final submission must be **one ≤5M network — no
  ensembles**; keep 2 candidate single models chosen by val + shifted val, not public LB.
- Teachers may be any size (trained from scratch on the same train data); use
  `--no-budget`. Only the submitted student must be ≤5M.

## Environment

- **HPC**: Open OnDemand Jupyter (use a CPU session for setup), edit the two paths
  in `run.sbatch`, create `logs/`, and submit with
  `./submit.sh --job-name=A_... run.sbatch ...` (not raw `sbatch`; it unsets the
  Jupyter Slurm env vars). ~2 A100 jobs/user; 300 GPU-hours for the team.
- **Local**: use `~/miniforge3/envs/ml/bin/python` (torch 2.10) or
  `~/venvs/dl-hw2` if present. Run from the repo root; imports are flat.
- **Data**: `hpc_data.zip` from the team Drive (ask Rahim) unzips to `data/`;
  `data/train_cache.npz` holds the decoded 50k train set, and the zip has no
  `train_images/`, so keep the cache. Expected: `train_cache.npz`, `test.csv`,
  `test_images/`, `sample_submission.csv`. `data/` is gitignored. Never train or
  select on the eval images.
- `opencode.json` holds the local Kaggle MCP token and is excluded via
  `.git/info/exclude`; never commit or print it.

## Commands

- Smoke: `python train.py --epochs 1 --warmup 1 --limit-batches 5 --workers 0 --out runs/B_smoke`
- Baseline: `python train.py --epochs 200 --workers 10 --save-every 5 --out runs/A_baseline`
- All tricks (team recipe): add `--autoaug trivial --cutout 16 --mixup-alpha 0.2 --cutmix-alpha 1.0 --mix-off-epochs 15 --sam --ema-decay 0.999 --epochs 300`
- AugMix (B3 winner): `--autoaug augmix` (one view; no JSD). In the 300-ep + SAM recipe it gives 97.06 val and 89.02 combo vs 84.32 for `trivial` `B_pilot` control. Prefer this over `trivial` for robustness experiments.
- `--extra-aug` is experimental (affine + color); the B3 pilot rejected it (−1.0 clean val, no shifted gain). Do not merge it to main.
- `--geo-aug` is mild affine only; B3 showed −0.5 clean for +0.6 shifted; not adopted.
- KD: add `--teachers t1.pt t2.pt --kd-alpha 0.9 --kd-temp 4 --epochs 600`; teachers must use the same `--val-size`. Label smoothing is auto-disabled when teachers are present.
- Teacher: `--arch wrn_28_10 --no-budget --epochs 400 ... --seed 0` (or `pyramid_110_270`, no SAM). Use a different seed per teacher.
- Teacher architecture preset: `pyramid_110_270` is 28.5M; `wrn_28_10` is 36.5M. Both need `--no-budget`.
- Full data: `--val-size 0` (no validation; final runs).
- Predict: `python predict.py runs/<run>/last.pt --out submission.csv` (EMA by default, hflip TTA, asserts ≤5M). One model only; no ensembles.
- Summarize: `python scripts/summarize.py --runs runs`
- Shift eval (B1/B2): `python scripts/shift_analysis.py --checkpoints runs/B_augmix_300_sam/last.pt --out-dir runs/analysis/B4`
  → clean + 9 corruptions on the seed-0 5k split, EMA weights, **no TTA** (`--tta`
  is parsed but unused, and it rewrites `B_shift.md`/`.json`/montage in `--out-dir`,
  so give each comparison its own directory).
- Modal (when local GPU time is not enough): `modal deploy modal_shared.py`, then
  `modal.Function.from_name("cifar-shared", "run_task").spawn("train", [...])`.
  Volumes `proj1-data`→`/data`, `proj1-runs`→`/runs`; A10G, ~19 s/epoch for a 5M
  student at 300 ep. Never `modal run` for long jobs (killing the CLI cancels them).
- If `log.csv` shows `data_seconds` dominates, add `--gpu-aug` and adjust `--workers`.
- HPC submit: `./submit.sh --job-name=A_... run.sbatch --arch ... --out runs/A_...`

## Conventions

- The val split is fixed at **seed 0** in `data.py`; never change it. Teachers and
  students must hold out the same images.
- `last.pt` is the resumable checkpoint (model/EMA/optimizer/scheduler/RNG);
  `results.json` holds final val/EMA/TTA; `log.csv` holds per-epoch rows.
- Checkpoint format is `config` + `model`/`ema`; **our old `../proj1` checkpoints are
  incompatible** with `models.load_model`.
- Name runs `A_...` or `B_...`; `scripts/summarize.py` sorts them apart. Add new
  rows to the README results table.
- `runs/` (and `*.pt`, `*.png`, `*.csv`, `*.zip`, `*.npz`) are gitignored, so
  checkpoints and shift reports stay on disk only — copy every new number into
  `STATUS.md`/README, not just into `runs/analysis/`.
- The 5M assert is in `train.py`/`predict.py`; `--no-budget` is only for teachers.
- The final submission is **one ≤5M network; ensembles are not allowed as the final
  model**. Use `predict.py` with a single checkpoint.
- Our archived code and older analysis live in `../proj1/our_code_backup/`; the
  shift report is copied to `runs/analysis/`.

## Submission

- `export KAGGLE_API_TOKEN=$(tr -d '[:space:]' < ~/.kaggle/kgat.txt)`
- `kaggle competitions submit -c cifar-10-res-net-architecture-challenge-dl -f submission.csv -m "run val=..."`
- 5/day for the team (2 ours, 2 theirs, 1 spare). Public score is 1k images
  (~±1.1pp). Final picks are single models chosen by val + shifted val, not public.
