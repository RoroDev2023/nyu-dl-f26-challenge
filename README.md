# nyu-dl-f26-challenge

## 1. Overview

NYU Tandon Deep Learning (Fall 2026) Kaggle competition: train a ResNet **from scratch** on
CIFAR-10 with **<= 5M trainable parameters**, scored on a hidden 2000-image test set.

Our approach:
- Strong augmentation (TrivialAugment, Cutout, mixup/CutMix) + SAM + EMA of the weights.
- Knowledge distillation from big teachers we train ourselves (WRN-28-10, PyramidNet+ShakeDrop)...
- ...into a ~4.9M-parameter ResNet student (`r3_64_3-5-3`).

## 2. Competition rules we must follow

- The final submission is **one** network with <= 5M trainable params. No ensemble as the final model.
- No pretrained weights, no external data.
- Teachers (any size) must be trained from scratch by us, on the same train data.
- Hidden test images are for inference only: no pseudo-labels, no test-time BN adaptation.
- Validate only on our own split of the train data, never on the official CIFAR-10 test set.
- 5 Kaggle submissions per day for the **whole team**.
- The repo stays **private** until the competition ends.

## 3. Repo layout

| File | What it is |
|---|---|
| `data.py` | Loads train/test images, fixed stratified 5k val split, CPU augmentation, `GPUTrainLoader` (`--gpu-aug`) |
| `models.py` | Student ResNet (+SE, drop path, dropout), WideResNet, PyramidNet+ShakeDrop, `ARCHS` / `TEACHER_ARCHS` presets, `build_model`, `load_model` |
| `tricks.py` | EMA, mixup/CutMix, KD loss + teacher ensemble, SAM/ASAM perturbation |
| `train.py` | Training script: all tricks are CLI flags, auto-resume from `last.pt`, mixed precision, logging |
| `predict.py` | Checkpoint -> `submission.csv` (EMA + hflip TTA, checks the 5M budget) |
| `scripts/summarize.py` | Collects every `runs/*/results.json` into one sorted table |
| `run.sbatch` | Slurm job script: 1 A100, runs `python train.py <your flags>` |
| `submit.sh` | Wrapper around `sbatch` for the HPC Jupyter session (see section 8) |
| `check_data.py` | Sanity check: loads train/test data and saves `check_data.png` with sample images |
| `check_loader.py` | Sanity check: shows an augmented training batch (`check_loader.png`) |
| `check_model.py` | Prints the param count of every student and teacher preset |

Not in git (see `.gitignore`): `data/`, `runs/`, `logs/`, checkpoints (`*.pt`), zips,
images, `submission*.csv`.

## 4. Data setup

The data is **not** in the repo.

1. Get `hpc_data.zip` from the team Google Drive link (ask Rahim).
2. Put it in the repo root and unzip:

```bash
unzip -q hpc_data.zip
```

Expected layout:

```
data/train_cache.npz        # all 50k train images, decoded once
data/train_labels.csv
data/test.csv
data/sample_submission.csv
data/test_images/           # the 2000 hidden test PNGs
```

`train_cache.npz` holds the 50k training PNGs decoded into one `(50000, 32, 32, 3)` uint8
array plus labels. Opening 50k PNGs takes about a minute, so `load_train()` builds this file
once and reuses it. The zip has no `data/train_images/`, so the cache can't be rebuilt from
the zip alone: keep the file.

**Downloading from Kaggle directly:** the "Download All" button and the bulk CLI download are
broken for this competition. Fallback: in a Kaggle notebook attached to the competition,
zip `/kaggle/input/...` into `/kaggle/working/data.zip`, then download that file from the
notebook's output.

## 5. Local setup (laptop / own GPU)

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision numpy pandas pillow pyyaml matplotlib
```

Quick checks:

```bash
python check_data.py      # loads the data, saves check_data.png
python check_model.py     # param counts of every preset, all students must say OK
```

Smoke test (a few batches, 1 epoch, a throwaway output folder):

```bash
python train.py --epochs 1 --warmup 1 --limit-batches 5 --workers 0 --out runs/smoke
```

`train.py` picks CUDA, then Apple MPS, then CPU. Force one with `--device cpu`.
On a Mac use `--workers 0`: data workers restart every epoch (needed for exact resume),
which is slow on macOS.

## 6. NYU HPC Cloud Bursting setup

1. **Open OnDemand** (NYU VPN if off campus) -> Interactive Apps -> **Jupyter Notebook**,
   account `cs_gy_6923-2026fa`.
   - For setup and monitoring use a **CPU session**: GPUs = 0, partition `n2c48m24`.
   - GPU sessions burn the team's 300 GPU-hour budget. They are also auto-killed after
     ~20 min of low GPU utilization. Run training as Slurm jobs instead (section 8).
2. **Terminal:** in Jupyter, File -> New -> Terminal. The terminal on the Files page doesn't work.
3. **Python env:**

   ```bash
   python -m venv ~/venvs/<name>
   source ~/venvs/<name>/bin/activate
   pip install torch torchvision numpy pandas pillow pyyaml matplotlib
   ```

4. **Clone the private repo** with a GitHub personal access token (classic, `repo` scope).
   Use it as the password when git asks. `credential.helper store` saves it, so it's asked once:

   ```bash
   git config --global credential.helper store
   git clone https://github.com/RoroDev2023/nyu-dl-f26-challenge.git
   cd nyu-dl-f26-challenge
   mkdir -p logs      # Slurm writes job output here and won't create the folder itself
   ```

5. **Data:** upload `hpc_data.zip` with the OnDemand file browser, then `unzip -q hpc_data.zip`.
   If the data lives outside the repo, link it (`data` is git-ignored):

   ```bash
   ln -s <path>/data data
   ```

6. **Edit `run.sbatch`:** two lines are Rahim's paths. Change them for your setup:

   ```bash
   source ~/venvs/dl-hw2/bin/activate   # -> your venv
   cd ~/project1/code                   # -> your clone of this repo
   ```

## 7. Running training

Every trick is off by default. Plain `python train.py` is the baseline: ResNet `r3_64_3-5-3`,
crop + flip, label smoothing 0.1, SGD + Nesterov, warmup + cosine LR, 200 epochs.

### Flags (defaults in brackets)

**Architecture**
- `--arch` [`r3_64_3-5-3`]: one of these presets.
  - Students: `r3_64_3-5-3`, `r3_80_2-3-3`, `r3_96_2-2-2`, `r4_64_3-3-2-1`, `r3_40_8-8-8`.
  - Teachers: `wrn_28_10`, `pyramid_110_270`.
  - Tiny teachers for tests: `wrn_16_1`, `pyramid_20_24`.
- `--no-budget`: skip the 5M assert. Teachers only.
- `--se`, `--drop-path RATE` [0], `--dropout P` [0]: squeeze-excitation, stochastic depth
  (linear over blocks), dropout before fc. ResNet students only.

**Knowledge distillation**
- `--teachers A.pt B.pt ...` [none]: teacher checkpoints. They see the exact same augmented +
  mixed batch as the student, and their softened predictions are averaged. Label smoothing is off when KD is on.
- `--kd-alpha` [0.9]: weight of the KD loss. `1 - alpha` goes to cross entropy on the hard labels.
- `--kd-temp` [4]: temperature.

**Augmentation**
- `--autoaug` [`none`]: one of `none`, `trivial`, `randaug`, `autoaug_cifar`, `augmix`.
  `augmix` is torchvision AugMix, severity 3, mixture width 3, **one view, no JSD loss**
  (B3 winner: shift proxy +3.1pp, clean unchanged; see STATUS.md).
- `--cutout SIZE` [0 = off]: one black square per image.
- `--geo-aug`: mild affine (translate/scale/rotate). B3: −0.5 clean for +0.6 shifted,
  not adopted by default.
- `--extra-aug`: strong affine + color jitter. B3: −1.0 clean, no shifted gain —
  experimental, do not use for real runs.
- `--gpu-aug`: augment whole batches on the GPU instead of in CPU workers.
- `--aug-chunk` [16]: with `--gpu-aug`, how many images share one autoaug draw.

**Mixing**
- `--mixup-alpha` [0], `--cutmix-alpha` [0]: 0 = off. If both are on, each batch picks one 50/50.
- `--mix-prob` [1.0]: chance to mix each batch.
- `--mix-off-epochs N` [0]: no mixing for the last N epochs.

**SAM / EMA**
- `--sam`, `--sam-rho` [0.05]: sharpness-aware minimization. Each step costs 2 forward/backward passes.
- `--asam`: adaptive SAM (implies `--sam`). Use `--sam-rho 0.5` with it.
- `--ema-decay` [0 = off]: e.g. `0.999`. Both raw and EMA accuracy are logged each epoch.

**Training and infrastructure**
- `--epochs` [200], `--batch-size` [128], `--lr` [0.1], `--wd` [5e-4], `--warmup` [5 epochs],
  `--label-smoothing` [0.1].
- `--seed` [0]. `--val-size` [5000]: `0` trains on all 50k, for final runs.
- `--out` [`runs/baseline`]: run folder. `--workers` [4]: CPU data workers.
- `--save-every N` [1]: checkpoint every N epochs.
- `--compile`: `torch.compile`.
- `--limit-batches N` [0 = off]: debug, stop each epoch after N batches.
- `--device` [auto].

Each run folder gets these files:
- `args.json`
- `log.csv`: per epoch, lr, train loss/acc, val acc raw + EMA, `seconds`, `data_seconds`.
- `last.pt`: model, EMA, optimizer, RNG state.
- `results.json`: final val acc, raw and EMA, with and without TTA.

### Commands that produced our results

```bash
# baseline student
python train.py --epochs 200 --workers 10 --save-every 5 --out runs/student_baseline

# all-tricks student, no KD
python train.py --arch r3_64_3-5-3 --epochs 300 --workers 10 --autoaug trivial --cutout 16 \
  --mixup-alpha 0.2 --cutmix-alpha 1.0 --mix-off-epochs 15 --sam --ema-decay 0.999 \
  --save-every 5 --out runs/student_tricks_300

# WRN-28-10 teacher
python train.py --arch wrn_28_10 --no-budget --epochs 400 --workers 10 --autoaug trivial --cutout 16 \
  --mixup-alpha 0.2 --cutmix-alpha 1.0 --mix-off-epochs 20 --sam --ema-decay 0.999 --seed 0 \
  --save-every 5 --out runs/teacher_wrn_s0

# PyramidNet teacher (no SAM)
python train.py --arch pyramid_110_270 --no-budget --epochs 400 --workers 10 --autoaug trivial --cutout 16 \
  --mixup-alpha 0.2 --cutmix-alpha 1.0 --mix-off-epochs 20 --ema-decay 0.999 --seed 2 \
  --save-every 5 --out runs/teacher_pyr_s2

# KD student from 2 teachers
python train.py --arch r3_64_3-5-3 --epochs 600 --workers 10 --autoaug trivial --cutout 16 \
  --mixup-alpha 0.2 --cutmix-alpha 1.0 --mix-off-epochs 30 --sam --ema-decay 0.999 --save-every 5 \
  --teachers runs/teacher_wrn_s0/last.pt runs/teacher_pyr_s2/last.pt --out runs/kd_r3_64_2t_600
```

Notes:
- **Teacher seeds:** every teacher needs a different `--seed`, or two teachers of the same arch end up near copies.
- **New `--out` per experiment:** if `<out>/last.pt` exists, `train.py` resumes it. It refuses
  to continue if flags changed (except `--workers`, `--save-every`, `--compile`, `--device`).
- **ASAM:** with `--asam`, use `--sam-rho 0.5`. The default 0.05 is for plain SAM.
- **Data speed:** if `data_seconds` in `log.csv` is a big part of `seconds`, the CPU data
  pipeline is the bottleneck. Add `--gpu-aug`.
- **Val split:** it's the same for every run, whatever `--seed`. Teachers and students must
  hold out the same images, otherwise the student's val accuracy is inflated. `train.py`
  warns if a teacher used a different `--val-size`.

## 8. Running on the HPC with Slurm

### `run.sbatch`, line by line

```bash
#SBATCH --account=cs_gy_6923-2026fa     # course account (GPU hours are charged here)
#SBATCH --partition=c12m85-a100-1       # cloud node with 1x A100 40GB, 12 CPUs
#SBATCH --gres=gpu:1                    # one GPU
#SBATCH --cpus-per-task=12              # CPU cores for the data workers (--workers 10)
#SBATCH --mem=64G                       # RAM
#SBATCH --time=24:00:00                 # wall-clock limit
#SBATCH --requeue                       # if the node is preempted, Slurm requeues the job
#SBATCH --output=logs/%x-%j.out         # logs/<job-name>-<job-id>.out (logs/ must exist)

source ~/venvs/dl-hw2/bin/activate      # your venv (edit)
cd ~/project1/code                      # your repo clone (edit)
nvidia-smi                              # records which GPU we got in the log
python train.py "$@"                    # everything after run.sbatch is passed to train.py
```

A requeued job restarts the same command with the same `--out`, so it **auto-resumes from
`last.pt`**. It loses at most `--save-every` epochs.

### Always submit with `./submit.sh`, not `sbatch`

The Jupyter session sets `SLURM_EXPORT_ENV=NONE` and `SLURM_GET_USER_ENV=1`. Jobs submitted
from it inherit them, and they make cloud nodes time out at boot: the job gets cancelled in
`CF` state, or ends up "user env retrieval failed requeued held". `submit.sh` unsets both and
calls `sbatch --export=ALL`.

```bash
# submit: sbatch options first, then run.sbatch, then train.py flags
./submit.sh --job-name=A_kd2t run.sbatch --arch r3_64_3-5-3 --epochs 600 --workers 10 ... --out runs/A_kd2t

# chain: start the student only after the teacher job (id 12345) finishes OK
./submit.sh --job-name=A_student --dependency=afterok:12345 run.sbatch ... --teachers runs/teacher_wrn_s0/last.pt ...

# override partition / memory from the command line (wins over the #SBATCH lines)
./submit.sh --job-name=B_test --partition=g2-standard-12 --mem=40G run.sbatch ...
```

### Monitoring

```bash
squeue -u $USER                                   # my jobs and their state
squeue -u $USER --start                           # expected start times (UTC)
sacct -u $USER --starttime today                  # finished / failed / cancelled jobs
tail -n 30 logs/<job-name>-<job-id>.out           # job output
for f in runs/*/log.csv; do echo "$f: $(tail -1 $f)"; done   # last epoch of every run
```

### Known behavior

- About 2 A100 jobs run at once per user. The rest wait in the queue.
- The L4 partition (`g2-standard-12`) has only 48GB RAM, so it needs `--mem=40G`. Its nodes
  often failed to boot.
- Jobs auto-resume from `last.pt` if requeued.

A100 timings per epoch:

| Run | s/epoch |
|---|---|
| Student, baseline | 11 |
| Student, all tricks + SAM | 19 |
| KD student, 2 teachers | ~30 |
| WRN-28-10 + SAM | 29 |
| PyramidNet-110-270 | 45 |

## 9. Predicting and submitting

```bash
python predict.py runs/kd_r3_64_2t_600/last.pt --out submission_kd2t.csv
```

By default `predict.py` does the following:
- Uses the EMA weights (`--no-ema` for raw) and hflip TTA (`--no-tta` to skip).
- Asserts <= 5M params.
- Checks the ids match `data/sample_submission.csv` and writes rows in its order.

Download the CSV, then submit with the Kaggle CLI. Its token goes in `~/.kaggle/access_token`
(never commit it):

```bash
kaggle competitions submit -c cifar-10-res-net-architecture-challenge-dl \
  -f submission_kd2t.csv -m "kd_r3_64_2t_600 EMA+TTA, val 97.76"
kaggle competitions submissions -c cifar-10-res-net-architecture-challenge-dl
```

Always put the run name and val accuracy in the submission message.

## 10. Results so far

| Run | Model | Params | Epochs | Tricks | Val % | Public LB |
|---|---|---|---|---|---|---|
| kd_r3_64_2t_600 | r3_64_3-5-3 | 4.92M | 600 | KD 2 teachers + all tricks | 97.76 | 0.904 |
| student_tricks_300 | r3_64_3-5-3 | 4.92M | 300 | all tricks, no KD | 97.36 (EMA+TTA) | - |
| student_baseline | r3_64_3-5-3 | 4.92M | 200 | crop + flip + label smoothing | 94.76 (95.42 TTA) | - |
| B_augmix_300_sam | r3_64_3-5-3 | 4.92M | 300 | all tricks + `--autoaug augmix` | 97.06 (97.04 EMA+TTA) | - |
| B_trivial_300_sam | r3_64_3-5-3 | 4.92M | 300 | all tricks (team recipe) | running | - |
| B_pilot_augmix | r3_64_3-5-3 | 4.92M | 150 | augmix, no SAM, no KD | 96.52 (TTA) | - |
| B_pilot_trivial | r3_64_3-5-3 | 4.92M | 150 | trivial, no SAM, no KD | 96.48 (TTA) | - |
| teacher_wrn_s1 | WRN-28-10 | 36.5M | 400 | teacher | 98.14 | - |
| teacher_wrn_s0 | WRN-28-10 | 36.5M | 400 | teacher | 98.08 | - |
| teacher_pyr_s2 | PyramidNet-110 | 28.5M | 400 | teacher | 97.66 | - |

Pending: `kd_r3_64_3t_600` (3 teachers) and the architecture sweep `runs/sweep_kd3t_300_*`.
Run `python scripts/summarize.py` for the latest.

**Key finding:** the hidden test set is distribution-shifted. Models score 7-11 points lower
on Kaggle than on our val split (our KD student: 97.76 val -> 0.904 public), and stronger
models hold up better on the shift. The public LB uses only 1000 images (about ±1 point of
noise), so never pick models by small public-LB differences.

## 11. Next steps: two tracks

### Track A (Rahim, HPC A100)

- [ ] **A1:** read the architecture sweep (`runs/sweep_kd3t_300_*`) and pick the student shape.
- [ ] **A2:** 2 vs 3 teachers.
- [ ] **A3:** KD settings at 300 epochs: `--kd-temp` 1 / 2 / 4, `--kd-alpha` 0.7 / 0.9 / 1.0,
      `--mix-prob` 0.5 / 1.0.
- [ ] **A4:** 600 vs 1000 epochs.
- [ ] **A5:** retrain 1-2 teachers with Track B's robust augmentation.
- [ ] **A6:** final runs with `--val-size 0`, 2 seeds, then predict + submit.

### Track B (partner, RTX 5000 + own HPC hours)

- [x] **B1:** characterize the test shift: per-channel mean/std, sharpness, noise. Look at
      the images, but don't label them. → new natural photos, not corrupted CIFAR;
      0/2000 exact matches. `runs/analysis/shift.md`.
- [x] **B2:** build a "shifted val" by corrupting our 5k val split (noise, blur, JPEG,
      contrast, color). Report clean + shifted val for every run.
      → `scripts/shift_analysis.py --checkpoints runs/B_.../last.pt`.
- [x] **B3:** robustness augmentation ablations at 150-200 epochs, no KD. Pick by shifted
      val, as long as clean val drops < 0.3.
      → **AugMix wins**: 87.42 vs 84.32 combo, clean 96.52 vs 96.48. RandAugment,
      AutoAugment-CIFAR, `--geo-aug`, `--extra-aug` all rejected.
- [x] **B4:** add the winner as a flag in the same style as the other tricks, on a branch.
      → `--autoaug augmix` on `b_shift`; 300 ep + SAM gives **97.06 val, 89.02 combo**.
- [ ] **B5:** test it on top of KD (3 teachers, 300 epochs).
      → **blocked**: needs `teacher_wrn_s0/s1` and `teacher_pyr_s2` checkpoints in this
      repo's format. Our old `../proj1` checkpoints do not load.

## 12. Team conventions

- Work on branches. Merge only tested code into `main`.
- Run names (`--out` / `--job-name`) start with `A_` or `B_`.
- Everything uses the default 5k val split (seed 0). Don't change `--val-size` except for final `--val-size 0` runs.
- Add new rows to the results table above.
- Kaggle submissions per day: Rahim 2, partner 2, 1 spare.
- Final pick: 2 single <= 5M models, chosen by clean + shifted val, **not** by public LB.
- Keep `runs/` and `logs/` on disk. Course staff may ask for checkpoints and logs.

## 13. References

- He et al. 2016, *Deep Residual Learning for Image Recognition* (ResNet)
- Zagoruyko & Komodakis 2016, *Wide Residual Networks*
- Han et al. 2017, *Deep Pyramidal Residual Networks* (PyramidNet)
- Yamada et al. 2019, *ShakeDrop Regularization for Deep Residual Learning*
- Foret et al. 2021, *Sharpness-Aware Minimization for Efficiently Improving Generalization* (SAM)
- Kwon et al. 2021, *ASAM: Adaptive Sharpness-Aware Minimization*
- Müller & Hutter 2021, *TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation*
- Cubuk et al. 2019, *AutoAugment: Learning Augmentation Strategies from Data*
- Cubuk et al. 2020, *RandAugment: Practical Automated Data Augmentation with a Reduced Search Space*
- DeVries & Taylor 2017, *Improved Regularization of Convolutional Neural Networks with Cutout*
- Zhang et al. 2018, *mixup: Beyond Empirical Risk Minimization*
- Yun et al. 2019, *CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features*
- Szegedy et al. 2016, *Rethinking the Inception Architecture for Computer Vision* (label smoothing)
- Hu et al. 2018, *Squeeze-and-Excitation Networks*
- Huang et al. 2016, *Deep Networks with Stochastic Depth*
- Hinton et al. 2015, *Distilling the Knowledge in a Neural Network* (KD)
- Beyer et al. 2022, *Knowledge Distillation: A Good Teacher is Patient and Consistent*
- Hendrycks et al. 2020, *AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty*
