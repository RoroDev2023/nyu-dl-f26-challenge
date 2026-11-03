"""Run train.py and predict.py on Modal (modal.com) cloud GPUs. See "Running on Modal" in
README.md. train.py, data.py and models.py run unchanged: in the container data/, teachers/
and runs/ are links into two Volumes, so their usual relative paths just work.

    modal run --detach modal_train.py --gpu A100-40GB --args "--out runs/X ..."
    modal run modal_train.py::predict_entry --checkpoint runs/X/last.pt --out runs/X/submission.csv
"""
import os
import shlex
import subprocess
import sys
import threading
from pathlib import Path

import modal

APP_DIR = "/root/app"
HERE = Path(__file__).parent
GPU_COST = {"A100-40GB": 2.10, "H100": 3.95}    # rough $/hour
COMMIT_EVERY = 5 * 60                           # seconds

app = modal.App("resnet5m")
data_vol = modal.Volume.from_name("resnet5m-data", create_if_missing=True)   # data/, teachers/
runs_vol = modal.Volume.from_name("resnet5m-runs", create_if_missing=True)   # runs/
VOLUMES = {"/mnt/data": data_vol, "/mnt/runs": runs_vol}
LINKS = {"data": "/mnt/data/data", "teachers": "/mnt/data/teachers", "runs": "/mnt/runs"}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "torchvision", "numpy", "pandas", "pillow", "pyyaml", "matplotlib")
    .workdir(APP_DIR)
    .add_local_dir(HERE / "scripts", f"{APP_DIR}/scripts", ignore=["__pycache__"])
)
# code only: data/, runs/, teachers/, venvs, zips and pngs never go into the image
if modal.is_local():
    for path in sorted(HERE.glob("*.py")):
        image = image.add_local_file(path, f"{APP_DIR}/{path.name}")


def link_volumes():
    for name, target in LINKS.items():
        os.makedirs(target, exist_ok=True)
        if not os.path.lexists(name):
            os.symlink(target, name)


def commit_runs():
    try:
        runs_vol.commit()
    except Exception as e:      # a failed commit must not kill the training
        print("runs volume commit failed:", e, flush=True)


def commit_loop(stop):
    while not stop.wait(COMMIT_EVERY):
        commit_runs()


def out_dir(argv):
    """train.py's --out, which must be inside runs/ or the checkpoints are lost."""
    for i, a in enumerate(argv):
        if a == "--out" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--out="):
            return a.split("=", 1)[1]
    return "runs/baseline"


@app.function(image=image, gpu="A100-40GB", cpu=8, memory=32768, timeout=24 * 3600,
              volumes=VOLUMES)
def train(args: str):
    argv = shlex.split(args)
    if not out_dir(argv).startswith("runs/"):
        raise SystemExit("--out must be under runs/ (the runs volume)")
    link_volumes()
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    # periodic commits: the checkpoints survive if the job is killed, and rerunning
    # the same args resumes from runs/<out>/last.pt
    stop = threading.Event()
    threading.Thread(target=commit_loop, args=(stop,), daemon=True).start()
    try:
        cmd = [sys.executable, "-u", "train.py", *argv]
        print("$", shlex.join(cmd), flush=True)
        subprocess.run(cmd, check=True)
    finally:
        stop.set()
        commit_runs()


@app.function(image=image, cpu=4, memory=8192, timeout=30 * 60, volumes=VOLUMES)
def predict(checkpoint: str, out: str):
    if not out.startswith("runs/"):
        raise SystemExit("--out must be under runs/ (the runs volume)")
    link_volumes()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    subprocess.run([sys.executable, "-u", "predict.py", checkpoint, "--out", out], check=True)
    runs_vol.commit()


# `modal run modal_train.py::predict_entry --checkpoint ... --out ...` runs predict directly.
# Not a second local_entrypoint: then plain `modal run modal_train.py` would be ambiguous.
predict_entry = predict


@app.local_entrypoint()
def main(args: str, gpu: str = "A100-40GB"):
    rate = GPU_COST.get(gpu)
    cost = f"~${rate:.2f}/h, 24h timeout = ~${24 * rate:.0f} max" if rate else "cost unknown"
    print(f"GPU: {gpu} ({cost})")
    print(f"train.py {args}")
    train.with_options(gpu=gpu).remote(args)
