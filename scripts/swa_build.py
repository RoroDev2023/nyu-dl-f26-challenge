"""Build one SWA model from a --lr-schedule cyclic run (Izmailov et al. 2018, timgaripov/swa).

  python scripts/swa_build.py runs/C_swa --start-cycle 2 --out runs/C_swa/swa.pt

- averages the RAW weights saved at the end of cycles start..last (swa_cycle<k>.pt), the
  lowest-LR point of each cycle. Cycle 1 is skipped by default (still close to A4).
- BatchNorm: running stats of the average are recomputed, not averaged: reset, then one
  pass (cumulative average, momentum=None) over the TRAINING split (val_size from the
  run's args, so the val holdout is never seen) with the run's own training augmentation
  (crop+flip, --autoaug, --cutout; no mixup/CutMix, those are batch-level). Dropout and
  DropPath are put in eval mode during the pass; only BN is in train mode.
- writes a NEW checkpoint (ema=None, so load_model uses these weights). The snapshot files
  and every other checkpoint are only read.
"""
import argparse
import glob
import os
import re
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import load_train, train_val_split, build_train_tf, CIFARDataset
from models import build_model, count_params


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--start-cycle", type=int, default=2)
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--also-avg-bn", action="store_true",
                   help="also save the average with averaged BN buffers (diagnostic only)")
    a = p.parse_args()
    assert not os.path.exists(a.out), f"{a.out} exists, pick a new --out"

    snaps = sorted(glob.glob(os.path.join(a.run, "swa_cycle*.pt")),
                   key=lambda s: int(re.findall(r"(\d+)\.pt$", s)[0]))
    snaps = [s for s in snaps if int(re.findall(r"(\d+)\.pt$", s)[0]) >= a.start_cycle]
    assert len(snaps) >= 2, f"need >= 2 snapshots from cycle {a.start_cycle}, found {snaps}"
    print("averaging:", *snaps, sep="\n  ")

    first = torch.load(snaps[0], map_location="cpu", weights_only=False)
    last_epoch = torch.load(snaps[-1], map_location="cpu", weights_only=False)["epoch"]
    config, run_args = first["config"], first["args"]
    model = build_model(config)
    param_names = {n for n, _ in model.named_parameters()}
    avg = {k: v.clone().double() if v.is_floating_point() else v.clone()
           for k, v in first["model"].items()}
    for s in snaps[1:]:
        sd = torch.load(s, map_location="cpu", weights_only=False)["model"]
        for k, v in sd.items():
            if v.is_floating_point():
                avg[k] += v.double()
    avg = {k: (v / len(snaps)).float() if v.is_floating_point() else v for k, v in avg.items()}
    model.load_state_dict(avg)

    meta = {"swa_snapshots": snaps, "swa_start_cycle": a.start_cycle}
    if a.also_avg_bn:
        out2 = a.out.replace(".pt", "_avgbn.pt")
        torch.save({"arch": first["arch"], "config": config, "args": run_args,
                    "model": model.state_dict(), "epoch": last_epoch, "ema": None,
                    **meta, "bn": "averaged buffers"}, out2)
        print("wrote", out2)

    dev = torch.device("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(dev)
    # reset BN stats, cumulative average over the pass
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
    model.eval()
    for m in bns:
        m.train()

    x, y = load_train()
    xtr, ytr, _, _ = train_val_split(x, y, run_args["val_size"])
    tf = build_train_tf(run_args["autoaug"], run_args["cutout"])
    loader = DataLoader(CIFARDataset(xtr, ytr, tf), batch_size=a.batch_size, shuffle=True,
                        num_workers=a.workers, drop_last=False)
    print(f"BN pass: {len(xtr)} train images, autoaug={run_args['autoaug']}, "
          f"cutout={run_args['cutout']}, device {dev}")
    with torch.no_grad():
        for xb, _ in loader:
            model(xb.to(dev))
    for m in bns:
        m.momentum = 0.1

    n = count_params(model)
    assert n <= 5_000_000
    torch.save({"arch": first["arch"], "config": config, "args": run_args,
                "model": {k: v.cpu() for k, v in model.state_dict().items()},
                "epoch": last_epoch, "ema": None, **meta,
                "bn": "recomputed on train split with training augmentation"}, a.out)
    print(f"wrote {a.out}: {n:,} params, {len(snaps)} snapshots")


if __name__ == "__main__":
    main()
