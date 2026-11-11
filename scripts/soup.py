"""Uniform weight soup (Wortsman et al. 2022): average the EMA weights of checkpoints that
share an architecture and come from the same training trajectory (e.g. F2s1 and a
finish started from F2s1), then recompute BatchNorm stats. Result: ONE model, same size.

  python scripts/soup.py --out runs/soup_F2s1/soup.pt --val-size 0 KEEP/F2s1.pt runs/F60_asam_F2s1/last.pt
  python scripts/soup.py --out runs/soup_kd2t/soup.pt --val-size 5000 ckpts/kd2t.pt runs/C60_asam_kd2t/last.pt

BN: reset, one pass (cumulative average) over the training split (--val-size 5000 keeps the
val holdout unseen; 0 = all 50k for full-data models) with crop+flip+TrivialAugment+Cutout16,
the training augmentation. Inputs are only read; --out must be new.
"""
import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import load_train, train_val_split, build_train_tf, CIFARDataset
from models import build_model, load_model, count_params


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpts", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--val-size", type=int, required=True)
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    assert not os.path.exists(a.out), f"{a.out} exists"
    assert "KEEP" not in a.out

    models, cks = zip(*[load_model(c, use_ema=not a.no_ema) for c in a.ckpts])
    cfg = cks[0].get("config")
    for c in cks[1:]:
        assert c.get("config") == cfg, "all checkpoints must have the same architecture"
    for c, ck in zip(a.ckpts, cks):
        print(f"  {c}: epoch {ck.get('epoch')}, val_size {ck.get('args', {}).get('val_size')}")
    soup = build_model(cfg)
    sds = [m.state_dict() for m in models]
    avg = {}
    for k in sds[0]:
        if sds[0][k].is_floating_point():
            avg[k] = sum(sd[k].double() for sd in sds) / len(sds)
            avg[k] = avg[k].float()
        else:
            avg[k] = sds[0][k].clone()
    soup.load_state_dict(avg)

    dev = torch.device("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available() else "cpu")
    soup.to(dev)
    bns = [m for m in soup.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
    soup.eval()
    for m in bns:
        m.train()
    x, y = load_train()
    if a.val_size > 0:
        x, y, _, _ = train_val_split(x, y, a.val_size)
    loader = DataLoader(CIFARDataset(x, y, build_train_tf("trivial", 16)), batch_size=a.batch_size,
                        shuffle=True, num_workers=a.workers)
    print(f"BN pass on {len(x)} train images, device {dev}")
    with torch.no_grad():
        for xb, _ in loader:
            soup(xb.to(dev))
    for m in bns:
        m.momentum = 0.1
    n = count_params(soup)
    assert n <= 5_000_000
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save({"arch": cks[0]["arch"], "config": cfg, "args": dict(cks[0].get("args", {}), val_size=a.val_size),
                "model": {k: v.cpu() for k, v in soup.state_dict().items()}, "ema": None,
                "epoch": cks[0].get("epoch"), "soup_of": list(a.ckpts)}, a.out)
    print(f"wrote {a.out}: {n:,} params, soup of {len(a.ckpts)}")


if __name__ == "__main__":
    main()
