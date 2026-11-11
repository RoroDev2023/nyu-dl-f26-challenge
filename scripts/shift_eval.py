"""Frozen shifted-validation check (diagnostic proxy, NOT the hidden test).
Applies 5 fixed mild changes to the 5000 val images and scores each model with flip TTA:
  bright  +25/255 on all pixels        contrast  x1.2 around the image mean
  gamma   0.8                          blur      3x3 Gaussian, sigma 0.7
  shift   2 px down-right (reflect)
Paired comparison groups all conditions of one image together: per image, count conditions
classified correctly; sign test on images where model B gets more vs fewer than the reference.

  python scripts/shift_eval.py runs/A/last.pt runs/B/last.pt [...]   (first = reference)
"""
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import load_train, train_val_split, MEAN, STD
from models import load_model


def conditions(x):
    """x: uint8 (N, 32, 32, 3) -> dict name -> float (N, 32, 32, 3) in 0..1."""
    f = x.astype(np.float32) / 255.
    out = {"clean": f}
    out["bright"] = np.clip(f + 25 / 255., 0, 1)
    m = f.mean(axis=(1, 2, 3), keepdims=True)
    out["contrast"] = np.clip((f - m) * 1.2 + m, 0, 1)
    out["gamma"] = f ** 0.8
    t = torch.from_numpy(f).permute(0, 3, 1, 2)
    k = torch.tensor([math.exp(-(i * i) / (2 * 0.7 ** 2)) for i in (-1, 0, 1)])
    k = (k[:, None] * k[None, :]); k = k / k.sum()
    w = k[None, None].repeat(3, 1, 1, 1)
    out["blur"] = F.conv2d(F.pad(t, (1, 1, 1, 1), mode="reflect"), w, groups=3).permute(0, 2, 3, 1).numpy()
    p = np.pad(f, ((0, 0), (2, 0), (2, 0), (0, 0)), mode="reflect")
    out["shift"] = p[:, :32, :32]
    return out


@torch.no_grad()
def correct(model, imgs, y, dev):
    mean = torch.tensor(MEAN).view(1, 3, 1, 1); std = torch.tensor(STD).view(1, 3, 1, 1)
    res = []
    for i in range(0, len(imgs), 500):
        xb = ((torch.from_numpy(imgs[i:i + 500]).permute(0, 3, 1, 2) - mean) / std).to(dev)
        p = F.softmax(model(xb).float(), 1) + F.softmax(model(xb.flip(-1)).float(), 1)
        res.append(p.argmax(1).cpu())
    return (torch.cat(res).numpy() == y)


def sign_p(a, b):
    n = a + b
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(a, b) + 1)) / 2 ** n)


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    x, y = load_train()
    _, _, xv, yv = train_val_split(x, y, 5000)
    conds = conditions(xv)
    names = [c for c in conds if c != "clean"]
    per_model = []
    for path in sys.argv[1:]:
        m, _ = load_model(path)
        m = m.to(dev).eval()
        ok = {c: correct(m, conds[c], yv, dev) for c in conds}
        per_model.append((path, ok))
        accs = "  ".join(f"{c} {ok[c].mean()*100:.2f}" for c in conds)
        print(f"{path}\n   {accs}   shifted-mean {np.mean([ok[c].mean() for c in names])*100:.2f}")
    ref_path, ref = per_model[0]
    ref_cnt = sum(ref[c].astype(int) for c in names)
    for path, ok in per_model[1:]:
        cnt = sum(ok[c].astype(int) for c in names)
        better, worse = int((cnt > ref_cnt).sum()), int((cnt < ref_cnt).sum())
        d = np.mean([ok[c].mean() - ref[c].mean() for c in names]) * 100
        print(f"{os.path.basename(os.path.dirname(path))} vs {os.path.basename(os.path.dirname(ref_path))}: "
              f"shifted-mean {d:+.2f} pt, images better {better} / worse {worse}, sign p={sign_p(better, worse):.3f}")


if __name__ == "__main__":
    main()
