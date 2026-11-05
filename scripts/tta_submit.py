"""5-crop (2px plus-shaped shifts) x hflip TTA submission from a checkpoint. Inference only.
The TTA code is in tta.py (shared with scripts/tta_eval.py). This is the procedure of the
0.911 F2s1 submission.
Usage: python scripts/tta_submit.py ckpts/F2s1.pt submission_F2s1_tta5.csv [--policy 5crop]"""
import argparse, os, sys
import pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import load_model, count_params
from data import load_test, DATA_DIR
from tta import view_probs, combine, POLICIES

p = argparse.ArgumentParser()
p.add_argument("ckpt"); p.add_argument("out")
p.add_argument("--policy", default="5crop", choices=list(POLICIES))
a = p.parse_args()
assert "KEEP" not in a.out, "never write into KEEP/"
dev = torch.device("mps" if torch.backends.mps.is_available() else
                   "cuda" if torch.cuda.is_available() else "cpu")
m, ck = load_model(a.ckpt); m = m.to(dev).eval()
n = count_params(m); assert n <= 5_000_000, n
print(f"{a.ckpt}: {n:,} params, epoch {ck['epoch']}, EMA={ck.get('ema') is not None}, device {dev}")
xt, ids = load_test()
probs = view_probs(m, xt, dev)
pred, flip_only = combine(probs, a.policy).argmax(1), combine(probs, "flip").argmax(1)
sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
assert set(sub["id"]) == set(ids)
sub["label"] = sub["id"].map(dict(zip(ids, pred.tolist())))
sub.to_csv(a.out, index=False)
print(f"wrote {a.out}: {len(sub)} rows, policy {a.policy}, changed vs flip-only TTA: {(pred != flip_only).sum()}")
