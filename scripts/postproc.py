"""Test-time BN adaptation + class-balanced (Sinkhorn) prediction. Allowed by the
professor (Slack), must be disclosed in the report."""
import copy, sys
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
sys.path.insert(0, ".")
from models import load_model
from data import load_train, load_test, train_val_split, CIFARDataset, eval_tf, DATA_DIR

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def loader(x):
    return DataLoader(CIFARDataset(x, transform=eval_tf), batch_size=500, shuffle=False)

@torch.no_grad()
def probs(model, x):  # softmax with hflip TTA
    model.eval(); out = []
    for xb, _ in loader(x):
        xb = xb.to(dev)
        out.append(((F.softmax(model(xb), 1) + F.softmax(model(xb.flip(-1)), 1)) / 2).cpu())
    return torch.cat(out).double()

@torch.no_grad()
def bn_adapt(model, x):  # recompute BN mean/var on these images, weights unchanged
    m = copy.deepcopy(model)
    for b in m.modules():
        if isinstance(b, torch.nn.modules.batchnorm._BatchNorm):
            b.reset_running_stats(); b.momentum = None   # cumulative average
    m.train()
    for xb, _ in loader(x):
        xb = xb.to(dev); m(xb); m(xb.flip(-1))
    return m.eval()

def balance(p, iters=50):  # Sinkhorn: each class gets N/10 total probability
    q = p.clone(); n, c = q.shape
    for _ in range(iters):
        q = q / q.sum(0, keepdim=True) * (n / c)
        q = q / q.sum(1, keepdim=True)
    return q

def variants(model, x):
    p, pb = probs(model, x), probs(bn_adapt(model, x), x)
    return {"plain": p, "balance": balance(p), "bn": pb, "bn_balance": balance(pb)}

# 1) sanity check on our val split, with V2 (trained without these 5k images)
x, y = load_train(); _, _, xva, yva = train_val_split(x, y, 5000)
m, _ = load_model("runs/V2_r3_80_1200/last.pt"); m = m.to(dev)
print("VAL (V2):")
for k, p in variants(m, xva).items():
    print(f"  {k:11s} acc {(p.argmax(1).numpy() == yva).mean():.4f}")

# 2) test predictions with F2
xt, ids = load_test()
m, _ = load_model("runs/F2_r3_80_1200_full/last.pt"); m = m.to(dev)
sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
v = variants(m, xt); plain = v["plain"].argmax(1).numpy()
print("TEST (F2):")
for k, p in v.items():
    pred = p.argmax(1).numpy()
    s = sub.copy(); s["label"] = s["id"].map(dict(zip(ids, pred.tolist())))
    s.to_csv(f"submission_F2_{k}.csv", index=False)
    print(f"  {k:11s} counts {np.bincount(pred, minlength=10).tolist()}  changed vs plain: {(pred != plain).sum()}")
