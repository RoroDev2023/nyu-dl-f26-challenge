"""Compare A4 vs FIN_ctrl vs FIN_light on the fixed val split.
Flip TTA and 5-crop TTA (2px shifts x flip). Reports which val errors each
finish fixes / introduces vs A4, and a sign test light vs ctrl.
Run from repo root:  python scripts/fin_eval.py
"""
import os, sys, math
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import load_train, train_val_split, MEAN, STD
from models import load_model

RUNS = {
    "A4": "runs/A4_r3_80_600/last.pt",
    "FIN_ctrl": "runs/FIN_ctrl_A4/last.pt",
    "FIN_light": "runs/FIN_light_A4/last.pt",
}
dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# --- val split (same as training: val_size=5000, seed=0) ---
x, y = load_train()
parts = train_val_split(x, y, val_size=5000, seed=0)
arrs = [np.asarray(p) for p in parts]
imgs = [a for a in arrs if a.ndim == 4]
labs = [a for a in arrs if a.ndim == 1]
xv = min(imgs, key=len); yv = min(labs, key=len)
assert len(xv) == len(yv) == 5000, (xv.shape, yv.shape)

xv = torch.tensor(xv).float() / 255.0
if xv.shape[-1] == 3:
    xv = xv.permute(0, 3, 1, 2)
mean = torch.tensor(MEAN).view(1, 3, 1, 1); std = torch.tensor(STD).view(1, 3, 1, 1)
xv = ((xv - mean) / std).contiguous()
yv = torch.tensor(yv).long()

def views(xb, mode):
    out = [xb, xb.flip(3)]
    if mode == "5crop":
        p = F.pad(xb, (2, 2, 2, 2), mode="reflect")
        for dy, dx in [(0, 0), (0, 4), (4, 0), (4, 4)]:
            c = p[:, :, dy:dy + 32, dx:dx + 32]
            out += [c, c.flip(3)]
    return out

@torch.no_grad()
def predict(model, mode):
    model.eval(); preds = []
    for i in range(0, len(xv), 500):
        xb = xv[i:i + 500].to(dev)
        prob = sum(F.softmax(model(v), 1) for v in views(xb, mode))
        preds.append(prob.argmax(1).cpu())
    return torch.cat(preds)

def sign_p(a, b):  # two-sided exact sign test
    n = a + b
    if n == 0: return 1.0
    k = min(a, b)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)

models = {}
for name, path in RUNS.items():
    m, _ = load_model(path)  # EMA weights if present
    models[name] = m.to(dev)
    print(f"loaded {name}: {sum(p.numel() for p in m.parameters()):,} params")

for mode in ["flip", "5crop"]:
    print(f"\n=== TTA: {mode} ===")
    ok = {n: (predict(m, mode) == yv) for n, m in models.items()}
    for n, c in ok.items():
        print(f"{n:10s} acc {c.float().mean().item()*100:.2f}  errors {int((~c).sum())}")
    for n in ["FIN_ctrl", "FIN_light"]:
        fix = int((ok[n] & ~ok["A4"]).sum()); intro = int((~ok[n] & ok["A4"]).sum())
        print(f"{n} vs A4: fixes {fix}, introduces {intro}, net {fix-intro:+d}, sign p={sign_p(fix,intro):.3f}")
    lw = int((ok["FIN_light"] & ~ok["FIN_ctrl"]).sum()); cw = int((ok["FIN_ctrl"] & ~ok["FIN_light"]).sum())
    print(f"light vs ctrl: light-only right {lw}, ctrl-only right {cw}, net {lw-cw:+d}, sign p={sign_p(lw,cw):.3f}")
