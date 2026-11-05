"""Does the test set's brightness/contrast/sharpness difference explain the accuracy gap?
Simulate it on our own val split; also make an input-renormalized test submission."""
import os, sys, numpy as np, pandas as pd, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
sys.path.insert(0, ".")
from models import load_model
from data import load_train, load_test, train_val_split, CIFARDataset, eval_tf, DATA_DIR
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); torch.set_num_threads(os.cpu_count())

@torch.no_grad()
def probs(m, x):
    m.eval(); out = []
    for xb, _ in DataLoader(CIFARDataset(x, None, eval_tf), batch_size=500, num_workers=4):
        xb = xb.to(dev); out.append((F.softmax(m(xb), 1) + F.softmax(m(xb.flip(-1)), 1)).cpu())
    return torch.cat(out)
f = lambda a: a.astype(np.float32) / 255
u8 = lambda b: (np.clip(b, 0, 1) * 255).round().astype(np.uint8)
def restat(a, m0, s0, m1, s1): return u8((f(a) - m0) / s0 * s1 + m1)
def sharpen(a, amt):
    t = torch.from_numpy(f(a)).permute(0, 3, 1, 2)
    blur = F.avg_pool2d(F.pad(t, (1, 1, 1, 1), mode="reflect"), 3, 1)
    return u8((t + amt * (t - blur)).permute(0, 2, 3, 1).numpy())
sharp = lambda a: float(np.abs(np.diff(f(a), axis=1)).mean() + np.abs(np.diff(f(a), axis=2)).mean())

x, y = load_train(); xtr, ytr, xva, yva = train_val_split(x, y, 5000); xt, ids = load_test()
mtr, s_tr = f(xtr).mean((0, 1, 2)), f(xtr).std((0, 1, 2)); mte, s_te = f(xt).mean((0, 1, 2)), f(xt).std((0, 1, 2))
for name in ["V2_r3_80_1200", "A4_r3_80_600"]:
    m, _ = load_model(f"runs/{name}/last.pt"); m = m.to(dev)
    acc = lambda a: float((probs(m, a).argmax(1).numpy() == yva).mean())
    v_col = restat(xva, mtr, s_tr, mte, s_te)
    print(f"{name}: clean val {acc(xva):.4f} | val with test colour stats {acc(v_col):.4f}")
    for amt in (0.15, 0.3):
        vs = sharpen(v_col, amt)
        print(f"   + sharpen {amt}: sharpness {sharp(vs):.4f} (test {sharp(xt):.4f}) acc {acc(vs):.4f}")

# test-time input renormalization: map test colour stats back to train stats (uses test
# input statistics only, like BN adaptation -> disclose in the report)
m, _ = load_model("runs/F2_r3_80_1200_full/last.pt"); m = m.to(dev)
base = probs(m, xt).argmax(1).numpy(); pn = probs(m, restat(xt, mte, s_te, mtr, s_tr)).argmax(1).numpy()
sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv"); sub["label"] = sub["id"].map(dict(zip(ids, pn.tolist())))
sub.to_csv("submission_F2_inputnorm.csv", index=False)
print(f"F2 test: input-renorm changes {(pn != base).sum()} predictions; counts {np.bincount(pn, minlength=10).tolist()}")
