"""Read-only audit: configs, teacher/ensemble/student val accuracy, pipeline parity,
BN recalibration on the TRAIN split only (copy, never saved), test-vs-train input stats."""
import copy, glob, itertools, json, os, sys
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
sys.path.insert(0, ".")
from models import load_model
from data import load_train, load_test, train_val_split, CIFARDataset, eval_tf
import train as TR

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(os.cpu_count())
dl = lambda x, y=None, bs=500, sh=False: DataLoader(CIFARDataset(x, y, eval_tf), batch_size=bs, shuffle=sh, num_workers=4)

@torch.no_grad()
def probs(m, x):                      # exactly the predict.py path: eval_tf, softmax, hflip
    m.eval(); out = []
    for xb, _ in dl(x):
        xb = xb.to(dev); out.append(((F.softmax(m(xb), 1) + F.softmax(m(xb.flip(-1)), 1)) / 2).cpu())
    return torch.cat(out)
acc = lambda p, y: float((p.argmax(1).numpy() == y).mean())

x, y = load_train(); xtr, ytr, xva, yva = train_val_split(x, y, 5000)

print("== 0.1 configs of every run")
for f in sorted(glob.glob("runs/*/args.json")):
    a = json.load(open(f))
    print(f"  {os.path.dirname(f):36s} {a['arch']:16s} val_size={a.get('val_size')} seed={a.get('seed')} "
          f"ep={a['epochs']} T={len(a.get('teachers', []))} kd_a={a.get('kd_alpha')} kd_T={a.get('kd_temp')} "
          f"aug={a.get('autoaug')} cut={a.get('cutout')} mixoff={a.get('mix_off_epochs')}")

print("== 0.2 teachers / ensembles / students on the same val split (hflip TTA)")
N = {"wrn_s0": "runs/teacher_wrn_s0/last.pt", "wrn_s1": "runs/teacher_wrn_s1/last.pt",
     "pyr_s2": "runs/teacher_pyr_s2/last.pt", "A4": "runs/A4_r3_80_600/last.pt", "V2": "runs/V2_r3_80_1200/last.pt"}
P, M = {}, {}
for k, p in N.items():
    m, ck = load_model(p); M[k] = m.to(dev); P[k] = probs(M[k], xva)
    print(f"  {k:7s} ckpt val_size={ck['args'].get('val_size')} val acc={acc(P[k], yva):.4f}")
for r in (2, 3):
    for c in itertools.combinations(["wrn_s0", "wrn_s1", "pyr_s2"], r):
        print(f"  ensemble {'+'.join(c):22s} {acc(sum(P[i] for i in c), yva):.4f}")

print("== 0.3 pipeline parity on V2 (EMA)")
vl = dl(xva, yva)
print(f"  train.evaluate plain={TR.evaluate(M['V2'], vl, dev):.4f} tta={TR.evaluate(M['V2'], vl, dev, tta=True):.4f}"
      f" | predict path tta={acc(P['V2'], yva):.4f} | results.json={json.load(open('runs/V2_r3_80_1200/results.json'))}")
pc = [float((P['V2'].argmax(1).numpy()[yva == c] == c).mean()) for c in range(10)]
print("  per-class val acc:", [round(v, 3) for v in pc])

@torch.no_grad()
def recal(m, xs):                     # BN stats from TRAIN split only, eval_tf (no aug), no grads
    m = copy.deepcopy(m)
    for b in m.modules():
        if isinstance(b, torch.nn.modules.batchnorm._BatchNorm):
            b.reset_running_stats(); b.momentum = None
    m.train()
    for xb, _ in dl(xs, bs=128, sh=True):
        m(xb.to(dev))
    return m.eval()

print("== 0.4 BN recalibration on train split (copies only)")
for k in ["A4", "V2"]:
    print(f"  {k}: EMA as saved {acc(P[k], yva):.4f} -> recalibrated {acc(probs(recal(M[k], xtr), xva), yva):.4f}")
    raw, _ = load_model(N[k], use_ema=False); raw = raw.to(dev)
    print(f"  {k}: raw as saved {acc(probs(raw, xva), yva):.4f} -> recalibrated {acc(probs(recal(raw, xtr), xva), yva):.4f}")

print("== 0.5 input statistics (no labels)")
xt, _ = load_test()
def st(a):
    a = a.astype(np.float32) / 255
    sh = np.abs(np.diff(a, axis=1)).mean() + np.abs(np.diff(a, axis=2)).mean()
    sat = (a.max(-1) - a.min(-1)).mean()
    return f"mean={a.mean((0, 1, 2)).round(3)} std={a.std((0, 1, 2)).round(3)} sharp={sh:.4f} sat={sat:.4f}"
print("  train", st(xtr[:5000])); print("  val  ", st(xva)); print("  test ", st(xt)); print("  dtypes", xtr.dtype, xt.dtype, xt.shape)

print("== 0.6 shift indicators (V2 student vs 3-teacher ensemble)")
pt_s = probs(M["V2"], xt); ens_v = sum(P[i] for i in ["wrn_s0", "wrn_s1", "pyr_s2"])
ens_t = sum(probs(M[i], xt) for i in ["wrn_s0", "wrn_s1", "pyr_s2"])
for nm, ps, pe in [("val ", P["V2"], ens_v), ("test", pt_s, ens_t)]:
    print(f"  {nm}: mean max-prob={ps.max(1).values.mean():.3f}  student-vs-ensemble agreement={(ps.argmax(1) == pe.argmax(1)).float().mean():.4f}"
          f"  pred counts={np.bincount(ps.argmax(1).numpy(), minlength=10).tolist()}")
