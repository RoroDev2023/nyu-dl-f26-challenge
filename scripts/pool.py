"""Val accuracy of candidate teacher pools (all members held out the same 5k val split)."""
import os, sys, numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
sys.path.insert(0, ".")
from models import load_model
from data import load_train, train_val_split, CIFARDataset, eval_tf
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); torch.set_num_threads(os.cpu_count())
x, y = load_train(); _, _, xva, yva = train_val_split(x, y, 5000)
@torch.no_grad()
def probs(m):
    m.eval(); out = []
    for xb, _ in DataLoader(CIFARDataset(xva, None, eval_tf), batch_size=500, num_workers=4):
        xb = xb.to(dev); out.append(((F.softmax(m(xb), 1) + F.softmax(m(xb.flip(-1)), 1)) / 2).cpu())
    return torch.cat(out)
R = {"wrn_s0": "teacher_wrn_s0", "wrn_s1": "teacher_wrn_s1", "pyr": "teacher_pyr_s2",
     "A4": "A4_r3_80_600", "V2": "V2_r3_80_1200", "kd2t": "kd_r3_64_2t_600",
     "sw80": "A_sweep_kd3t_300_r3_80_2-3-3", "sw96": "A_sweep_kd3t_300_r3_96_2-2-2"}
P = {}
for k, r in R.items():
    m, ck = load_model(f"runs/{r}/last.pt"); assert ck["args"]["val_size"] == 5000
    P[k] = probs(m.to(dev)); print(f"{k:7s} {(P[k].argmax(1).numpy() == yva).mean():.4f}")
acc = lambda ks: (sum(P[k] for k in ks).argmax(1).numpy() == yva).mean()
for ks in [["wrn_s0", "wrn_s1", "pyr"], ["wrn_s0", "wrn_s1"],
           ["wrn_s0", "wrn_s1", "pyr", "A4", "V2"], ["wrn_s0", "wrn_s1", "A4", "V2"],
           ["wrn_s0", "wrn_s1", "pyr", "A4", "V2", "kd2t", "sw80", "sw96"],
           ["A4", "V2", "kd2t", "sw80", "sw96"]]:
    print(f"pool {'+'.join(ks):45s} {acc(ks):.4f}")
