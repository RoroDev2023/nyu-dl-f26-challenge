"""Multi-crop TTA: average softmax over shifted crops (+ hflip). Inference only."""
import sys, numpy as np, pandas as pd, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
sys.path.insert(0, ".")
from models import load_model
from data import load_train, load_test, train_val_split, CIFARDataset, eval_tf, DATA_DIR
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
S = {"flip": [(0, 0)],
     "5crop2": [(0, 0), (2, 0), (-2, 0), (0, 2), (0, -2)],
     "9crop2": [(a, b) for a in (-2, 0, 2) for b in (-2, 0, 2)],
     "9crop4": [(a, b) for a in (-4, 0, 4) for b in (-4, 0, 4)]}

@torch.no_grad()
def probs(model, x, shifts):
    model.eval(); out = []
    for xb, _ in DataLoader(CIFARDataset(x, transform=eval_tf), batch_size=500):
        xb = xb.to(dev); p = 0
        for b in (xb, xb.flip(-1)):
            pad = F.pad(b, (4, 4, 4, 4), mode="reflect")
            for dy, dx in shifts:
                p = p + F.softmax(model(pad[:, :, 4 + dy:36 + dy, 4 + dx:36 + dx]), 1)
        out.append(p.cpu())
    return torch.cat(out)

x, y = load_train(); _, _, xva, yva = train_val_split(x, y, 5000)
for name in ["V2_r3_80_1200", "A4_r3_80_600"]:
    m, _ = load_model(f"runs/{name}/last.pt"); m = m.to(dev)
    print(name, {k: round(float((probs(m, xva, s).argmax(1).numpy() == yva).mean()), 4) for k, s in S.items()})

xt, ids = load_test(); sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
m, _ = load_model("runs/F2_r3_80_1200_full/last.pt"); m = m.to(dev)
base = probs(m, xt, S["flip"]).argmax(1).numpy()
for k, s in S.items():
    pred = probs(m, xt, s).argmax(1).numpy()
    out = sub.copy(); out["label"] = out["id"].map(dict(zip(ids, pred.tolist())))
    out.to_csv(f"submission_F2_tta_{k}.csv", index=False)
    print(f"F2 test {k}: changed vs flip-only = {(pred != base).sum()}")
