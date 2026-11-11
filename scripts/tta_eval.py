"""Cache per-view TTA probabilities once, then score any policy / comparison from the cache.
All TTA math is in tta.py (the same code tta_submit.py uses).

  # 1) cache (val = the fixed 5000 holdout, val_size=5000 seed=0; test = the 2000 Kaggle images)
  python scripts/tta_eval.py cache runs/A4_r3_80_600/last.pt probs/A4_val.npz --split val
  python scripts/tta_eval.py cache runs/A4_r3_80_600/last.pt probs/A4raw_val.npz --split val --no-ema
  python scripts/tta_eval.py cache KEEP/F2s1.pt probs/F2s1_test.npz --split test

  # 2) TTA policy experiment: tune half picks, confirm half checks
  python scripts/tta_eval.py policies probs/A4_val.npz probs/V2_val.npz

  # 3) paired comparison of models under one policy (first file = reference)
  python scripts/tta_eval.py compare --policy 5crop probs/A4_val.npz probs/C_sam_val.npz ...

  # 4) ensemble of caches (e.g. two WRN-40-2), with error overlap and total params
  python scripts/tta_eval.py ensemble --policy 5crop probs/W1_val.npz probs/W2_val.npz

  # 5) submission from a test cache (+ check against an existing csv)
  python scripts/tta_eval.py submit probs/F2s1_test.npz out.csv --policy 5crop \
      --check KEEP/submission_F2s1_tta5.csv
"""
import argparse
import hashlib
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SPLIT_SEED = 2026      # tune / confirm halves of the val set, stratified by class


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sign_p(a, b):
    """Two-sided exact sign test on discordant pairs a vs b."""
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def halves(y):
    rng = np.random.default_rng(SPLIT_SEED)
    tune = np.zeros(len(y), dtype=bool)
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        tune[idx[: len(idx) // 2]] = True
    return tune, ~tune


def load(path):
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


def nll(p, y):
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean())


# ---------------------------------------------------------------- cache
def cmd_cache(a):
    import torch
    from data import load_train, load_test, train_val_split
    from models import load_model, count_params
    from tta import view_probs, SHIFTS, SHIFTS9

    dev = torch.device("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available() else "cpu")
    model, ck = load_model(a.ckpt, use_ema=not a.no_ema)
    model = model.to(dev)
    n = count_params(model)
    used_ema = (not a.no_ema) and ck.get("ema") is not None
    if a.split == "val":
        ckval = ck.get("args", {}).get("val_size")
        if ckval != 5000:
            print(f"WARNING: checkpoint trained with val_size={ckval}: val numbers are not clean")
        x, y = load_train()
        _, _, x, y = train_val_split(x, y, 5000)
        ids = np.arange(len(y))
    else:
        x, ids = load_test()
        y = np.full(len(ids), -1)
    probs = view_probs(model, x, dev, shifts=SHIFTS9 if a.views == 18 else SHIFTS)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez(a.out, probs=probs.astype(np.float32), y=y, ids=ids, params=n,
             epoch=ck.get("epoch", -1), ema=used_ema, ckpt=a.ckpt, md5=md5(a.ckpt),
             arch=str(ck.get("arch")))
    msg = f"{a.ckpt}: {n:,} params, epoch {ck.get('epoch')}, {'EMA' if used_ema else 'raw'}, md5 {md5(a.ckpt)}"
    if a.split == "val":
        from tta import combine
        msg += (f"\n  flip acc {(combine(probs, 'flip').argmax(1) == y).mean()*100:.2f}"
                f"  all-{probs.shape[0]}-views acc {(combine(probs, 'all').argmax(1) == y).mean()*100:.2f}")
    print(msg, "->", a.out)


# ---------------------------------------------------------------- policies
def cmd_policies(a):
    from tta import combine, POLICIES
    rows = {}
    for path in a.caches:
        d = load(path)
        y = d["y"]
        tune, conf = halves(y)
        name = os.path.basename(path).replace(".npz", "")
        print(f"\n=== {name}  ({d['ckpt']}, {'EMA' if d['ema'] else 'raw'}) ===")
        print(f"{'policy':8s} {'tune acc':>9s} {'conf acc':>9s} {'all acc':>8s} {'all NLL':>8s}")
        preds = {}
        for pol in POLICIES:
            p = combine(d["probs"], pol)
            ok = p.argmax(1) == y
            preds[pol] = ok
            print(f"{pol:8s} {ok[tune].mean()*100:9.2f} {ok[conf].mean()*100:9.2f} "
                  f"{ok.mean()*100:8.2f} {nll(p, y):8.4f}")
        best = max(POLICIES, key=lambda k: (preds[k][tune].mean(), k == "5crop"))
        rows[name] = (best, preds, conf)
        print(f"picked on tune half: {best}")
        for pol in POLICIES:
            if pol == "5crop":
                continue
            fix = int((preds[pol] & ~preds["5crop"])[conf].sum())
            intro = int((~preds[pol] & preds["5crop"])[conf].sum())
            print(f"  confirm half, {pol} vs 5crop: fixes {fix}, introduces {intro}, "
                  f"net {fix-intro:+d}, sign p={sign_p(fix, intro):.3f}")
    picks = {k: v[0] for k, v in rows.items()}
    print("\npicks per model:", picks)
    if len(set(picks.values())) > 1:
        print("models disagree on the best policy -> keep 5crop (the deployed 0.911 procedure)")


# ---------------------------------------------------------------- compare
def cmd_compare(a):
    from tta import combine
    ds = [load(p) for p in a.caches]
    names = [os.path.basename(p).replace(".npz", "") for p in a.caches]
    y = ds[0]["y"]
    for d, n in zip(ds, names):
        assert (d["y"] == y).all(), f"{n}: different labels / split"
    tune, conf = halves(y)
    ok = [combine(d["probs"], a.policy).argmax(1) == y for d in ds]
    print(f"policy {a.policy}, reference {names[0]}")
    print(f"{'run':28s} {'raw/ema':7s} {'params':>10s} {'acc':>6s} {'err':>4s} {'NLL':>7s} "
          f"{'fixes':>5s} {'intro':>5s} {'net':>4s} {'p':>6s}")
    for d, n, o in zip(ds, names, ok):
        p = combine(d["probs"], a.policy)
        fix = int((o & ~ok[0]).sum())
        intro = int((~o & ok[0]).sum())
        print(f"{n:28s} {'EMA' if d['ema'] else 'raw':7s} {int(d['params']):>10,d} "
              f"{o.mean()*100:6.2f} {int((~o).sum()):4d} {nll(p, y):7.4f} "
              f"{fix:5d} {intro:5d} {fix-intro:+4d} {sign_p(fix, intro):6.3f}")
    if a.save_errors:
        np.savez(a.save_errors, names=np.array(names), correct=np.stack(ok), y=y)
        print("per-image correctness ->", a.save_errors)


# ---------------------------------------------------------------- ensemble
def cmd_ensemble(a):
    from tta import combine
    ds = [load(p) for p in a.caches]
    y = ds[0]["y"]
    total = sum(int(d["params"]) for d in ds)
    ps = [combine(d["probs"], a.policy) for d in ds]
    oks = [p.argmax(1) == y for p in ps]
    for p, o, path in zip(ps, oks, a.caches):
        print(f"{os.path.basename(path):28s} acc {o.mean()*100:.2f}  NLL {nll(p, y):.4f}")
    ens = np.mean(ps, 0)
    print(f"equal-weight prob average: acc {(ens.argmax(1) == y).mean()*100:.2f}  "
          f"NLL {nll(ens, y):.4f}")
    if len(oks) == 2:
        e1, e2 = ~oks[0], ~oks[1]
        print(f"errors: {e1.sum()} / {e2.sum()}, both wrong {int((e1 & e2).sum())}, "
              f"overlap (Jaccard) {(e1 & e2).sum() / max(1, (e1 | e2).sum()):.3f}")
    print(f"complete system params: {total:,}  ({'OK' if total <= 5_000_000 else 'OVER 5M'})")


# ---------------------------------------------------------------- submit
def cmd_submit(a):
    import pandas as pd
    from data import DATA_DIR
    from tta import combine
    d = load(a.cache)
    assert int(d["params"]) <= 5_000_000
    pred = combine(d["probs"], a.policy).argmax(1)
    sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
    assert set(sub["id"]) == set(d["ids"].tolist())
    sub["label"] = sub["id"].map(dict(zip(d["ids"].tolist(), pred.tolist())))
    sub.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(sub)} rows from {d['ckpt']} (md5 {d['md5']}), policy {a.policy}")
    if a.check:
        old = pd.read_csv(a.check).set_index("id")["label"]
        diff = int((sub.set_index("id")["label"] != old.loc[sub["id"]].values).sum())
        print(f"differences vs {a.check}: {diff}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    s = p.add_subparsers(dest="cmd", required=True)
    c = s.add_parser("cache"); c.add_argument("ckpt"); c.add_argument("out")
    c.add_argument("--split", choices=["val", "test"], required=True)
    c.add_argument("--no-ema", action="store_true")
    c.add_argument("--views", type=int, default=10, choices=[10, 18], help="18 = add diagonal shifts")
    c = s.add_parser("policies"); c.add_argument("caches", nargs="+")
    c = s.add_parser("compare"); c.add_argument("caches", nargs="+")
    c.add_argument("--policy", default="5crop"); c.add_argument("--save-errors")
    c = s.add_parser("ensemble"); c.add_argument("caches", nargs="+")
    c.add_argument("--policy", default="5crop")
    c = s.add_parser("submit"); c.add_argument("cache"); c.add_argument("out")
    c.add_argument("--policy", default="5crop"); c.add_argument("--check")
    a = p.parse_args()
    {"cache": cmd_cache, "policies": cmd_policies, "compare": cmd_compare,
     "ensemble": cmd_ensemble, "submit": cmd_submit}[a.cmd](a)


if __name__ == "__main__":
    main()
