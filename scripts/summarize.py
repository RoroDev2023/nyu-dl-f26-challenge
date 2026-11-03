"""Collect every runs/*/results.json into one table, best run first, for the report.
Usage: python scripts/summarize.py [--runs runs] [--csv summary.csv]"""
import argparse
import glob
import json
import os

import pandas as pd

# flag -> value that means "off". Anything else is listed in the "tricks" column.
BASELINE = {"teachers": [], "autoaug": "none", "cutout": 0, "mixup_alpha": 0, "cutmix_alpha": 0,
            "sam": False, "asam": False, "ema_decay": 0, "se": False, "drop_path": 0,
            "dropout": 0, "val_size": 5000, "no_budget": False, "extra_aug": False,
            "geo_aug": False}


def tricks(args):
    on = []
    for key, off in BASELINE.items():
        value = args.get(key, off)
        if value != off:
            on.append(f"{key}={len(value)}" if key == "teachers" else
                      key if value is True else f"{key}={value}")
    return " ".join(on)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--csv", default=None, help="also save the table here")
    args = p.parse_args()

    rows = []
    for path in glob.glob(os.path.join(args.runs, "*", "results.json")):
        run = os.path.dirname(path)
        with open(path) as f:
            r = json.load(f)
        run_args = {}
        if os.path.exists(os.path.join(run, "args.json")):
            with open(os.path.join(run, "args.json")) as f:
                run_args = json.load(f)
        rows.append({
            "run": os.path.basename(run), "arch": r.get("arch"), "params": r.get("params"),
            "epochs": r.get("epochs"), "seed": run_args.get("seed"),
            "val": r.get("val_acc"), "val_tta": r.get("val_acc_tta"),
            "ema": r.get("val_acc_ema"), "ema_tta": r.get("val_acc_ema_tta"),
            "tricks": tricks(run_args),
        })
    if not rows:
        raise SystemExit(f"no results.json in {args.runs}/*/")

    table = pd.DataFrame(rows)
    table["best"] = table[["val", "val_tta", "ema", "ema_tta"]].astype(float).max(axis=1)
    table = table.sort_values("best", ascending=False, na_position="last")
    print(table.to_string(index=False, float_format="%.4f", na_rep="-"))
    if args.csv:
        table.to_csv(args.csv, index=False)


if __name__ == "__main__":
    main()
