"""Progress bars for training runs, read from each run's log.csv.

    python scripts/progress.py                 # every run with a log.csv under runs/
    python scripts/progress.py runs/B_kd3t_*   # specific runs
    python scripts/progress.py --watch 30      # redraw every 30 s until all finish

Nothing is inferred from the process table, so it also works for runs on another
machine/volume once their log.csv is local.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WIDTH = 22


def bar(frac: float) -> str:
    filled = int(round(frac * WIDTH))
    return "[" + "#" * filled + "-" * (WIDTH - filled) + "]"


def read_run(run_dir: Path) -> dict | None:
    log = run_dir / "log.csv"
    if not log.exists():
        return None
    rows = list(csv.DictReader(open(log)))
    if not rows:
        return None
    try:
        rows = [r for r in rows if r.get("epoch") and r["epoch"].isdigit()]
    except AttributeError:
        return None
    if not rows:
        return None
    last = rows[-1]
    epoch = int(last["epoch"])

    times = []
    for r in rows[-20:]:
        try:
            times.append(float(r["seconds"]))
        except (KeyError, ValueError, TypeError):
            pass
    sec = statistics.median(times) if times else float("nan")

    def acc(key):
        try:
            return float(last[key]) * 100
        except (KeyError, ValueError, TypeError):
            return None

    val_ema_hist = []
    for r in rows[-10:]:
        try:
            val_ema_hist.append(float(r["val_acc_ema"]) * 100)
        except (KeyError, ValueError, TypeError):
            pass
    trend = (val_ema_hist[-1] - val_ema_hist[0]) if len(val_ema_hist) >= 2 else 0.0

    return {
        "name": run_dir.name,
        "epoch": epoch,
        "seconds": sec,
        "val": acc("val_acc"),
        "val_ema": acc("val_acc_ema"),
        "trend": trend,
        "log_mtime": log.stat().st_mtime,
    }


def total_epochs(run_dir: Path) -> int:
    args_json = run_dir / "args.json"
    if args_json.exists():
        import json

        try:
            return int(json.loads(args_json.read_text()).get("epochs", 0)) or 300
        except (ValueError, TypeError):
            pass
    return 300


def draw(runs: list[Path]) -> list[dict]:
    infos = []
    for rd in runs:
        info = read_run(rd)
        if info is None:
            continue
        info["total"] = total_epochs(rd)
        infos.append(info)

    now = time.time()
    lines = []
    for info in infos:
        frac = min(1.0, info["epoch"] / info["total"])
        remaining = info["total"] - info["epoch"]
        secs = info["seconds"]
        stale = now - info["log_mtime"]
        if remaining <= 0:
            eta = "done"
        elif stale > 600:
            # no new rows for 10+ min: killed (e.g. cloud spend limit) rather than slow
            eta = f"STALLED {stale / 60:.0f} min (killed?)"
        elif secs == secs and secs > 0:  # not NaN
            done_at = time.strftime("%H:%M", time.localtime(now + remaining * secs))
            eta = f"ETA {done_at} ({remaining * secs / 3600:.1f}h)"
        else:
            eta = "ETA ?"
        flag = "" if stale <= 600 else "  <-- log stale %.0f min" % (stale / 60)
        val_ema = f"{info['val_ema']:.2f}" if info["val_ema"] is not None else "  -  "
        trend = f"{info['trend']:+.2f}/10ep" if info["trend"] else ""
        lines.append(
            f"{info['name']:<20} {bar(frac)} {info['epoch']:>3}/{info['total']:<3} "
            f"{frac * 100:5.1f}%  {secs:5.0f}s/ep  {eta:<22} val_ema {val_ema} {trend}{flag}"
        )
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="*", type=Path)
    ap.add_argument("--watch", type=int, default=0, metavar="SECONDS",
                    help="redraw every N seconds until every run reaches its last epoch")
    args = ap.parse_args()

    runs = args.runs or sorted(p for p in (ROOT / "runs").glob("*") if p.is_dir())
    runs = [r if r.is_absolute() or r.exists() else ROOT / r for r in runs]

    if not args.watch:
        out = draw(runs)
        print("\n".join(out) if out else "no log.csv found")
        return

    try:
        while True:
            lines = draw(runs)
            sys.stdout.write("\033[H\033[J")  # clear screen, no scrollback spam
            print("\n".join(lines))
            sys.stdout.flush()
            infos = [read_run(r) for r in runs]
            if all(i is None or i["epoch"] >= total_epochs(r) for i, r in zip(infos, runs)):
                break
            time.sleep(args.watch)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
