"""Start train.py on the deployed Modal app and exit right away: the job runs in the cloud,
the laptop can sleep or go offline. Deploy first (and again after every code change):
    modal deploy modal_train.py

    python scripts/modal_launch.py launch --gpu A100-40GB --args "--out runs/X <flags>"
    python scripts/modal_launch.py status fc-...     # running / done / failed
    python scripts/modal_launch.py logs fc-...       # stream this call's logs (Ctrl-C to stop)
    python scripts/modal_launch.py cancel fc-...

Every launch is also appended to runs/modal_calls.log (local), so call ids aren't lost.
"""
import argparse
import os
import subprocess
import sys
import time

import modal

APP, FUNCTION = "resnet5m", "train"
GPU_COST = {"A100-40GB": 2.10, "H100": 3.95}    # rough $/hour
CALLS_LOG = os.path.join(os.path.dirname(__file__), "..", "runs", "modal_calls.log")


def launch(gpu, args):
    rate = GPU_COST.get(gpu)
    cost = f"~${rate:.2f}/h, 24h timeout = ~${24 * rate:.0f} max" if rate else "cost unknown"
    print(f"GPU: {gpu} ({cost})")
    print(f"train.py {args}")
    train = modal.Function.from_name(APP, FUNCTION).with_options(gpu=gpu)
    call = train.spawn(args=args)
    print(f"spawned: {call.object_id}")
    print(f"  status: python scripts/modal_launch.py status {call.object_id}")
    print(f"  logs:   python scripts/modal_launch.py logs {call.object_id}")
    os.makedirs(os.path.dirname(CALLS_LOG), exist_ok=True)
    with open(CALLS_LOG, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{call.object_id}\t{gpu}\t{args}\n")


def status(call_id):
    try:
        modal.FunctionCall.from_id(call_id).get(timeout=0)
        print("done")
    except TimeoutError:
        print("running (or still waiting for a GPU)")
    except Exception as e:      # the remote error, a cancel or a timeout
        print(f"failed: {type(e).__name__}: {e}")
        sys.exit(1)


def logs(call_id, follow):
    cmd = [sys.executable, "-m", "modal", "app", "logs", APP, "--function-call", call_id]
    cmd += ["--follow"] if follow else ["--tail", "200"]
    sys.exit(subprocess.call(cmd))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("launch")
    s.add_argument("--gpu", default="A100-40GB", help="A100-40GB or H100")
    s.add_argument("--args", required=True, help="train.py flags, one quoted string")
    for name in ["status", "logs", "cancel"]:
        s = sub.add_parser(name)
        s.add_argument("call_id", help="fc-... as printed by launch")
        if name == "logs":
            s.add_argument("--no-follow", action="store_true", help="last 200 lines, then exit")
    a = p.parse_args()

    if a.cmd == "launch":
        launch(a.gpu, a.args)
    elif a.cmd == "status":
        status(a.call_id)
    elif a.cmd == "logs":
        logs(a.call_id, not a.no_follow)
    else:
        modal.FunctionCall.from_id(a.call_id).cancel()
        print("cancelled", a.call_id)


if __name__ == "__main__":
    main()
