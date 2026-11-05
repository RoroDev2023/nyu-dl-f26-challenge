"""Exact architecture summary and trainable-parameter count of checkpoints.
python scripts/param_counts.py teachers/*.pt ckpts/*.pt"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import load_model, count_params, build_model, TEACHER_ARCHS
for path in sys.argv[1:]:
    m, ck = load_model(path)
    n = count_params(m)
    print(f"{path}: arch {ck.get('arch')}, config {ck.get('config')}, {n:,} params "
          f"-> {'ELIGIBLE (<=5M)' if n <= 5_000_000 else f'{n/4.8e6:.1f}x over a 4.8M target'}")
n = count_params(build_model(TEACHER_ARCHS["wrn_40_2"]))
print(f"wrn_40_2 (fresh): {n:,} params, two of them: {2*n:,}")
