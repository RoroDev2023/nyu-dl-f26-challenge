"""The one TTA implementation, used for both validation (scripts/tta_eval.py) and
submissions (scripts/tta_submit.py). Copied exactly from the scripts/tta_submit.py that
made the 0.911 F2s1 submission:
  - input: eval_tf (ToImage, /255, Normalize(MEAN, STD)) -- no other preprocessing
  - reflect-pad 4 px (of the normalized image), then 32x32 windows shifted by
    SHIFTS = (0,0), (2,0), (-2,0), (0,2), (0,-2)  as (dy, dx): a plus shape, not corners
  - the same 5 windows of the h-flipped image
  - softmax per view, probabilities added (not logits)
View order in caches: [orig s0..s4, flip s0..s4]; view 0 = plain image, view 5 = its flip.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import CIFARDataset, eval_tf

SHIFTS = [(0, 0), (2, 0), (-2, 0), (0, 2), (0, -2)]
# 18-view variant: the 5 above plus the 4 diagonals (+-2, +-2), each with its flip
SHIFTS9 = SHIFTS + [(2, 2), (2, -2), (-2, 2), (-2, -2)]
N_VIEWS = 2 * len(SHIFTS)
CENTER_VIEWS = (0, 5)          # plain image and its flip

# view weights, each sums to 1. "flip" = center + flip only. "5crop" = the 0.911 procedure.
# cf50 / cf80: center + flip get 50% / 80% of the total weight, the 8 shifted views share
# the rest equally.
POLICIES = {
    "flip": [0.5 if v in CENTER_VIEWS else 0.0 for v in range(N_VIEWS)],
    "5crop": [1 / N_VIEWS] * N_VIEWS,
    "cf50": [0.25 if v in CENTER_VIEWS else 0.5 / 8 for v in range(N_VIEWS)],
    "cf80": [0.40 if v in CENTER_VIEWS else 0.2 / 8 for v in range(N_VIEWS)],
}


def views(xb, shifts=SHIFTS):
    out = []
    for b in (xb, xb.flip(-1)):
        pad = F.pad(b, (4, 4, 4, 4), mode="reflect")
        for dy, dx in shifts:
            out.append(pad[:, :, 4 + dy:36 + dy, 4 + dx:36 + dx])
    return out


@torch.no_grad()
def view_probs(model, x_uint8, device, batch_size=250, shifts=SHIFTS):
    """x_uint8: (N, 32, 32, 3) uint8. Returns float32 array (N_VIEWS, N, 10) of softmax."""
    model.eval()
    out = []
    for xb, _ in DataLoader(CIFARDataset(x_uint8, None, eval_tf), batch_size=batch_size):
        xb = xb.to(device)
        out.append(torch.stack([F.softmax(model(v).float(), 1) for v in views(xb, shifts)]).cpu())
    return torch.cat(out, 1).numpy()


def combine(probs, policy):
    """probs (N_VIEWS, N, 10) -> (N, 10) weighted average for a policy name or weights."""
    if isinstance(policy, str) and policy == "all":       # equal weight over every cached view
        return probs.astype(np.float64).mean(0)
    if isinstance(policy, str) and policy == "flip":      # plain view + its flip, any cache size
        n = probs.shape[0]
        return (probs[0].astype(np.float64) + probs[n // 2]) / 2
    w = np.asarray(POLICIES[policy] if isinstance(policy, str) else policy, dtype=np.float64)
    return np.tensordot(w, probs.astype(np.float64), axes=1)
