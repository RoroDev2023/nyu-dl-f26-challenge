import argparse

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import load_test, CIFARDataset, eval_tf, DATA_DIR
from models import load_model, count_params
from train import get_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--no-ema", action="store_true", help="use the raw weights, not EMA")
    p.add_argument("--no-tta", action="store_true", help="skip the hflip TTA")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    device = get_device()
    model, ckpt = load_model(args.checkpoint, use_ema=not args.no_ema)
    n_params = count_params(model)
    print(f"{args.checkpoint}: {n_params:,} params, epoch {ckpt['epoch']}, "
          f"{'EMA' if ckpt.get('ema') and not args.no_ema else 'raw'} weights")
    assert n_params <= 5_000_000, "over the 5M budget!"
    model.to(device).eval()

    x, ids = load_test()
    loader = DataLoader(CIFARDataset(x, transform=eval_tf), batch_size=500,
                        shuffle=False, num_workers=args.workers)
    preds = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            probs = F.softmax(model(images), 1)
            if not args.no_tta:
                probs = probs + F.softmax(model(images.flip(-1)), 1)
            preds.append(probs.argmax(1).cpu())
    labels = dict(zip(ids, torch.cat(preds).tolist()))

    # same rows, same order as the sample file
    submission = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
    assert set(submission["id"]) == set(labels), "ids don't match sample_submission.csv"
    submission["label"] = submission["id"].map(labels)
    submission.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(submission)} rows")


if __name__ == "__main__":
    main()
