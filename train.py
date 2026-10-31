import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn

from data import get_loaders
from models import ResNet, ARCHS, count_params


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += len(y)
    return correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="r3_64_3-5-3", choices=ARCHS.keys())
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default="runs/baseline")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    device = get_device()
    print("device:", device)
    train_loader, val_loader = get_loaders(args.batch_size, num_workers=args.workers)

    model = ResNet(**ARCHS[args.arch]).to(device)
    n_params = count_params(model)
    print(f"arch {args.arch}: {n_params:,} params")
    assert n_params <= 5_000_000, "over the 5M budget!"

    # no weight decay on BatchNorm weights/biases (1-D params)
    decay = [p for p in model.parameters() if p.ndim > 1]
    no_decay = [p for p in model.parameters() if p.ndim <= 1]
    optimizer = torch.optim.SGD(
        [{"params": decay, "weight_decay": args.wd},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, momentum=0.9, nesterov=True)

    # LR schedule per step: linear warmup, then cosine down to 0
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup * steps_per_epoch

    def lr_factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # bf16 mixed precision on NVIDIA GPUs only (faster); plain fp32 on Mac/CPU
    use_amp = device.type == "cuda"

    log_path = os.path.join(args.out, "log.csv")
    with open(log_path, "w") as f:
        f.write("epoch,lr,train_loss,train_acc,val_acc,seconds\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        loss_sum, correct, seen = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits.float(), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)

        val_acc = evaluate(model, val_loader, device)
        secs = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch:3d}/{args.epochs}  lr {lr:.4f}  loss {loss_sum / seen:.4f}  "
              f"train_acc {correct / seen:.4f}  val_acc {val_acc:.4f}  ({secs:.0f}s)", flush=True)
        with open(log_path, "a") as f:
            f.write(f"{epoch},{lr:.6f},{loss_sum / seen:.5f},{correct / seen:.5f},{val_acc:.5f},{secs:.1f}\n")

        torch.save({"arch": args.arch, "model": model.state_dict(), "epoch": epoch,
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
                   os.path.join(args.out, "last.pt"))

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump({"arch": args.arch, "params": n_params, "val_acc": val_acc}, f, indent=2)
    print("done. final val_acc:", val_acc)


if __name__ == "__main__":
    main()