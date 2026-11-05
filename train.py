import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from data import get_loaders
from models import ARCHS, TEACHER_ARCHS, build_model, load_model, count_params
from tricks import EMA, mix_batch, mixed_ce, teacher_probs, kd_loss, sam_perturb, sam_restore

# flags that can change between a run and its resume without changing the result
RESUME_SAFE_ARGS = {"workers", "save_every", "compile", "device"}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)     # also seeds CUDA and MPS


def get_rng_state(device):
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch": torch.get_rng_state()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state()
    elif device.type == "mps":
        state["mps"] = torch.mps.get_rng_state()
    return state


def set_rng_state(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device.type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"])
    elif device.type == "mps" and "mps" in state:
        torch.mps.set_rng_state(state["mps"])


def get_amp(device):
    """Mixed precision on NVIDIA GPUs only: bf16 if the GPU has it, else fp16 + GradScaler.
    Plain fp32 on Mac/CPU. Returns (autocast dtype or None, scaler). The scaler does
    nothing when disabled, so the training loop is the same in all cases."""
    dtype = None
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return dtype, torch.amp.GradScaler(device.type, enabled=dtype == torch.float16)


def save_atomic(state, path):
    """A kill during torch.save only ever damages the tmp file, never last.pt."""
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def evaluate(model, loader, device, tta=False):
    """Accuracy. tta: average the predictions of each image and its mirror image."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if device.type == "cuda":
                x = x.to(memory_format=torch.channels_last)
            pred = F.softmax(model(x), 1)
            if tta:
                pred = pred + F.softmax(model(x.flip(-1)), 1)
            correct += (pred.argmax(1) == y).sum().item()
            total += len(y)
    return correct / total


def check_resume_args(saved, current, defaults=None):
    """Refuse to continue a run with different settings: it would silently be another run.
    A flag added after the run started counts as its default value."""
    defaults = defaults or {}
    changed = {k: (saved.get(k, defaults.get(k)), v) for k, v in current.items()
               if k not in RESUME_SAFE_ARGS and saved.get(k, defaults.get(k)) != v}
    if changed:
        raise SystemExit(f"{current['out']} holds a run with other settings, use a new --out "
                         f"(saved, now): {changed}")


def trim_log(log_path, last_epoch):
    """Drop log rows from epochs after the checkpoint: they will be trained again."""
    with open(log_path) as f:
        lines = f.readlines()
    with open(log_path, "w") as f:
        f.writelines(lines[:1] + [l for l in lines[1:] if int(l.split(",")[0]) <= last_epoch])


def fmt(acc):
    return "" if acc is None else f"{acc:.5f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="r3_64_3-5-3", choices=[*ARCHS, *TEACHER_ARCHS])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default="runs/baseline")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-size", type=int, default=5000, help="0 = train on all 50k")
    p.add_argument("--no-budget", action="store_true", help="teachers: skip the 5M assert")
    # knowledge distillation
    p.add_argument("--teachers", nargs="*", default=[], help="teacher checkpoints")
    p.add_argument("--kd-alpha", type=float, default=0.9)
    p.add_argument("--kd-temp", type=float, default=4)
    p.add_argument("--kd-mode", default="kl", choices=["kl", "lskd"],
                   help="lskd: z-score teacher and student logits (Sun et al. 2024). Its loss "
                        "is --ce-weight * CE + --kd-weight * KL * T^2 (not --kd-alpha)")
    p.add_argument("--kd-weight", type=float, default=9.0, help="lskd only (authors: 9)")
    p.add_argument("--ce-weight", type=float, default=0.1, help="lskd only (authors: 0.1)")
    # augmentation
    p.add_argument("--autoaug", default="none",
                   choices=["none", "trivial", "randaug", "autoaug_cifar"])
    p.add_argument("--cutout", type=int, default=0, help="square size in pixels, 0 = off")
    p.add_argument("--mixup-alpha", type=float, default=0)
    p.add_argument("--cutmix-alpha", type=float, default=0)
    p.add_argument("--mix-prob", type=float, default=1.0, help="chance to mix each batch")
    p.add_argument("--mix-off-epochs", type=int, default=0, help="no mixing for the last N")
    p.add_argument("--gpu-aug", action="store_true", help="augment batches on the GPU")
    p.add_argument("--aug-chunk", type=int, default=16, help="--gpu-aug: images per policy draw")
    # SAM, EMA
    p.add_argument("--sam", action="store_true")
    p.add_argument("--sam-rho", type=float, default=0.05,
                   help="with --asam use a larger rho, the ASAM paper uses 0.5 on CIFAR-10")
    p.add_argument("--asam", action="store_true", help="adaptive SAM (implies --sam)")
    p.add_argument("--asam-eta", type=float, default=0.01, help="ASAM eta (authors: 0.01)")
    p.add_argument("--ema-decay", type=float, default=0, help="0 = off")
    # architecture options (ResNet students only)
    p.add_argument("--se", action="store_true")
    p.add_argument("--drop-path", type=float, default=0)
    p.add_argument("--dropout", type=float, default=0)
    # infrastructure
    p.add_argument("--save-every", type=int, default=1, help="checkpoint every N epochs")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--limit-batches", type=int, default=0, help="debug: stop epochs early")
    p.add_argument("--device", default=None, help="default: cuda, else mps, else cpu")
    p.add_argument("--init-from", default=None, help="start a new run from these weights (EMA if present)")
    # LR schedule. cyclic (for SWA): --cycle-epochs long cycles, each going linearly from
    # --lr down to --lr-min; the raw weights at the end of every cycle are saved to
    # <out>/swa_cycle<k>.pt for scripts/swa_build.py
    p.add_argument("--lr-schedule", default="cosine", choices=["cosine", "cyclic"])
    p.add_argument("--cycle-epochs", type=int, default=10)
    p.add_argument("--lr-min", type=float, default=0.001)
    args = p.parse_args()
    defaults = vars(p.parse_args([]))
    if args.lr_schedule == "cyclic" and args.epochs % args.cycle_epochs:
        p.error("--epochs must be a multiple of --cycle-epochs")

    if args.arch in ARCHS:
        config = dict(ARCHS[args.arch], se=args.se, drop_path=args.drop_path,
                      dropout=args.dropout)
    elif args.se or args.drop_path or args.dropout:
        p.error("--se, --drop-path and --dropout only apply to ResNet students")
    else:
        config = dict(TEACHER_ARCHS[args.arch])

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    set_seed(args.seed)
    device = torch.device(args.device) if args.device else get_device()
    print("device:", device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True     # images are all the same size
    train_loader, val_loader = get_loaders(args.batch_size, args.val_size, args.workers,
                                           args.autoaug, args.cutout, args.gpu_aug, device,
                                           args.aug_chunk)

    raw_model = build_model(config).to(device)
    if args.init_from and not os.path.exists(os.path.join(args.out, "last.pt")):
        _w, _ = load_model(args.init_from)   # EMA weights if present; source file is only read
        raw_model.load_state_dict(_w.state_dict())
        print("initialized from", args.init_from)
    n_params = count_params(raw_model)
    print(f"arch {args.arch}: {n_params:,} params")
    assert args.no_budget or n_params <= 5_000_000, "over the 5M budget!"
    if device.type == "cuda":
        raw_model.to(memory_format=torch.channels_last)
    # checkpoints and EMA use raw_model: a compiled model renames all its weights
    model = torch.compile(raw_model) if args.compile else raw_model
    ema = EMA(raw_model, args.ema_decay) if args.ema_decay > 0 else None

    teachers = []
    for path in args.teachers:
        teacher, ckpt = load_model(path)
        print(f"teacher {path}: {ckpt['arch']}, {count_params(teacher):,} params")
        teachers.append(teacher.to(device).eval().requires_grad_(False))
        if args.val_size > 0 and ckpt.get("args", {}).get("val_size") != args.val_size:
            print("WARNING: this teacher did not hold out the same val set, "
                  "so val accuracy of this run can't be trusted")

    # no weight decay on BatchNorm weights/biases (1-D params)
    decay = [p for p in raw_model.parameters() if p.ndim > 1]
    no_decay = [p for p in raw_model.parameters() if p.ndim <= 1]
    # SAM/ASAM perturb these; ASAM scales by |w| + eta only params named "*weight*"
    names = {id(p): n for n, p in raw_model.named_parameters()}
    sam_params = decay + no_decay
    sam_is_weight = ["weight" in names[id(p)] for p in sam_params]
    optimizer = torch.optim.SGD(
        [{"params": decay, "weight_decay": args.wd},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, momentum=0.9, nesterov=True)

    # LR schedule per step: linear warmup, then cosine down to 0
    steps_per_epoch = len(train_loader)
    if args.limit_batches:
        steps_per_epoch = min(steps_per_epoch, args.limit_batches)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup * steps_per_epoch

    def lr_factor(step):
        if args.lr_schedule == "cyclic":
            # linear from 1 to lr_min/lr inside each cycle; reaches lr_min on its last step
            cycle_steps = args.cycle_epochs * steps_per_epoch
            pos = (step % cycle_steps + 1) / cycle_steps
            return 1 - pos * (1 - args.lr_min / args.lr)
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    label_smoothing = 0.0 if teachers else args.label_smoothing

    amp_dtype, scaler = get_amp(device)
    print("mixed precision:", amp_dtype)

    def autocast():
        return torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None)

    stand = args.kd_mode == "lskd"
    if teachers:
        w_kd, w_ce = ((args.kd_weight, args.ce_weight) if stand
                      else (args.kd_alpha, 1 - args.kd_alpha))
        print(f"loss = {w_ce:g} * CE + {w_kd:g} * KL(T={args.kd_temp:g}) * T^2, "
              f"logit standardization: {stand}")

    def compute_loss(logits, y_a, y_b, lam, probs):
        """Returns (total, weighted CE part, weighted KD part)."""
        hard = mixed_ce(logits, y_a, y_b, lam, label_smoothing)
        if not teachers:
            return hard, hard.detach(), torch.zeros_like(hard)
        ce_part = w_ce * hard
        kd_part = w_kd * kd_loss(logits, probs, args.kd_temp, stand)
        return ce_part + kd_part, ce_part.detach(), kd_part.detach()

    # resume if this folder already has a checkpoint
    last_path = os.path.join(args.out, "last.pt")
    log_path = os.path.join(args.out, "log.csv")
    start_epoch = 0
    if os.path.exists(last_path):
        # on the CPU first: the RNG states must stay there
        ckpt = torch.load(last_path, map_location="cpu", weights_only=False)
        if "args" not in ckpt:
            raise SystemExit(f"{last_path} has an old format, use a new --out")
        check_resume_args(ckpt["args"], vars(args), defaults)
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        if ema:
            ema.load_state_dict(ckpt["ema"])
        set_rng_state(ckpt["rng"], device)
        start_epoch = ckpt["epoch"]
        trim_log(log_path, start_epoch)
        print(f"resumed from epoch {start_epoch}")
    else:
        with open(log_path, "w") as f:
            f.write("epoch,lr,train_loss,train_acc,val_acc,val_acc_ema,seconds,data_seconds,"
                    "ce_part,kd_part\n")

    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        t0 = time.time()
        data_secs = 0.0
        # running [sum of loss, sum of correct] stays on the GPU: .item() would sync every step
        totals = torch.zeros(4, device=device)    # loss, correct, ce part, kd part
        seen = 0
        mix_on = epoch <= args.epochs - args.mix_off_epochs
        batches = iter(train_loader)
        for _ in range(steps_per_epoch):
            t_data = time.time()
            x, y = next(batches)
            data_secs += time.time() - t_data
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            y_a, y_b, lam = y, y, 1.0
            if mix_on:
                x, y_a, y_b, lam = mix_batch(x, y, args.mixup_alpha, args.cutmix_alpha,
                                             args.mix_prob)
            if device.type == "cuda":
                x = x.to(memory_format=torch.channels_last)

            with autocast():
                # teachers see exactly the student's augmented + mixed batch
                probs = teacher_probs(teachers, x, args.kd_temp, stand) if teachers else None
                logits = model(x)
                loss, ce_part, kd_part = compute_loss(logits.float(), y_a, y_b, lam, probs)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            if args.sam or args.asam:
                # second pass at the perturbed weights; its gradients make the real step.
                # BatchNorm running stats must come from the first pass only. Copying them
                # back also works with --compile (changing bn.momentum might not).
                params, backup = sam_perturb(sam_params, args.sam_rho, args.asam,
                                             scaler.get_scale(), args.asam_eta, sam_is_weight)
                bn_stats = [b.clone() for b in raw_model.buffers()]
                optimizer.zero_grad(set_to_none=True)
                with autocast():
                    loss2 = compute_loss(model(x).float(), y_a, y_b, lam, probs)[0]
                scaler.scale(loss2).backward()
                sam_restore(params, backup)
                for b, saved in zip(raw_model.buffers(), bn_stats):
                    b.copy_(saved)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if ema:
                ema.update(raw_model)

            pred = logits.detach().argmax(1)
            correct = lam * (pred == y_a).sum() + (1 - lam) * (pred == y_b).sum()
            totals += torch.stack([loss.detach().float() * len(y), correct.float(),
                                   ce_part.float() * len(y), kd_part.float() * len(y)])
            seen += len(y)

        val_acc = evaluate(raw_model, val_loader, device) if val_loader else None
        ema_acc = evaluate(ema.module, val_loader, device) if val_loader and ema else None
        secs = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        loss_avg, acc_avg, ce_avg, kd_avg = (totals / seen).tolist()
        print(f"epoch {epoch:3d}/{args.epochs}  lr {lr:.4f}  loss {loss_avg:.4f} "
              f"(ce {ce_avg:.4f} kd {kd_avg:.4f})  "
              f"train_acc {acc_avg:.4f}  val_acc {fmt(val_acc)}  val_ema {fmt(ema_acc)}  "
              f"({secs:.0f}s, {data_secs:.0f}s waiting for data)", flush=True)
        with open(log_path, "a") as f:
            f.write(f"{epoch},{lr:.6f},{loss_avg:.5f},{acc_avg:.5f},{fmt(val_acc)},{fmt(ema_acc)},"
                    f"{secs:.1f},{data_secs:.1f},{ce_avg:.5f},{kd_avg:.5f}\n")

        if args.lr_schedule == "cyclic" and epoch % args.cycle_epochs == 0:
            # raw weights + buffers at the lowest LR of this cycle, for SWA
            k = epoch // args.cycle_epochs
            save_atomic({"arch": args.arch, "config": config, "args": vars(args),
                         "model": raw_model.state_dict(), "epoch": epoch, "ema": None,
                         "cycle": k}, os.path.join(args.out, f"swa_cycle{k}.pt"))

        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_atomic({"arch": args.arch, "config": config, "args": vars(args),
                         "model": raw_model.state_dict(), "epoch": epoch,
                         "ema": ema.state_dict() if ema else None,
                         "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                         "scaler": scaler.state_dict(), "rng": get_rng_state(device)}, last_path)

    results = {"arch": args.arch, "params": n_params, "epochs": args.epochs}
    for name, net in [("val_acc", raw_model), ("val_acc_ema", ema.module if ema else None)]:
        have_val = net is not None and val_loader is not None
        results[name] = evaluate(net, val_loader, device) if have_val else None
        results[name + "_tta"] = evaluate(net, val_loader, device, tta=True) if have_val else None
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("done.", results)


if __name__ == "__main__":
    main()
