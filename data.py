import os
import numpy as np
import pandas as pd
from PIL import Image

DATA_DIR = "data"
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


def load_split(csv_name, img_folder):
    """Read a CSV, open every image listed in its 'filename' column."""
    df = pd.read_csv(os.path.join(DATA_DIR, csv_name))
    images = np.stack([
        np.array(Image.open(os.path.join(DATA_DIR, img_folder, f)).convert("RGB"))
        for f in df["filename"]
    ])
    return images, df


def load_train():
    """Returns x: (50000, 32, 32, 3) uint8, y: (50000,) int64.
    Opening 50k PNGs is slow (~1 min), so save a cache the first time."""
    cache = os.path.join(DATA_DIR, "train_cache.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return d["x"], d["y"]
    x, df = load_split("train_labels.csv", "train_images")
    y = df["label"].values.astype(np.int64)
    np.savez(cache, x=x, y=y)
    return x, y


def load_test():
    """Returns x: (2000, 32, 32, 3) uint8 and the ids (for the submission)."""
    x, df = load_split("test.csv", "test_images")
    return x, df["id"].values

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def train_val_split(x, y, val_size=5000, seed=0):
    """Hold out val_size images, same number from every class."""
    rng = np.random.default_rng(seed)
    per_class = val_size // 10
    val_idx = np.concatenate([
        rng.permutation(np.where(y == c)[0])[:per_class] for c in range(10)
    ])
    train_mask = np.ones(len(y), dtype=bool)
    train_mask[val_idx] = False
    return x[train_mask], y[train_mask], x[val_idx], y[val_idx]


def get_autoaug(name):
    """Strong augmentation policy, works on uint8 images. None if name == 'none'."""
    return {
        "none": None,
        "trivial": v2.TrivialAugmentWide(),
        "randaug": v2.RandAugment(),
        "autoaug_cifar": v2.AutoAugment(v2.AutoAugmentPolicy.CIFAR10),
    }[name]


def cutout(x, size):
    """Black out (0 = mean color after Normalize) one size x size square per image of a
    batch (N, C, H, W). The center can be near the border, so the square may be clipped."""
    n, _, h, w = x.shape
    cy = torch.randint(0, h, (n, 1), device=x.device)
    cx = torch.randint(0, w, (n, 1), device=x.device)
    rows = torch.arange(h, device=x.device)[None]
    cols = torch.arange(w, device=x.device)[None]
    in_rows = (rows >= cy - size // 2) & (rows < cy - size // 2 + size)
    in_cols = (cols >= cx - size // 2) & (cols < cx - size // 2 + size)
    return x.masked_fill(in_rows[:, None, :, None] & in_cols[:, None, None, :], 0.0)


class Cutout:
    """cutout() for one image, to use in a v2.Compose."""
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        return cutout(img[None], self.size)[0]


def build_train_tf(autoaug="none", cutout_size=0):
    # training: random shift (pad 4px then crop back to 32) + random flip
    steps = [
        v2.ToImage(),                                   # numpy HWC -> tensor CHW
        v2.RandomCrop(32, padding=4, padding_mode="reflect"),
        v2.RandomHorizontalFlip(),
    ]
    if autoaug != "none":
        steps.append(get_autoaug(autoaug))
    steps += [
        v2.ToDtype(torch.float32, scale=True),          # 0..255 -> 0..1
        v2.Normalize(MEAN, STD),
    ]
    if cutout_size > 0:
        steps.append(Cutout(cutout_size))
    return v2.Compose(steps)


train_tf = build_train_tf()

# val/test: no randomness, just convert + normalize
eval_tf = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(MEAN, STD),
])


class AugMixViews:
    """AugMix + JSD views (Hendrycks et al. 2020, google-research/augmix): one shared
    crop+flip, then the clean view and two independent AugMix views of it.
    Returns a (3, C, H, W) float tensor. all_ops=False as in the reference code (no
    brightness/contrast/color/sharpness ops, which overlap the corruption benchmarks)."""
    def __init__(self, severity=3, width=3, depth=-1, all_ops=False):
        self.geo = v2.Compose([v2.ToImage(), v2.RandomCrop(32, padding=4, padding_mode="reflect"),
                               v2.RandomHorizontalFlip()])
        self.augmix = v2.AugMix(severity=severity, mixture_width=width, chain_depth=depth,
                                alpha=1.0, all_ops=all_ops)
        self.norm = v2.Compose([v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD)])

    def __call__(self, img):
        x = self.geo(img)
        return torch.stack([torch.as_tensor(self.norm(x)), torch.as_tensor(self.norm(self.augmix(x))),
                            torch.as_tensor(self.norm(self.augmix(x)))])


class CIFARDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x, self.y, self.transform = x, y, transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        img = self.transform(self.x[i])
        label = -1 if self.y is None else int(self.y[i])
        return img, label


class GPUTrainLoader:
    """Replaces the training DataLoader for --gpu-aug: keeps the whole train set on the
    GPU and augments whole batches there, in the same order as train_tf. No CPU workers,
    so the data pipeline can't starve the GPU. Yields batches already on the device."""
    def __init__(self, x, y, batch_size, device, autoaug="none", cutout_size=0, chunk=16):
        # reflect-pad once up front; a random crop is then just a random 32x32 window
        x = np.pad(x, ((0, 0), (4, 4), (4, 4), (0, 0)), mode="reflect")
        self.x = torch.from_numpy(x).permute(0, 3, 1, 2).to(device)
        self.y = torch.from_numpy(y).to(device)
        self.batch_size, self.device = batch_size, device
        self.autoaug, self.cutout_size, self.chunk = get_autoaug(autoaug), cutout_size, chunk
        self.mean = torch.tensor(MEAN, device=device)[None, :, None, None]
        self.std = torch.tensor(STD, device=device)[None, :, None, None]

    def __len__(self):
        return len(self.y) // self.batch_size      # drop_last

    def augment(self, x):
        n = len(x)
        # random crop: per-image window offsets 0..8
        dy = torch.randint(0, 9, (n, 1), device=self.device)
        dx = torch.randint(0, 9, (n, 1), device=self.device)
        rows = dy + torch.arange(32, device=self.device)
        cols = dx + torch.arange(32, device=self.device)
        batch = torch.arange(n, device=self.device)[:, None, None]
        x = x[batch, :, rows[:, :, None], cols[:, None, :]].permute(0, 3, 1, 2)
        # random flip
        flip = torch.rand(n, device=self.device) < 0.5
        x = torch.where(flip[:, None, None, None], x.flip(-1), x)
        # the policies draw one op per call, so apply them to small chunks, not the batch
        if self.autoaug is not None:
            x = torch.cat([self.autoaug(c) for c in x.split(self.chunk)])
        x = (x.float() / 255 - self.mean) / self.std
        if self.cutout_size > 0:
            x = cutout(x, self.cutout_size)
        return x

    def __iter__(self):
        perm = torch.randperm(len(self.y), device=self.device)
        for i in range(len(self)):
            idx = perm[i * self.batch_size:(i + 1) * self.batch_size]
            yield self.augment(self.x[idx]), self.y[idx]


def get_loaders(batch_size=128, val_size=5000, num_workers=4, autoaug="none", cutout_size=0,
                gpu_aug=False, device=None, aug_chunk=16, augmix_views=False, augmix_severity=3):
    """val_loader is None when val_size == 0 (final run on all 50k).
    The split seed stays fixed on purpose: every student and teacher, whatever its --seed,
    must hold out the same images, or teachers would have trained on the student's val."""
    x, y = load_train()
    xtr, ytr, xva, yva = train_val_split(x, y, val_size)
    if augmix_views:
        train_loader = DataLoader(CIFARDataset(xtr, ytr, AugMixViews(augmix_severity)),
                                  batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  drop_last=True, pin_memory=torch.cuda.is_available())
    elif gpu_aug:
        train_loader = GPUTrainLoader(xtr, ytr, batch_size, device, autoaug, cutout_size,
                                      aug_chunk)
    else:
        # no persistent workers: fresh workers get their seeds from the main process every
        # epoch, which is what lets a resumed run continue with the same random numbers
        train_loader = DataLoader(CIFARDataset(xtr, ytr, build_train_tf(autoaug, cutout_size)),
                                  batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  drop_last=True, pin_memory=torch.cuda.is_available())
    val_loader = None
    if val_size > 0:
        val_loader = DataLoader(CIFARDataset(xva, yva, eval_tf), batch_size=500,
                                shuffle=False, num_workers=num_workers)
    return train_loader, val_loader
