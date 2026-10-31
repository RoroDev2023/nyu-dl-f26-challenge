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


# training: random shift (pad 4px then crop back to 32) + random flip
train_tf = v2.Compose([
    v2.ToImage(),                                   # numpy HWC -> tensor CHW
    v2.RandomCrop(32, padding=4, padding_mode="reflect"),
    v2.RandomHorizontalFlip(),
    v2.ToDtype(torch.float32, scale=True),          # 0..255 -> 0..1
    v2.Normalize(MEAN, STD),
])

# val/test: no randomness, just convert + normalize
eval_tf = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(MEAN, STD),
])


class CIFARDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x, self.y, self.transform = x, y, transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        img = self.transform(self.x[i])
        label = -1 if self.y is None else int(self.y[i])
        return img, label


def get_loaders(batch_size=128, val_size=5000, num_workers=4):
    x, y = load_train()
    xtr, ytr, xva, yva = train_val_split(x, y, val_size)
    train_loader = DataLoader(CIFARDataset(xtr, ytr, train_tf), batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, drop_last=True,
                              persistent_workers=num_workers > 0)
    val_loader = DataLoader(CIFARDataset(xva, yva, eval_tf), batch_size=500,
                            shuffle=False, num_workers=num_workers)
    return train_loader, val_loader