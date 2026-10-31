import torch
import matplotlib.pyplot as plt
from data import get_loaders, MEAN, STD, CLASSES

if __name__ == "__main__":       # needed on Mac when num_workers > 0
    train_loader, val_loader = get_loaders(batch_size=128)
    print("train batches:", len(train_loader), " val batches:", len(val_loader))

    xb, yb = next(iter(train_loader))
    print("batch:", xb.shape, xb.dtype, " labels:", yb[:10].tolist())
    print("mean per channel:", xb.mean(dim=(0, 2, 3)).round(decimals=2).tolist(),
          " std:", xb.std(dim=(0, 2, 3)).round(decimals=2).tolist())

    # undo normalization to look at the augmented images
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    fig, axes = plt.subplots(2, 8, figsize=(14, 4))
    for ax, i in zip(axes.flat, range(16)):
        img = (xb[i] * std + mean).clamp(0, 1).permute(1, 2, 0)
        ax.imshow(img)
        ax.set_title(CLASSES[yb[i]], fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("check_loader.png")
    print("saved check_loader.png")