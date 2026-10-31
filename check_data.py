import numpy as np
import matplotlib.pyplot as plt
from data import load_train, load_test, CLASSES

x, y = load_train()
xt, ids = load_test()
print("train:", x.shape, x.dtype, " labels:", y.shape)
print("test:", xt.shape, " ids:", ids[:5], "...", ids[-1])
print("images per class:", np.bincount(y))

# show 16 images with their labels, to make sure labels match the pictures
fig, axes = plt.subplots(2, 8, figsize=(14, 4))
for ax, i in zip(axes.flat, range(16)):
    ax.imshow(x[i])
    ax.set_title(CLASSES[y[i]], fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.savefig("check_data.png")
print("saved check_data.png")