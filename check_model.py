import torch
from models import ResNet, count_params

configs = {
    "r3_64_3-5-3":   dict(channels=(64, 128, 256), blocks=(3, 5, 3)),
    "r3_80_2-3-3":   dict(channels=(80, 160, 256), blocks=(2, 3, 3)),
    "r3_96_2-2-2":   dict(channels=(96, 192, 320), blocks=(2, 2, 2)),
    "r4_64_3-3-2-1": dict(channels=(64, 128, 256, 320), blocks=(3, 3, 2, 1)),
    "r3_40_8-8-8":   dict(channels=(40, 80, 160), blocks=(8, 8, 8)),
}

x = torch.randn(4, 3, 32, 32)   # fake batch of 4 images
for name, cfg in configs.items():
    model = ResNet(**cfg)
    n = count_params(model)
    out = model(x)
    print(f"{name:15s} params={n:>10,}  {'OK' if n <= 5_000_000 else 'OVER 5M!'}  output={tuple(out.shape)}")