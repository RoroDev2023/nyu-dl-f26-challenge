import torch
from models import ARCHS, TEACHER_ARCHS, build_model, count_params

x = torch.randn(4, 3, 32, 32)   # fake batch of 4 images
for name, cfg in ARCHS.items():
    for se in (False, True):
        model = build_model(dict(cfg, se=se))
        n = count_params(model)
        out = model(x)
        label = name + ("+se" if se else "")
        print(f"{label:22s} params={n:>11,}  {'OK' if n <= 5_000_000 else 'OVER 5M!'}  output={tuple(out.shape)}")

for name, cfg in TEACHER_ARCHS.items():
    model = build_model(cfg)
    print(f"{name:22s} params={count_params(model):>11,}  teacher  output={tuple(model(x).shape)}")
