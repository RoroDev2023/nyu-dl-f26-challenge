import torch
import torch.nn as nn
import torch.nn.functional as F


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class SEBlock(nn.Module):
    """Squeeze-excitation: reweight channels using a global summary of the image."""
    def __init__(self, ch, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(ch, ch // reduction)
        self.fc2 = nn.Linear(ch // reduction, ch)

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1).flatten(1)
        w = torch.sigmoid(self.fc2(F.relu(self.fc1(w))))
        return x * w[:, :, None, None]


class DropPath(nn.Module):
    """Stochastic depth: randomly drop F(x) per image, so the block acts as S(x) alone.
    Kept images are scaled by 1/keep so the expected output is unchanged."""
    def __init__(self, rate):
        super().__init__()
        self.keep = 1 - rate

    def forward(self, x):
        if not self.training or self.keep == 1:
            return x
        mask = x.new_empty(x.size(0), 1, 1, 1).bernoulli_(self.keep)
        return x * mask / self.keep


class BasicBlock(nn.Module):
    """out = relu( F(x) + S(x) )"""
    def __init__(self, in_ch, out_ch, stride=1, se=False, drop_path=0.0):
        super().__init__()
        # F(x): conv -> BN -> ReLU -> conv -> BN
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        nn.init.zeros_(self.bn2.weight)   # F(x) starts at 0 -> block starts as identity
        self.se = SEBlock(out_ch) if se else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        # S(x): identity, or 1x1 conv when the shape changes
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.drop_path(self.se(self.bn2(self.conv2(out))))
        return F.relu(out + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, channels=(64, 128, 256), blocks=(3, 5, 3), num_classes=10,
                 se=False, drop_path=0.0, dropout=0.0):
        super().__init__()
        # stem: 3 colors -> channels[0] features, keep 32x32 (no maxpool)
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(),
        )
        # stages: first block of every stage after the first halves the image (stride 2)
        # drop_path rate grows linearly: 0 at the first block, drop_path at the last
        layers = []
        in_ch = channels[0]
        total = sum(blocks)
        for stage, (ch, n) in enumerate(zip(channels, blocks)):
            for b in range(n):
                stride = 2 if (b == 0 and stage > 0) else 1
                rate = drop_path * len(layers) / max(1, total - 1)
                layers.append(BasicBlock(in_ch, ch, stride, se, rate))
                in_ch = ch
        self.layers = nn.Sequential(*layers)
        # head: global average pool -> (dropout) -> linear
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(in_ch, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = self.layers(self.stem(x))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(self.dropout(x))


class WideBlock(nn.Module):
    """Pre-activation block: out = F(x) + S(x), F = BN-ReLU-conv-BN-ReLU-conv."""
    def __init__(self, in_ch, out_ch, stride=1, dropout=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)

        self.shortcut = None
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)

    def forward(self, x):
        pre = F.relu(self.bn1(x))
        out = self.conv1(pre)
        out = self.conv2(self.dropout(F.relu(self.bn2(out))))
        # when the shape changes the shortcut shares the pre-activation
        return out + (x if self.shortcut is None else self.shortcut(pre))


class WideResNet(nn.Module):
    """WRN-depth-widen (Zagoruyko & Komodakis 2016). Stages have 16k, 32k, 64k channels."""
    def __init__(self, depth=28, widen=10, dropout=0.0, num_classes=10):
        super().__init__()
        n = (depth - 4) // 6
        self.stem = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        layers = []
        in_ch = 16
        for stage, ch in enumerate((16 * widen, 32 * widen, 64 * widen)):
            for b in range(n):
                stride = 2 if (b == 0 and stage > 0) else 1
                layers.append(WideBlock(in_ch, ch, stride, dropout))
                in_ch = ch
        self.layers = nn.Sequential(*layers)
        self.bn = nn.BatchNorm2d(in_ch)
        self.fc = nn.Linear(in_ch, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = F.relu(self.bn(self.layers(self.stem(x))))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


class ShakeDrop(nn.Module):
    """Per image: keep F(x) with prob keep, otherwise multiply it by a random factor,
    using a *different* random factor for the backward pass (Yamada et al. 2019).
    forward factor  = 1 if kept else alpha ~ U(-1, 1)
    backward factor = 1 if kept else beta  ~ U(0, 1)
    At test time this is the expected value: keep * F(x)."""
    def __init__(self, keep):
        super().__init__()
        self.keep = keep

    def forward(self, x):
        if not self.training:
            return x * self.keep
        shape = (x.size(0), 1, 1, 1)
        gate = x.new_empty(shape).bernoulli_(self.keep)
        alpha = x.new_empty(shape).uniform_(-1, 1)
        beta = x.new_empty(shape).uniform_(0, 1)
        fwd = gate + (1 - gate) * alpha
        bwd = gate + (1 - gate) * beta
        # value = fwd * x, gradient = bwd * dx (the detached part carries no gradient)
        return bwd * x + ((fwd - bwd) * x).detach()


class PyramidBlock(nn.Module):
    """F = BN-conv-BN-ReLU-conv-BN-ShakeDrop. S = (avg pool) + zero-pad the new channels,
    so PyramidNet shortcuts have no parameters."""
    def __init__(self, in_ch, out_ch, stride, keep):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.shake = ShakeDrop(keep)
        self.stride = stride

    def forward(self, x):
        out = self.conv1(self.bn1(x))
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.shake(self.bn3(out))
        s = F.avg_pool2d(x, 2) if self.stride == 2 else x
        return out + F.pad(s, (0, 0, 0, 0, 0, out.size(1) - s.size(1)))


class PyramidNet(nn.Module):
    """PyramidNet (Han et al. 2017) + ShakeDrop. Width grows by alpha in total, a little
    every block. ShakeDrop keep prob falls linearly from 1 to final_keep over the blocks."""
    def __init__(self, depth=110, alpha=270, final_keep=0.5, num_classes=10):
        super().__init__()
        n = (depth - 2) // 6
        total = 3 * n
        self.stem = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(16))
        layers = []
        in_ch = 16
        for i in range(total):
            out_ch = round(16 + alpha * (i + 1) / total)
            stride = 2 if (i > 0 and i % n == 0) else 1
            keep = 1 - (i + 1) / total * (1 - final_keep)
            layers.append(PyramidBlock(in_ch, out_ch, stride, keep))
            in_ch = out_ch
        self.layers = nn.Sequential(*layers)
        self.bn = nn.BatchNorm2d(in_ch)
        self.fc = nn.Linear(in_ch, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = F.relu(self.bn(self.layers(self.stem(x))))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


# a config is a plain dict: "type" picks the class, the rest are its arguments.
# checkpoints store the config, so build_model(config) can rebuild any model.
MODEL_TYPES = {"resnet": ResNet, "wrn": WideResNet, "pyramidnet": PyramidNet}

# students: must stay under the 5M budget
ARCHS = {
    "r3_64_3-5-3":   dict(type="resnet", channels=(64, 128, 256), blocks=(3, 5, 3)),
    "r3_80_2-3-3":   dict(type="resnet", channels=(80, 160, 256), blocks=(2, 3, 3)),
    "r3_96_2-2-2":   dict(type="resnet", channels=(96, 192, 320), blocks=(2, 2, 2)),
    "r4_64_3-3-2-1": dict(type="resnet", channels=(64, 128, 256, 320), blocks=(3, 3, 2, 1)),
    "r3_40_8-8-8":   dict(type="resnet", channels=(40, 80, 160), blocks=(8, 8, 8)),
}

# teachers: any size, train with --no-budget. The tiny ones are for smoke tests.
TEACHER_ARCHS = {
    "wrn_28_10":       dict(type="wrn", depth=28, widen=10),
    "wrn_16_1":        dict(type="wrn", depth=16, widen=1),
    "pyramid_110_270": dict(type="pyramidnet", depth=110, alpha=270),
    "pyramid_20_24":   dict(type="pyramidnet", depth=20, alpha=24),
    # not a teacher: ~2.24M, two of them fit the 5M budget together (ensemble experiment).
    # train.py still asserts <= 5M for it (no --no-budget needed)
    "wrn_40_2":        dict(type="wrn", depth=40, widen=2),
}


def build_model(config):
    config = dict(config)
    return MODEL_TYPES[config.pop("type")](**config)


def load_model(path, use_ema=True):
    """Rebuild a model from a checkpoint, on the CPU. Uses the EMA weights if asked and
    available. Returns (model, checkpoint)."""
    # our own files; they also hold numpy RNG state, which weights_only would reject
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    config = ckpt["config"] if "config" in ckpt else ARCHS[ckpt["arch"]]
    model = build_model(config)
    ema = ckpt.get("ema")
    model.load_state_dict(ema["model"] if use_ema and ema else ckpt["model"])
    return model, ckpt
