import torch
import torch.nn as nn
import torch.nn.functional as F


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class BasicBlock(nn.Module):
    """out = relu( F(x) + S(x) )"""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        # F(x): conv -> BN -> ReLU -> conv -> BN
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        nn.init.zeros_(self.bn2.weight)   # F(x) starts at 0 -> block starts as identity

        # S(x): identity, or 1x1 conv when the shape changes
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, channels=(64, 128, 256), blocks=(3, 5, 3), num_classes=10):
        super().__init__()
        # stem: 3 colors -> channels[0] features, keep 32x32 (no maxpool)
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(),
        )
        # stages: first block of every stage after the first halves the image (stride 2)
        layers = []
        in_ch = channels[0]
        for stage, (ch, n) in enumerate(zip(channels, blocks)):
            for b in range(n):
                stride = 2 if (b == 0 and stage > 0) else 1
                layers.append(BasicBlock(in_ch, ch, stride))
                in_ch = ch
        self.layers = nn.Sequential(*layers)
        # head: global average pool -> linear
        self.fc = nn.Linear(in_ch, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = self.layers(self.stem(x))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)

ARCHS = {
    "r3_64_3-5-3":   dict(channels=(64, 128, 256), blocks=(3, 5, 3)),
    "r3_80_2-3-3":   dict(channels=(80, 160, 256), blocks=(2, 3, 3)),
    "r3_96_2-2-2":   dict(channels=(96, 192, 320), blocks=(2, 2, 2)),
    "r4_64_3-3-2-1": dict(channels=(64, 128, 256, 320), blocks=(3, 3, 2, 1)),
    "r3_40_8-8-8":   dict(channels=(40, 80, 160), blocks=(8, 8, 8)),
}