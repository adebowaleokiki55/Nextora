

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    

    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class DropPath(nn.Module):
    

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, device=x.device, dtype=x.dtype)
        mask = torch.floor_(mask + keep_prob)
        return x * mask / keep_prob


class ResBlock(nn.Module):
    

    def __init__(self, in_ch, out_ch, stride=1, se_reduction=16, drop_path_rate=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SEBlock(out_ch, reduction=se_reduction)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.drop_path(out)
        out = self.relu(out + identity)
        return out


class InvertedResBlock(nn.Module):
    

    def __init__(self, in_ch, out_ch, stride=1, expansion=4,
                 se_reduction=16, drop_path_rate=0.0):
        super().__init__()
        mid_ch = in_ch * expansion

        self.expand = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, 3, stride=stride, padding=1,
                      groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.se = SEBlock(out_ch, reduction=se_reduction)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.expand(x)
        out = self.depthwise(out)
        out = self.project(out)
        out = self.se(out)
        out = self.drop_path(out)
        out = self.relu(out + identity)
        return out


class MultiScaleStem(nn.Module):

    def __init__(self, in_ch=3, out_ch=48):
        super().__init__()
        branch_ch = out_ch // 4  # 12 channels per branch

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_ch, branch_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.ReLU(inplace=True),
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_ch, branch_ch, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.ReLU(inplace=True),
        )
        self.branch7 = nn.Sequential(
            nn.Conv2d(in_ch, branch_ch, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(branch_ch * 3, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        b7 = self.branch7(x)
        return self.fuse(torch.cat([b3, b5, b7], dim=1))


class PlasmoSENet(nn.Module):


    def __init__(self, num_classes=2, drop_path_rate=0.1, dropout=0.2, expansion=4):
        super().__init__()
        channels = [48, 96, 192, 256]
        blocks_per_stage = [2, 2, 3, 2]
        self.last_channel = channels[-1]

        # Linearly increasing drop path rates across 9 residual blocks
        num_blocks = sum(blocks_per_stage)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_blocks)]

        features = []
        idx = 0

        # Multi-scale stem (3 -> 48, spatial /2)
        features.append(MultiScaleStem(in_ch=3, out_ch=channels[0]))

        # Stage 1 — 48 channels, 112x112, standard ResBlocks
        for i in range(blocks_per_stage[0]):
            features.append(ResBlock(
                channels[0], channels[0], stride=1, drop_path_rate=dpr[idx],
            ))
            idx += 1

        # Stage 2 — 96 channels, 56x56, standard ResBlocks
        for i in range(blocks_per_stage[1]):
            in_ch = channels[0] if i == 0 else channels[1]
            stride = 2 if i == 0 else 1
            features.append(ResBlock(
                in_ch, channels[1], stride=stride, drop_path_rate=dpr[idx],
            ))
            idx += 1

        # Stage 3 — 192 channels, 28x28, InvertedResBlocks
        for i in range(blocks_per_stage[2]):
            in_ch = channels[1] if i == 0 else channels[2]
            stride = 2 if i == 0 else 1
            features.append(InvertedResBlock(
                in_ch, channels[2], stride=stride, expansion=expansion,
                drop_path_rate=dpr[idx],
            ))
            idx += 1

        # Stage 4 — 256 channels, 14x14, InvertedResBlocks
        for i in range(blocks_per_stage[3]):
            in_ch = channels[2] if i == 0 else channels[3]
            stride = 2 if i == 0 else 1
            features.append(InvertedResBlock(
                in_ch, channels[3], stride=stride, expansion=expansion,
                drop_path_rate=dpr[idx],
            ))
            idx += 1

        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(self.last_channel, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        """Kaiming initialization for training from scratch."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
