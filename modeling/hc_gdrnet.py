from typing import Optional, Tuple
import os

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models


class ResNetBackbone(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        resnet = models.resnet50(weights=None)

        if pretrained:
            weight_path = "resnet50.pth"
            if os.path.exists(weight_path):
                print(f"[Backbone] Loading local weights from {weight_path}")
                # [FIX]: 加上 weights_only=True 消除警告
                state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
                resnet.load_state_dict(state_dict)
            else:
                # 如果没有本地权重，尝试联网下载 (登录节点使用)
                print("[Backbone] Local weights not found, trying to download...")
                resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return c4


class UNetDecoder(nn.Module):
    def __init__(self, in_channels: int = 2048):
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_channels, 512, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, x: torch.Tensor, target_size: Tuple[int, int]):
        out = self.up(x)
        if out.shape[-2:] != target_size:
            out = F.interpolate(out, size=target_size, mode="bilinear", align_corners=False)
        return out


class HierarchicalProjector(nn.Module):
    def __init__(self, in_dim: int, num_l1: int, num_l2: int):
        super().__init__()
        self.proj_l1 = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_l1)
        )
        self.proj_l2 = nn.Sequential(
            nn.Linear(in_dim + num_l1, 256),
            nn.ReLU(),
            nn.Linear(256, num_l2)
        )

    def forward(self, x: torch.Tensor):
        # [ROBUST FIX]: 自动检测输入是否为 4D [B, C, H, W]，如果是则压扁
        if x.dim() == 4:
            x = x.mean(dim=(2, 3))

        z_l1 = self.proj_l1(x)
        # L2 input depends on L1 output (Conditioning)
        x_l2 = torch.cat([x, F.relu(z_l1)], dim=1)
        z_l2 = self.proj_l2(x_l2)
        return z_l1, z_l2


class HCGDRNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        num_l1_concepts: int = 2,
        num_l2_concepts: int = 4,
        pretrained_backbone: bool = True,
        concept_bank: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.backbone = ResNetBackbone(pretrained=pretrained_backbone)
        self.decoder = UNetDecoder(in_channels=2048)
        self.projector = HierarchicalProjector(in_dim=2048, num_l1=num_l1_concepts, num_l2=num_l2_concepts)
        self.feat_adapter = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )
        self.classifier = nn.Linear(num_l2_concepts, num_classes)
        self.register_buffer("concept_bank", concept_bank)

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)
        pred_mask = self.decoder(feat, target_size=x.shape[-2:])

        clean_mask = pred_mask.detach()
        clean_mask = torch.sigmoid(clean_mask)

        mask_small = F.interpolate(clean_mask, size=feat.shape[-2:], mode="nearest")
        gated_feat = feat * mask_small

        # 由于 Projector 现在自带鲁棒性，这里不改也没事，但 Projector 会自动处理 GAP
        z_l1, z_l2 = self.projector(gated_feat)

        return z_l1, z_l2
