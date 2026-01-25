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
                state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
                resnet.load_state_dict(state_dict)
            else:
                print("[Backbone] Local weights not found, downloading...")
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
    def __init__(self, in_channels: int = 2048, num_classes: int = 6):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(in_channels, 512, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True)
        )
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True)
        )
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        )
        self.final_conv = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(self, x, target_size):
        x = self.up1(x)
        x = self.conv1(x)
        x = self.up2(x)
        x = self.conv2(x)
        x = self.up3(x)
        x = self.conv3(x)
        x = self.final_conv(x)
        if x.shape[-2:] != target_size:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return x


class HierarchicalProjector(nn.Module):
    def __init__(self, in_dim: int, num_l1: int, num_l2: int):
        super().__init__()
        self.proj_l1 = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_l1)
        )
        self.proj_l2 = nn.Sequential(
            nn.Linear(in_dim + num_l1, 512),
            nn.ReLU(),
            nn.Linear(512, num_l2)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_l1 = self.proj_l1(x)
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
            seg_classes: int = 6
    ) -> None:
        super().__init__()
        self.backbone = ResNetBackbone(pretrained=pretrained_backbone)
        self.decoder = UNetDecoder(in_channels=2048, num_classes=seg_classes)
        self.projector = HierarchicalProjector(in_dim=2048, num_l1=num_l1_concepts, num_l2=num_l2_concepts)

        self.feat_adapter = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(num_l2_concepts, num_classes)
        )

        self.register_buffer("concept_bank", concept_bank if concept_bank is not None else torch.randn(6, 768))

    def forward(self, x):
        # 1. Feature Extraction
        feat = self.backbone(x)

        # 2. Hard Gating
        with torch.no_grad():
            x_red = x[:, 0:1, :, :]
            # Relaxed threshold to -1.8 to prevent deleting retinal information after CLAHE
            hard_mask = (x_red > -1.8).float()
            mask_small = F.interpolate(hard_mask, size=feat.shape[-2:], mode="nearest")

        feat_masked = feat * mask_small
        pred_mask = self.decoder(feat_masked, target_size=x.shape[-2:])

        # 6. Soft Attention
        probs = torch.softmax(pred_mask, dim=1)
        lesion_prob = probs[:, 2:].sum(dim=1, keepdim=True)
        lesion_attn = F.interpolate(lesion_prob, size=feat.shape[-2:], mode="bilinear", align_corners=False)

        feat_boosted = feat_masked + (feat_masked * lesion_attn)

        sum_feat = feat_boosted.sum(dim=(2, 3))
        sum_area = mask_small.sum(dim=(2, 3)) + 1e-6
        pooled_feat = sum_feat / sum_area

        z_l1, z_l2 = self.projector(pooled_feat)
        vis_emb = self.feat_adapter(pooled_feat)
        pred_cls = self.classifier(z_l2)

        return pred_cls, pred_mask, vis_emb, z_l1, z_l2