from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models


class ResNetBackbone(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        return feat1, feat2, feat3


class UNetDecoder(nn.Module):
    def __init__(self, in_channels: int = 1024):
        super().__init__()
        self.up2 = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )
        self.up1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )
        self.up0 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )
        self.out_conv = nn.Conv2d(128, 1, kernel_size=1)

    def forward(self, feat3: torch.Tensor) -> torch.Tensor:
        x = self.up2(feat3)
        x = self.up1(x)
        x = self.up0(x)
        return self.out_conv(x)


class ConceptProjector(nn.Module):
    def __init__(self, in_dim: int, concept_dim: int):
        super().__init__()
        self.projector_lv1 = nn.Sequential(
            nn.Linear(in_dim, concept_dim),
            nn.ReLU(inplace=True),
        )
        self.projector_lv2 = nn.Sequential(
            nn.Linear(concept_dim, concept_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projector_lv1(x)
        return self.projector_lv2(x)


class HCGDRNet(nn.Module):
    def __init__(
        self,
        concept_dim: int = 512,
        num_classes: int = 5,
        concept_bank: Optional[torch.Tensor] = None,
        pretrained_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = ResNetBackbone(pretrained=pretrained_backbone)
        self.decoder = UNetDecoder(in_channels=1024)
        self.projector = ConceptProjector(in_dim=1024, concept_dim=concept_dim)
        self.classifier = nn.Linear(concept_dim, num_classes)
        self.register_buffer("concept_bank", concept_bank)

    def forward(self, x: torch.Tensor):
        feat1, feat2, feat3 = self.backbone(x)
        pred_mask = self.decoder(feat3)

        clean_mask = pred_mask.detach()
        clean_mask = torch.sigmoid(clean_mask)
        clean_mask = F.interpolate(clean_mask, size=feat3.shape[-2:], mode="bilinear", align_corners=False)
        gated_feats = feat3 * clean_mask

        pooled = F.adaptive_avg_pool2d(gated_feats, output_size=1).flatten(1)
        concept_vec = self.projector(pooled)
        logits = self.classifier(concept_vec)
        return pred_mask, concept_vec, logits

    def set_concept_bank(self, concept_bank: torch.Tensor) -> None:
        self.concept_bank = concept_bank
