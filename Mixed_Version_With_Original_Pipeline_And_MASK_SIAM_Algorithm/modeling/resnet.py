import os
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import random
import numpy as np

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass

    def out_features(self):
        """Output feature dimension."""
        if self.__dict__.get("_out_features") is None:
            return None
        return self._out_features


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


# [New from MaskSiam] 核心风格增强模块
class DistributionUncertainty(nn.Module):
    """
    Distribution Uncertainty Module
        Args:
        p   (float): probabilty of foward distribution uncertainty module, p in [0,1].
    """

    def __init__(self, p=0.5, eps=1e-6):
        super(DistributionUncertainty, self).__init__()
        self.eps = eps
        self.p = p
        self.factor = 1.0

    def _reparameterize(self, mu, std):
        epsilon = torch.randn_like(std) * self.factor
        return mu + epsilon * std

    def sqrtvar(self, x):
        t = (x.var(dim=0, keepdim=True) + self.eps).sqrt()
        t = t.repeat(x.shape[0], 1)
        return t

    def forward(self, x):
        if (not self.training) or (np.random.random()) > self.p:
            return x

        mean = x.mean(dim=[2, 3], keepdim=False)
        std = (x.var(dim=[2, 3], keepdim=False) + self.eps).sqrt()

        sqrtvar_mu = self.sqrtvar(mean)
        sqrtvar_std = self.sqrtvar(std)

        beta = self._reparameterize(mean, sqrtvar_mu)
        gamma = self._reparameterize(std, sqrtvar_std)

        x = (x - mean.reshape(x.shape[0], x.shape[1], 1, 1)) / std.reshape(x.shape[0], x.shape[1], 1, 1)
        x = x * gamma.reshape(x.shape[0], x.shape[1], 1, 1) + beta.reshape(x.shape[0], x.shape[1], 1, 1)
        return x


class ResNet(Backbone):
    def __init__(
            self,
            cfg,
            block,
            layers,
            ms_class=None,
            ms_layers=["layer1", "layer2", "layer3"],
            ms_p=0.5,
            ms_a=0.1,
            **kwargs
    ):
        self.inplanes = 64
        super().__init__()

        self.cfg = cfg
        # backbone network
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        self._out_features = 512 * block.expansion

        self.mixstyle = None
        if ms_layers:
            # [Robustness Fix] 检查 cfg 是否存在以及是否有 P 属性
            if self.cfg is not None:
                p = getattr(self.cfg, 'P', 0.5)
            else:
                p = 0.5  # Default

            self.mixstyle_coor1 = DistributionUncertainty(p=p)
            self.mixstyle_coor2 = DistributionUncertainty(p=p)
            self.mixstyle1 = DistributionUncertainty(p=p)
            self.mixstyle2 = DistributionUncertainty(p=p)
            self.mixstyle3 = DistributionUncertainty(p=p)
            self.mixstyle4 = DistributionUncertainty(p=p)

            for layer_name in ms_layers:
                assert layer_name in ["layer1", "layer2", "layer3"]

        self.ms_layers = ms_layers
        self._init_params()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # [Crucial for MASK_SIAM] 更新 featuremaps 支持 perturb 和 distill
    def featuremaps(self, x, perturb=False, distill=False):
        x = self.conv1(x)
        self.x0 = x
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        self.x1 = x

        x = self.layer1(x)
        self.x2 = x
        if perturb:
            if "layer1" in self.ms_layers:
                x = self.mixstyle1(x)

        x = self.layer2(x)
        self.x3 = x
        if perturb:
            if "layer2" in self.ms_layers:
                x = self.mixstyle2(x)

        x = self.layer3(x)
        self.x4 = x
        if perturb:
            if "layer3" in self.ms_layers:
                x = self.mixstyle3(x)

        # 如果需要蒸馏，返回中间层特征
        if distill:
            return self.x2, self.x3, self.x4, self.layer4(x)
        else:
            return self.layer4(x)

    # [Crucial for MASK_SIAM] 更新 forward 支持 distill 和 perturb
    def forward(self, x, drop_rate=0.0, perturb=False, distill=False):
        if distill:
            f1, f2, f3, f = self.featuremaps(x, perturb=perturb, distill=distill)
            # 在 layer4 后应用 mixstyle4
            if hasattr(self, 'mixstyle4'):
                f = self.mixstyle4(f)

            v = self.global_avgpool(f)
            if drop_rate > 0.0:
                v = F.dropout(v, p=float(drop_rate))
            return f1, f2, f3, v.view(v.size(0), -1)
        else:
            f = self.featuremaps(x, perturb=perturb)
            if hasattr(self, 'mixstyle4'):
                f = self.mixstyle4(f)
            v = self.global_avgpool(f)
            if drop_rate > 0.0:
                v = F.dropout(v, p=float(drop_rate))
            return v.view(v.size(0), -1)


# [Original] 保留 Original 的预训练加载逻辑，支持本地文件
def init_pretrained_weights(model, model_url):
    filename = model_url.split('/')[-1]
    local_path = os.path.join('pretrained', filename)

    if os.path.exists(local_path):
        print(f"Loading pretrained model from local path: {local_path}")
        pretrain_dict = torch.load(local_path, map_location='cpu')
    else:
        print(f"Local file not found at {local_path}, trying to download from {model_url}")
        pretrain_dict = model_zoo.load_url(model_url)

    model.load_state_dict(pretrain_dict, strict=False)


# 接口函数：更新为接受 cfg 参数
def resnet18(cfg=None, pretrained=True, **kwargs):
    model = ResNet(cfg, block=BasicBlock, layers=[2, 2, 2, 2])
    if pretrained:
        init_pretrained_weights(model, model_urls["resnet18"])
    return model


def resnet34(cfg=None, pretrained=True, **kwargs):
    model = ResNet(cfg, block=BasicBlock, layers=[3, 4, 6, 3])
    if pretrained:
        init_pretrained_weights(model, model_urls["resnet34"])
    return model


def resnet50(cfg=None, pretrained=True, **kwargs):
    model = ResNet(cfg, block=Bottleneck, layers=[3, 4, 6, 3])
    if pretrained:
        init_pretrained_weights(model, model_urls["resnet50"])
    return model


def resnet101(cfg=None, pretrained=True, **kwargs):
    model = ResNet(cfg, block=Bottleneck, layers=[3, 4, 23, 3])
    if pretrained:
        init_pretrained_weights(model, model_urls["resnet101"])
    return model


def resnet152(cfg=None, pretrained=True, **kwargs):
    model = ResNet(cfg, block=Bottleneck, layers=[3, 8, 36, 3])
    if pretrained:
        init_pretrained_weights(model, model_urls["resnet152"])
    return model