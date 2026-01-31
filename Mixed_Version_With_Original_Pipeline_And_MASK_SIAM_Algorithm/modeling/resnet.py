import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch
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


class DistributionUncertainty(nn.Module):
    """
    Distribution Uncertainty Module (DSU)
    用于在特征统计量中引入随机性，模拟潜在的域偏移。
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
            ms_layers=["layer1", "layer2", "layer3"],
            **kwargs
    ):
        self.inplanes = 64
        super().__init__()

        # 兼容性处理：如果 cfg 为 None，提供默认值防止报错
        self.cfg = cfg

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        self._out_features = 512 * block.expansion

        # MixStyle / DSU 配置
        self.ms_layers = ms_layers
        if self.cfg is not None:
            p = getattr(self.cfg, "P", 0.5)  # 默认概率
        else:
            p = 0.5

        # 初始化扰动模块 (DSU)
        self.mixstyle1 = DistributionUncertainty(p=p)
        self.mixstyle2 = DistributionUncertainty(p=p)
        self.mixstyle3 = DistributionUncertainty(p=p)
        self.mixstyle4 = DistributionUncertainty(p=p)

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
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
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

    def featuremaps(self, x, perturb=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Layer 1
        x = self.layer1(x)
        self.x2 = x  # 保存特征
        if perturb and "layer1" in self.ms_layers:
            x = self.mixstyle1(x)

        # Layer 2
        x = self.layer2(x)
        self.x3 = x  # 保存特征
        if perturb and "layer2" in self.ms_layers:
            x = self.mixstyle2(x)

        # Layer 3
        x = self.layer3(x)
        self.x4 = x  # 保存特征
        if perturb and "layer3" in self.ms_layers:
            x = self.mixstyle3(x)

        # Layer 4
        x = self.layer4(x)
        return x

    def forward(self, x, drop_rate=0.0, perturb=False, distill=False):
        """
        修改后的 Forward 函数，支持扰动(perturb)和蒸馏(distill)
        """
        f = self.featuremaps(x, perturb=perturb)

        # 可选：在 layer4 之后再加一层扰动
        # if perturb: f = self.mixstyle4(f)

        v = self.global_avgpool(f)
        v = v.view(v.size(0), -1)

        if drop_rate > 0.0:
            v = F.dropout(v, p=float(drop_rate), training=self.training)

        if distill:
            # 如果需要蒸馏，返回中间层特征 (x2, x3, x4) 和最终特征 v
            # 注意：这里的 self.x2 等是在 featuremaps 中被赋值的
            return self.x2, self.x3, self.x4, v
        else:
            return v


def init_pretrained_weights(model, model_url):
    pretrain_dict = model_zoo.load_url(model_url)
    model.load_state_dict(pretrain_dict, strict=False)


def resnet18(pretrained=True, **kwargs):
    model = ResNet(cfg=None, block=BasicBlock, layers=[2, 2, 2, 2])
    if pretrained: init_pretrained_weights(model, model_urls["resnet18"])
    return model


def resnet34(pretrained=True, **kwargs):
    model = ResNet(cfg=None, block=BasicBlock, layers=[3, 4, 6, 3])
    if pretrained: init_pretrained_weights(model, model_urls["resnet34"])
    return model


def resnet50(cfg=None, pretrained=True, **kwargs):
    # 注意：这里接收 cfg 参数
    model = ResNet(cfg, block=Bottleneck, layers=[3, 4, 6, 3])
    if pretrained: init_pretrained_weights(model, model_urls["resnet50"])
    return model


def resnet101(pretrained=True, **kwargs):
    model = ResNet(cfg=None, block=Bottleneck, layers=[3, 4, 23, 3])
    if pretrained: init_pretrained_weights(model, model_urls["resnet101"])
    return model