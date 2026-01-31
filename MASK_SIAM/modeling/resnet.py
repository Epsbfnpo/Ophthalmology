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


class TriD(nn.Module):
    """TriD.
    Reference:
      Chen et al. Treasure in Distribution: A Domain Randomization based Multi-Source Domain Generalization for 2D Medical Image Segmentation. MICCAI 2023.
    """
    def __init__(self, p=0.5, eps=1e-6, alpha=0.1):  ##### 
        """
        Args:
          p (float): probability of using TriD.
          eps (float): scaling parameter to avoid numerical issues.
          alpha (float): parameter of the Beta distribution.
        """
        super().__init__()
        self.p = p
        self.eps = eps
        self._activated = True  # Train: True, Test: False
        self.beta = torch.distributions.Beta(alpha, alpha)

    def set_activation_status(self, status=True):
        self._activated = status

    def forward(self, x):
        
        # if permute and self.training:

        if not self.training:
            return x

        if random.random() > self.p:
            return x

        N, C, H, W = x.shape

        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig    ########### 实例归一化

        # Sample mu and var from an uniform distribution, i.e., mu ～ U(0.0, 1.0), var ～ U(0.0, 1.0)
        mu_random = torch.empty((N, C, 1, 1), dtype=torch.float32).uniform_(0.0, 1.0).to(x.device)  ## 均值 
        var_random = torch.empty((N, C, 1, 1), dtype=torch.float32).uniform_(0.0, 1.0).to(x.device)  ## 方差

        lmda = self.beta.sample((N, C, 1, 1))
        bernoulli = torch.bernoulli(lmda).to(x.device)

        mu_mix = mu_random * bernoulli + mu * (1. - bernoulli)
        sig_mix = var_random * bernoulli + sig * (1. - bernoulli)
        return x_normed * sig_mix + mu_mix
    

class AdaptiveInstanceNorm2d(nn.Module):
    
    """EFDMix.

    Reference:
      Zhang et al. Exact Feature Distribution Matching for Arbitrary Style Transfer and Domain Generalization. CVPR 2022.
    """
    
    def __init__(self, p=0.01, eps=1e-5):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.p = p
        self.eps = eps

    def forward(self, x):
        # permute = random.random() < self.p
        # if permute and self.training:
        #     perm_indices = torch.randperm(x.size()[0])
        # else:
        #     return x
        # size = x.size()
        # N, C, H, W = size
        # if (H, W) == (1, 1):
        #     print('encountered bad dims')
        #     return x
        return instance_normalization(x)

    def extra_repr(self) -> str:
        return 'p={}'.format(
            self.p
        )

class PermuteAdaptiveInstanceNorm2d(nn.Module):
    
    """EFDMix.

    Reference:
      Zhang et al. Exact Feature Distribution Matching for Arbitrary Style Transfer and Domain Generalization. CVPR 2022.
    """
    
    def __init__(self, p=0.01, eps=1e-5):
        super(PermuteAdaptiveInstanceNorm2d, self).__init__()
        self.p = p
        self.eps = eps

    def forward(self, x):
        permute = random.random() < self.p
        if permute and self.training:
            perm_indices = torch.randperm(x.size()[0])
        else:
            return x
        size = x.size()
        N, C, H, W = size
        if (H, W) == (1, 1):
            print('encountered bad dims')
            return x
        return adaptive_instance_normalization(x, x[perm_indices], self.eps)

    def extra_repr(self) -> str:
        return 'p={}'.format(
            self.p
        )

def adaptive_instance_normalization(content_feat, style_feat, eps=1e-5):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat.detach(), eps)
    content_mean, content_std = calc_mean_std(content_feat, eps)
    content_std = content_std + eps  # to avoid division by 0
    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

        
def instance_normalization(feat, eps=1e-5):
    # assert (content_feat.size()[:2] == style_feat.size()[:2])
    content_feat = feat
    size = feat.size()
    # style_mean, style_std = calc_mean_std(style_feat.detach(), eps)
    content_mean, content_std = calc_mean_std(content_feat, eps)
    content_std = content_std + eps  # to avoid division by 0
    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat #* style_std.expand(size) + style_mean.expand(size)

def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert (len(size) == 4)
    N, C, H, W = size
    feat_std = torch.sqrt(feat.view(N, C, -1).var(dim=2).view(N, C, 1, 1) + eps)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std
    

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


class CorrelatedDistributionUncertainty(nn.Module):
    """
    ["Domain Generalization with Correlated Style Uncertainty"](https://arxiv.org/abs/2212.09950), WACV2024

        Args:
        p   (float): probabilty of foward distribution uncertainty module, p in [0,1].
        dim   (int): dimension of feature map channels

    """

    def __init__(self, p=0.5, eps=1e-6, alpha=0.3):
        super(CorrelatedDistributionUncertainty, self).__init__()
        self.eps = eps
        self.p = p
        self.alpha = alpha
        self.beta = torch.distributions.Beta(alpha, alpha)
    
    def __repr__(self):
        return f'CorrelatedDistributionUncertainty with p {self.p} and alpha {self.alpha}'

    def forward(self, x):
        if (not self.training) or (np.random.random()) > self.p:
            return x

        B, C = x.size(0), x.size(1)
        mu = torch.mean(x, dim=[2, 3], keepdim=True)
        sig = (x.var(dim=[2, 3], keepdim=True) + self.eps).sqrt()
        # mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig

        factor = self.beta.sample((B, 1, 1, 1)).to(x.device)

        mu_squeeze = torch.squeeze(mu)
        mean_mu = torch.mean(mu_squeeze, dim=0, keepdim=True)
        correlation_mu = (mu_squeeze-mean_mu).T @ (mu_squeeze-mean_mu) / B

        sig_squeeze = torch.squeeze(sig)
        mean_sig = torch.mean(sig_squeeze, dim=0, keepdim=True)
        correlation_sig = (sig_squeeze.T-mean_sig.T) @ (sig_squeeze-mean_sig) / B

        with torch.no_grad():
            try:
                _, mu_eng_vector = torch.linalg.eigh(C*correlation_mu+self.eps*torch.eye(C, device=x.device))
                # mu_corr_matrix = mu_eng_vector @ torch.sqrt(torch.diag(torch.clip(mu_eng_value, min=1e-10))) @ (mu_eng_vector.T)
            except:
                mu_eng_vector = torch.eye(C, device=x.device)
            
            if not torch.all(torch.isfinite(mu_eng_vector)) or torch.any(torch.isnan(mu_eng_vector)):
                mu_eng_vector = torch.eye(C, device=x.device)

            try:
                _, sig_eng_vector = torch.linalg.eigh(C*correlation_sig+self.eps*torch.eye(C, device=x.device))
                # sig_corr_matrix = sig_eng_vector @ torch.sqrt(torch.diag(torch.clip(sig_eng_value, min=1e-10))) @ (sig_eng_vector.T)
            except:
                sig_eng_vector = torch.eye(C, device=x.device)

            if not torch.all(torch.isfinite(sig_eng_vector )) or torch.any(torch.isnan(sig_eng_vector)):
                sig_eng_vector = torch.eye(C, device=x.device)

        mu_corr_matrix = mu_eng_vector @ torch.diag(torch.sqrt(torch.clip(torch.diag((mu_eng_vector.T)@ correlation_mu @ mu_eng_vector),min=1e-12))) @ (mu_eng_vector.T)
        sig_corr_matrix = sig_eng_vector @ torch.diag(torch.sqrt(torch.clip(torch.diag((sig_eng_vector.T)@ correlation_sig @ sig_eng_vector), min=1e-12))) @ (sig_eng_vector.T)

        gaussian_mu = (torch.randn(B, 1, C, device=x.device) @ mu_corr_matrix)
        gaussian_mu = torch.reshape(gaussian_mu, (B, C, 1, 1))

        gaussian_sig = (torch.randn(B, 1, C, device=x.device) @ sig_corr_matrix)
        gaussian_sig = torch.reshape(gaussian_sig, (B, C, 1, 1))

        mu_mix = mu + factor*gaussian_mu
        sig_mix = sig + factor*gaussian_sig

        return x_normed * sig_mix + mu_mix
    

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

        self.cfg=cfg
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

        self.mixstyle = None    ####  
        if ms_layers:
            p=self.cfg.P #0.2
            # if self.cfg.DG_MODE =='DG':
            #     p=0.15

            # self.mixstyle1 = PermuteAdaptiveInstanceNorm2d(p=0.2) 
            # self.mixstyle2 = PermuteAdaptiveInstanceNorm2d(p=0.2) 
            # self.mixstyle3 = PermuteAdaptiveInstanceNorm2d(p=0.2) 

            # self.mixstyle_coor1 = CorrelatedDistributionUncertainty(p=p) 
            # self.mixstyle_coor2 = CorrelatedDistributionUncertainty(p=p) 
            # self.mixstyle1 = CorrelatedDistributionUncertainty(p=p) 
            # self.mixstyle2 = CorrelatedDistributionUncertainty(p=p) 
            # self.mixstyle3 = CorrelatedDistributionUncertainty(p=p) 
            # self.mixstyle4 = CorrelatedDistributionUncertainty(p=p) 


            self.mixstyle_coor1 = DistributionUncertainty(p=p) 
            self.mixstyle_coor2 = DistributionUncertainty(p=p) 
            self.mixstyle1 = DistributionUncertainty(p=p) 
            self.mixstyle2 = DistributionUncertainty(p=p) 
            self.mixstyle3 = DistributionUncertainty(p=p) 
            self.mixstyle4 = DistributionUncertainty(p=p) 

            # self.mixstyle_coor1 = TriD(p=p) 
            # self.mixstyle_coor2 = TriD(p=p) 
            # self.mixstyle1 = TriD(p=p) 
            # self.mixstyle2 = TriD(p=p)  
            # self.mixstyle3 = TriD(p=p) 
            # self.mixstyle4 = TriD(p=p) 



            for layer_name in ms_layers:
                assert layer_name in ["layer1", "layer2", "layer3"]
            print(
                f"Insert {self.mixstyle1.__class__.__name__} after {ms_layers}"
            )
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

    def featuremaps(self, x, perturb=False, distill=False):
        x = self.conv1(x)
        # if self.mixstyle1.__class__.__name__ == 'CorrelatedDistributionUncertainty' or "DistributionUncertainty":
        #     x = self.mixstyle_coor1(x)
        self.x0=x
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # if self.mixstyle1.__class__.__name__ == 'CorrelatedDistributionUncertainty' or "DistributionUncertainty":
        #     x = self.mixstyle_coor2(x)
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
        if distill:
            return self.x2,self.x3,self.x4, self.layer4(x)
        else:
            return self.layer4(x)


    def forward(self, x, drop_rate=0.0, perturb=False, distill=False):
        if distill:
            
            f1, f2, f3, f = self.featuremaps(x, perturb=perturb, distill=distill)
            if self.mixstyle1.__class__.__name__ == 'CorrelatedDistributionUncertainty' or 'DistributionUncertainty':
                f = self.mixstyle4(f)
            # print(f.shape) # torch.Size([32, 2048, 7, 7])
            v = self.global_avgpool(f)
            if drop_rate>0.0:
                v = F.dropout(v, p=float(drop_rate))
            return f1, f2, f3, v.view(v.size(0), -1)
        else:
            f = self.featuremaps(x,perturb=perturb)
            if self.mixstyle1.__class__.__name__ == 'CorrelatedDistributionUncertainty' or 'DistributionUncertainty':
                f = self.mixstyle4(f)
            v = self.global_avgpool(f)
            if drop_rate>0.0:
                v = F.dropout(v, p=float(drop_rate))
            return v.view(v.size(0), -1)
        #return v


def init_pretrained_weights(model, model_url):
    pretrain_dict = model_zoo.load_url(model_url)
    # pretrain_dict = torch.load(model_url)
    model.load_state_dict(pretrain_dict, strict=False)


"""
Residual network configurations:
--
resnet18: block=BasicBlock, layers=[2, 2, 2, 2]
resnet34: block=BasicBlock, layers=[3, 4, 6, 3]
resnet50: block=Bottleneck, layers=[3, 4, 6, 3]
resnet101: block=Bottleneck, layers=[3, 4, 23, 3]
resnet152: block=Bottleneck, layers=[3, 8, 36, 3]
"""

def resnet18(pretrained=True, **kwargs):
    model = ResNet(block=BasicBlock, layers=[2, 2, 2, 2])

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet18"])

    return model


def resnet34(pretrained=True, **kwargs):
    model = ResNet(block=BasicBlock, layers=[3, 4, 6, 3])

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet34"])

    return model


def resnet50(cfg=None, pretrained=True, **kwargs):
    model = ResNet(cfg, block=Bottleneck,  layers=[3, 4, 6, 3])

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet50"])

    return model


def resnet101(pretrained=True, **kwargs):
    model = ResNet(block=Bottleneck, layers=[3, 4, 23, 3])

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet101"])

    return model


def resnet152(pretrained=True, **kwargs):
    model = ResNet(block=Bottleneck, layers=[3, 8, 36, 3])

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet152"])

    return model