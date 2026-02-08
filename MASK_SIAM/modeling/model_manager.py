
from .resnet import resnet18, resnet34, resnet50, resnet101
from .networks import ResNet_tea, ResNet_Distill
from .nets import *
import torch
from .LaDeDa import LaDeDa33, LaDeDa17, LaDeDa9, LaDeDa5

def get_net(cfg):
    if cfg.ALGORITHM == 'ERM' or cfg.ALGORITHM == 'GDRNet':
        net = get_backbone(cfg)
    elif cfg.ALGORITHM == 'GDRNet_MASK_SIAM':
        net = get_backbone(cfg)
    elif cfg.ALGORITHM == 'GDRNet_DUAL':
        net1, net2 = get_backbone_dual(cfg)
        return net1, net2
    elif cfg.ALGORITHM == 'GDRNet_DUAL_MASK':
        net1, net2 = get_backbone_dual(cfg)
        return net1, net2
    elif cfg.ALGORITHM == 'GDRNet_DUAL_MASK_SIAM':
        net1, net2 = get_backbone_dual(cfg)
        return net1, net2
    elif cfg.ALGORITHM == 'GDRNet_Mask':
        net = get_backbone(cfg)
    elif cfg.ALGORITHM == 'GREEN':
        net = SoftLabelGCN(cfg)
    elif cfg.ALGORITHM == 'CABNet':
        net = CABNet(cfg)
    elif cfg.ALGORITHM == 'MixupNet':
        net = MixupNet(cfg)
    elif cfg.ALGORITHM == 'MixStyleNet':
        net = MixStyleNet(cfg)
    elif cfg.ALGORITHM == 'Fishr' or cfg.ALGORITHM == 'DRGen':
        net = FishrNet(cfg)
    else:
        raise ValueError('Wrong type')
    return net

def get_backbone(cfg):
    if cfg.BACKBONE == 'resnet18':
        model = resnet18(pretrained=True)
    elif cfg.BACKBONE == 'resnet34':
        model = resnet34(pretrained=True)
    elif cfg.BACKBONE == 'resnet50':
        model = resnet50(cfg, pretrained=True)
    elif cfg.BACKBONE == 'resnet50_tea':  ### 
        model = ResNet_tea(cfg)
    elif cfg.BACKBONE == 'resnet50_distill':  ### 
        model = ResNet_Distill(cfg)
    elif cfg.BACKBONE == 'resnet101':
        model = resnet101(pretrained=True)
    elif cfg.BACKBONE == "LaDeDa33":
        model = LaDeDa33()
    else:
        raise ValueError('Wrong type')
    return model


def get_backbone_dual(cfg):
    # if cfg.BACKBONE == 'resnet18':
    #     model = resnet18(pretrained=True)
    # elif cfg.BACKBONE == 'resnet50':
    #     model = resnet50(pretrained=True)
    # elif cfg.BACKBONE == 'resnet101':
    #     model = resnet101(pretrained=True)
    # else:
    #     raise ValueError('Wrong type')
    
    if cfg.BACKBONE1 == 'resnet18':
        model1 = resnet18(pretrained=True)
    elif cfg.BACKBONE1 == 'resnet34':
        model1 = resnet34(pretrained=True)
    elif cfg.BACKBONE1 == 'resnet50':
        model1 = resnet50(pretrained=True)
    elif cfg.BACKBONE1 == 'resnet50_tea':  ### 
        model1 = ResNet_tea(cfg)
    elif cfg.BACKBONE1 == 'resnet101':
        model1 = resnet101(pretrained=True)


    if cfg.BACKBONE2 == 'resnet18':
        model2 = resnet18(pretrained=True)
    elif cfg.BACKBONE2 == 'resnet34':
        model2 = resnet34(pretrained=True)
    elif cfg.BACKBONE2 == 'resnet50':
        model2 = resnet50(pretrained=True)
    elif cfg.BACKBONE2 == 'resnet50_tea':  ### 
        model2 = ResNet_tea(cfg)
    elif cfg.BACKBONE2 == 'resnet101':
        model2 = resnet101(pretrained=True)
    else:
        raise ValueError('Wrong type')
    return model1, model2


def get_classifier(out_feature_size, cfg):
    return torch.nn.Linear(out_feature_size, cfg.DATASET.NUM_CLASSES)