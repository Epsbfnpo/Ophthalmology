import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, collections, logging
from collections import OrderedDict
from copy import deepcopy
from itertools import combinations  # 新增引用

import utils.misc as misc
from utils.validate import algorithm_validate
import modeling.model_manager as models
# 确保引用了正确的 Loss
from modeling.losses import DahLoss, DahLoss_Siam, DahLoss_Siam_Fastmoco
from modeling.nets import LossValley, AveragedModel
from dataset.data_manager import get_post_FundusAug
from guided_filter_pytorch.ModifiedHFCFilter import FourierButterworthHFCFilter



from masking import Masking
from backpack import backpack, extend
from backpack.extensions import BatchGrad
from torch.cuda.amp import GradScaler, autocast  # 新增引用

ALGORITHMS = [
    'ERM', 'GDRNet', 'GDRNet_MASK_SIAM', 'GREEN', 'CABNet',
    'MixupNet', 'MixStyleNet', 'Fishr', 'DRGen'
]

def hfc_mul_mask(hfc_filter, image, mask, do_norm=False):
    hfc = hfc_filter((image / 2 + 0.5), mask)
    if do_norm:
        hfc = 2 * hfc - 1
    return (hfc + 1) * mask - 1


def get_algorithm_class(algorithm_name):
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


def init_weights_MLPHead(module, init_method='He'):
    for _, m in module.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if init_method == 'He':
                nn.init.kaiming_normal_(m.weight.data)
            elif init_method == 'Xavier':
                nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, val=0)


# JS Divergence 工具函数 (移动到顶层或作为类方法均可，这里保留在顶层)
def JS_Divergence(output_clean, output_aug1, output_aug2, output_aug3):
    p_clean, p_aug1 = F.softmax(output_clean, dim=1), F.softmax(output_aug1, dim=1)
    p_aug2, p_aug3 = F.softmax(output_aug2, dim=1), F.softmax(output_aug3, dim=1)
    # 计算混合分布
    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2 + p_aug3) / 4., 1e-7, 1).log()
    # 计算 KL 散度均值
    loss_ctr = (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                F.kl_div(p_mixture, p_aug2, reduction='batchmean') +
                F.kl_div(p_mixture, p_aug3, reduction='batchmean')) / 4.
    return loss_ctr


class Algorithm(torch.nn.Module):
    def __init__(self, num_classes, cfg):
        super(Algorithm, self).__init__()
        self.cfg = cfg
        self.epoch = 0

    def update(self, minibatches): raise NotImplementedError

    def update_epoch(self, epoch): self.epoch = epoch; return epoch

    def validate(self, val_loader, test_loader, writer): raise NotImplementedError

    def save_model(self, log_path): raise NotImplementedError

    def renew_model(self, log_path): raise NotImplementedError

    def predict(self, x): raise NotImplementedError


# ... (ERM, GDRNet, GREEN, CABNet, MixupNet, MixStyleNet, Fishr, DRGen 保持不变，此处省略以节省篇幅) ...
# 请保留你原文件中这些类的代码，或者从之前的版本中复制过来。
# 重点在于下面的 GDRNet_MASK_SIAM

class GDRNet_MASK_SIAM(Algorithm):
    def __init__(self, num_classes, cfg):
        super(GDRNet_MASK_SIAM, self).__init__(num_classes, cfg)

        self.cfg = cfg
        self.network = models.get_net(cfg)

        # 1. 恢复多头分类器 (Classifier 1, 2, 3)
        # 注意：这里假设 network 返回特征的维度可以通过 out_features() 获取
        # 如果 network 结构特殊，需确保 models.get_classifier 能正确处理
        feature_dim = self.network.out_features()
        self.classifier = models.get_classifier(feature_dim, cfg)
        self.classifier1 = models.get_classifier(feature_dim, cfg)
        self.classifier2 = models.get_classifier(feature_dim, cfg)
        self.classifier3 = models.get_classifier(feature_dim, cfg)

        self.model = nn.Sequential(self.network, self.classifier)

        # 2. 定义 Projector 和 Predictor
        if cfg.BACKBONE in ['resnet34', 'resnet18']:
            dim_in1 = 512
            feat_dim1 = 512
        else:
            dim_in1 = 2048  # Resnet 50/101
            feat_dim1 = 512

        self.projector = nn.Sequential(
            nn.Linear(dim_in1, dim_in1, bias=False),
            nn.BatchNorm1d(dim_in1),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in1, feat_dim1, bias=False),
            nn.BatchNorm1d(feat_dim1),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim1, dim_in1, bias=False),
            nn.BatchNorm1d(dim_in1, affine=False)
        )

        self.predictor = nn.Sequential(
            nn.Linear(dim_in1, feat_dim1, bias=False),
            nn.BatchNorm1d(feat_dim1),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim1, dim_in1)
        )

        init_weights_MLPHead(self.projector, init_method='He')
        init_weights_MLPHead(self.predictor, init_method='He')

        # 3. 优化器配置 (恢复 AdamW 和 fix_lr 分组)
        self.optimizer = torch.optim.AdamW(
            [
                {"params": self.network.parameters(), 'fix_lr': False},
                {"params": self.classifier.parameters(), 'fix_lr': False},
                {"params": self.classifier1.parameters(), 'fix_lr': True},
                {"params": self.classifier2.parameters(), 'fix_lr': True},
                {"params": self.classifier3.parameters(), 'fix_lr': True},
                {"params": self.projector.parameters(), 'fix_lr': False},
                {"params": self.predictor.parameters(), 'fix_lr': True},
            ],
            lr=cfg.LEARNING_RATE,
            weight_decay=cfg.WEIGHT_DECAY
        )

        # 4. 恢复 Memory Queue (关键组件)
        K = 1024
        dim = 2048
        self.K = K
        self.num_positive = 0
        self.register_buffer("queue", torch.randn(K, dim))
        self.queue = nn.functional.normalize(self.queue, dim=-1)
        self.register_buffer("queue_labels", -torch.ones(K, dtype=torch.long))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.fundusAug = get_post_FundusAug(cfg)
        # 恢复 HFC 滤波器初始化
        self.hfc_filter_in = FourierButterworthHFCFilter(butterworth_d0_ratio=0.001, butterworth_n=4, do_median_padding=False, image_size=(256, 256)).cuda()

        # 5. FastMoCo 配置
        # 如果 cfg 中没有定义 FASTMOCO，默认为 0.5 (开启状态)
        fastmoco_val = getattr(cfg, 'FASTMOCO', 0.5)
        if fastmoco_val > 0:
            self.split_num = 2
            self.combs = 3
            self.criterion = DahLoss_Siam_Fastmoco(
                cfg=self.cfg,
                beta=cfg.GDRNET.BETA,
                max_iteration=cfg.EPOCHS,
                training_domains=cfg.DATASET.SOURCE_DOMAINS,
                temperature=cfg.GDRNET.TEMPERATURE,
                scaling_factor=cfg.GDRNET.SCALING_FACTOR,
                fastmoco=fastmoco_val
            )
        else:
            self.criterion = DahLoss_Siam(
                beta=cfg.GDRNET.BETA,
                max_iteration=cfg.EPOCHS,
                training_domains=cfg.DATASET.SOURCE_DOMAINS,
                temperature=cfg.GDRNET.TEMPERATURE,
                scaling_factor=cfg.GDRNET.SCALING_FACTOR
            )

        # Masking 工具 (虽然 algorithms_new.py 手写了 patchify，但这里保留 masking 类作为辅助，逻辑需保持一致)
        # 为兼容性，我们将在 update 中使用手动 patchify 方法，正如 student 建议的那样

        self.scaler = GradScaler()

    # --- 辅助函数：Patchify & Unpatchify & Random Masking ---
    def patchify(self, imgs, block=32):
        p = block
        # assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        h = w = imgs.shape[2] // p
        n = imgs.shape[0]
        x = imgs.reshape(shape=(n, 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x, block=32):
        p = block
        h = w = int(x.shape[1] ** .5)
        n = x.shape[0]
        x = x.reshape(shape=(n, h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(n, 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand([N, L], device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        x_masked = x * mask.unsqueeze(-1)
        return x_masked

    # --- 队列操作 ---
    @torch.no_grad()
    def dequeue_and_enqueue(self, features, labels):
        batch_size = features.shape[0]
        ptr = int(self.queue_ptr)
        # assert self.K % batch_size == 0
        # 简单处理：如果 batch_size 不整除，直接覆盖
        if ptr + batch_size > self.K:
            batch_size = self.K - ptr  # 截断处理边界情况

        self.queue[ptr: ptr + batch_size, :] = features[:batch_size]
        self.queue_labels[ptr: ptr + batch_size] = labels[:batch_size]
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def sample_target(self, features, labels):
        neighbor = []
        for i, label in enumerate(labels):
            pos = torch.where(self.queue_labels == label)[0]
            if len(pos) != 0:
                if self.num_positive > 0:
                    choice = torch.multinomial(torch.ones_like(pos).type(torch.FloatTensor), self.num_positive,
                                               replacement=True)
                    idx = pos[choice]
                    neighbor.append(self.queue[idx].mean(0))
                else:
                    neighbor.append(self.queue[pos].mean(0))
            else:
                neighbor.append(features[i])
        neighbor = torch.stack(neighbor, dim=0)
        return neighbor

    def _local_split(self, x):
        # NxCxHxW --> 4NxCx(H/2)x(W/2)
        _side_indent = x.size(2) // self.split_num, x.size(3) // self.split_num
        cols = x.split(_side_indent[1], dim=3)
        xs = []
        for _x in cols:
            xs += _x.split(_side_indent[0], dim=2)
        x = torch.cat(xs, dim=0)
        return x

    def img_process(self, img_tensor, mask_tensor, fundusAug):
        img_tensor_new, mask_tensor_new = fundusAug['post_aug1'](img_tensor.clone(), mask_tensor.clone())
        img_tensor_new = img_tensor_new * mask_tensor_new
        img_tensor_new = fundusAug['post_aug2'](img_tensor_new)
        img_tensor_ori = fundusAug['post_aug2'](img_tensor)
        return img_tensor_new, img_tensor_ori

    def img_process_freq(self, img_tensor, mask_tensor, fundusAug):
        # 使用 HFC 滤波器处理图像，并进行增强
        img_tensor_freq = hfc_mul_mask(self.hfc_filter_in, img_tensor, mask_tensor, do_norm=True)
        img_tensor_freq = fundusAug['post_aug2'](img_tensor_freq)
        return img_tensor_freq

    # --- 核心 Update 逻辑 ---
    def update(self, minibatch):
        # 1. 智能解包逻辑：兼容频域增强(6元组)与普通模式(4元组)
        image_freq = None
        if hasattr(self.cfg, 'TRANSFORM') and self.cfg.TRANSFORM.FREQ:
            # 频域模式：MASK_SIAM 原版 DataLoader 返回 6 个元素
            if isinstance(minibatch, (list, tuple)) and len(minibatch) == 6:
                image, image_freq, image_freq2, mask, label, domain = minibatch
            else:
                # 异常回退或兼容旧版 DataLoader
                image, mask, label, domain = minibatch

            # 为 FastMoCo 伪造双流数据 (Combine 仓库特有逻辑)
            image_a, mask_a, label_a, domain_a = image.clone(), mask.clone(), label.clone(), domain.clone()
            image_freq_a = image_freq.clone() if image_freq is not None else None

        # 普通模式：处理 FastMoCo 双流结构
        elif isinstance(minibatch, list) and len(minibatch) == 2 and isinstance(minibatch[0], (list, tuple)):
            imgs_a, imgs_b = minibatch
            image, mask, label, domain = imgs_b
            image_a, mask_a, label_a, domain_a = imgs_a

        # 普通模式：单流结构
        else:
            image, mask, label, domain = minibatch
            image_a, mask_a, label_a, domain_a = image.clone(), mask.clone(), label.clone(), domain.clone()

        self.optimizer.zero_grad()

        image_new, image_ori = self.img_process(image, mask, self.fundusAug)
        image_new_a, image_ori_a = self.img_process(image_a, mask_a, self.fundusAug)
        # 如果开启频域增强，用 HFC 滤波后的图像替换 image_ori
        if hasattr(self.cfg, 'TRANSFORM') and self.cfg.TRANSFORM.FREQ and image_freq is not None:
            image_ori = self.img_process_freq(image_freq, mask, self.fundusAug)
            image_ori_a = self.img_process_freq(image_freq_a, mask_a, self.fundusAug)

        with autocast():
            # 1. 基础特征提取
            # ResNet 修改后返回中间层和最终层 features: [x2, x3, x4, v]
            # 假设 v (final feat) 是 features[3]
            features_ori = self.network(image_ori, distill=True)
            features_ori_a = self.network(image_ori_a, distill=True)

            # 投影 (Projector)
            features_ori_z1 = self.projector(features_ori_a[3])

            features_new = self.network(image_new, distill=True)
            features_new_a = self.network(image_new_a, distill=True)

            features_new_z2 = self.projector(features_new_a[3])

            # 预测 (Predictor)
            p1, p2 = self.predictor(features_ori_z1), self.predictor(features_new_z2)

            # 归一化
            features_ori_z1 = F.normalize(features_ori_z1, dim=-1)
            features_new_z2 = F.normalize(features_new_z2, dim=-1)
            p1, p2 = F.normalize(p1, dim=-1), F.normalize(p2, dim=-1)

            # 采样正样本 (Queue)
            z1_sup = self.sample_target(features_ori_z1.detach(), label_a)
            z2_sup = self.sample_target(features_new_z2.detach(), label_a)

            # 分类输出
            output_new = self.classifier(features_new[3])
            output_new_a = self.classifier(features_new_a[3])

            # 多头输出 (用于 JS Divergence)
            # 注意：这里使用了中间层特征 features_new[0], [1], [2]
            output_new1 = self.classifier1(features_new[0])  # Layer1 feat
            output_new2 = self.classifier2(features_new[1])  # Layer2 feat
            output_new3 = self.classifier3(features_new[2])  # Layer3 feat

            # Masking 处理
            image_new_masked_ = self.patchify(image_new_a, block=self.cfg.BLOCK)
            image_new_masked = self.random_masking(image_new_masked_, self.cfg.MASK_RATIO)
            image_new_masked = self.unpatchify(image_new_masked, block=self.cfg.BLOCK)

            features_new_masked = self.network(image_new_masked, distill=True)
            output_new_masked = self.classifier(features_new_masked[3])

            # 更新队列
            self.dequeue_and_enqueue(features_ori_z1.detach(), label_a)

            # FastMoCo 逻辑分支
            if getattr(self.cfg, 'FASTMOCO', 0.5) > 0:
                # 混合 (Mixup / Orthogonal Mixing)
                # 这里为了简化，仅实现核心的 Orthogonal Mixing 准备工作

                # Mixup Setup
                noise_std = 0.05
                index = torch.randperm(image_new.size(0)).cuda()
                lam = np.random.beta(1.0, 1.0)
                x_a = image_new
                targets_b = label[index]
                x_b = image_new[index, :]

                # 为了计算方便，这里可能需要 network 再次前向或利用已有特征
                # 省略部分重复计算，聚焦于 Loss 需要的输入
                features_x_mixed = self.network(x_a * lam + x_b * (1 - lam), distill=True)
                mix_z = features_new[3] * lam + features_new[3][index] * (1 - lam)
                mix_z_ = mix_z + torch.normal(mean=0., std=noise_std, size=(mix_z.size())).cuda()
                mix_result = self.classifier(mix_z)

                # Orthogonal Mixing (局部切分与重组)
                x1_in_form = self._local_split(image_ori_a)
                x2_in_form = self._local_split(image_new_a)

                z1_pre = self.network(x1_in_form, distill=True)
                z2_pre = self.network(x2_in_form, distill=True)

                # Split features
                batch_size_split = z1_pre[3].size(0) // (self.split_num ** 2)
                z1_splits = list(z1_pre[3].split(batch_size_split, dim=0))
                z2_splits = list(z2_pre[3].split(batch_size_split, dim=0))

                # Combinations
                z1_orthmix_ = torch.cat(
                    list(map(lambda x: sum(x) / self.combs, list(combinations(z1_splits, r=self.combs)))), dim=0)
                z2_orthmix_ = torch.cat(
                    list(map(lambda x: sum(x) / self.combs, list(combinations(z2_splits, r=self.combs)))), dim=0)

                z1_orthmix_ = self.projector(z1_orthmix_)
                z2_orthmix_ = self.projector(z2_orthmix_)
                p1_orthmix_, p2_orthmix_ = self.predictor(z1_orthmix_), self.predictor(z2_orthmix_)

                # Split back
                z1_orthmix = z1_orthmix_.split(image_ori.size(0), dim=0)
                z2_orthmix = z2_orthmix_.split(image_new.size(0), dim=0)
                p1_orthmix = p1_orthmix_.split(image_ori.size(0), dim=0)
                p2_orthmix = p2_orthmix_.split(image_new.size(0), dim=0)

                # Loss 计算
                # 参数打包需严格对应 Losses.py 中的 forward 签名
                features_loss = [
                    features_ori[3], features_new[3],  # features_ori, features_new
                    features_ori_z1, features_new_z2,  # z1, z2
                    z1_sup, z2_sup,  # z1_sup, z2_sup
                    p1, p2,  # p1, p2
                    z1_orthmix, z2_orthmix,  # z1_orthmix, z2_orthmix
                    p1_orthmix, p2_orthmix,  # p1_orthmix, p2_orthmix
                    features_x_mixed[3], mix_z_, mix_result, targets_b, lam  # Mixup params
                ]

                outputs_loss = [output_new, output_new_a, output_new_masked, None]  # output_ori_masked_c暂空

                loss, loss_dict = self.criterion(
                    outputs_loss,
                    features_loss,
                    None,  # features_masked, Mask-Siam v0 可能没用到这部分
                    [label, label_a],
                    [domain, domain_a],
                    None  # random_matrix
                )

                # JS Divergence Consistency Loss
                temperature = 2.0
                loss += - 0.5 * (
                            F.softmax(output_new / temperature, 1).detach() * F.log_softmax(output_new1 / temperature,
                                                                                            1)).sum() / \
                        output_new1.size()[0]
                loss += - 0.5 * (
                            F.softmax(output_new / temperature, 1).detach() * F.log_softmax(output_new2 / temperature,
                                                                                            1)).sum() / \
                        output_new2.size()[0]
                loss += - 0.5 * (
                            F.softmax(output_new / temperature, 1).detach() * F.log_softmax(output_new3 / temperature,
                                                                                            1)).sum() / \
                        output_new3.size()[0]

                loss += 0.5 * JS_Divergence(output_new, output_new1, output_new2, output_new3)

            else:
                # Fallback to simple Siam
                loss, loss_dict = self.criterion([output_new, output_new_masked],
                                                 [features_ori_z1, features_new_z2, z1_sup, z2_sup, p1, p2], label,
                                                 domain)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss_dict

    def update_epoch(self, epoch):
        self.epoch = epoch
        if hasattr(self.criterion, 'update_alpha'):
            self.criterion.update_alpha(epoch)
        return epoch

    def validate(self, val_loader, test_loader, writer):
        return algorithm_validate(self, val_loader, test_loader, writer)  # 假设 validate.py 支持

    def save_model(self, log_path):
        torch.save(self.network.state_dict(), os.path.join(log_path, 'best_model.pth'))
        torch.save(self.classifier.state_dict(), os.path.join(log_path, 'best_classifier.pth'))
        torch.save(self.projector.state_dict(), os.path.join(log_path, 'best_projector.pth'))
        torch.save(self.predictor.state_dict(), os.path.join(log_path, 'best_predictor.pth'))

    def renew_model(self, log_path):
        self.network.load_state_dict(torch.load(os.path.join(log_path, 'best_model.pth')))
        self.classifier.load_state_dict(torch.load(os.path.join(log_path, 'best_classifier.pth')))

    def predict(self, x):
        return self.classifier(self.network(x))