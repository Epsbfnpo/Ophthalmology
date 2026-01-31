"""
This code is partially borrowed from https://github.com/HobbitLong/SupContrast
"""
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# --- 辅助模块: Logit Adjust & Distance Function ---

class LogitAdjust(nn.Module):
    def __init__(self, cls_num_list, tau=1, weight=None):
        super(LogitAdjust, self).__init__()
        # 确保 cls_num_list 是 tensor
        if not isinstance(cls_num_list, torch.Tensor):
            cls_num_list = torch.tensor(cls_num_list)
        cls_num_list = cls_num_list.float().cuda()

        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = tau * torch.log(cls_p_list)
        self.m_list = m_list.view(1, -1)
        self.weight = weight

    def forward(self, x, target):
        x_m = x + self.m_list
        return F.cross_entropy(x_m, target, weight=self.weight)


def D(p, z, random_matrix=None, version='simplified'):
    if version == 'original':
        z = z.detach()
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        return -(p * z).sum(dim=1).mean()
    elif version == 'simplified':
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    elif version == 'random':
        # 随机投影策略 (Repository B 特性)
        if random_matrix is not None:
            p = torch.matmul(p, random_matrix)
            z = torch.matmul(z, random_matrix)
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception


# --- 原有 SupConLoss (保留) ---
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07, reduction='mean'):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.reduction = reduction

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...]')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        sum_mask = mask.sum(1)
        sum_mask[sum_mask == 0] = 1
        mean_log_prob_pos = (mask * log_prob).sum(1) / sum_mask

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean() if self.reduction == 'mean' else loss.view(anchor_count,
                                                                                                     batch_size)
        return loss


# --- 原有 DahLoss (保留) ---
class DahLoss(nn.Module):
    def __init__(self, max_iteration, training_domains, beta=0.8, scaling_factor=4, alpha=1, temperature=0.07) -> None:
        super(DahLoss, self).__init__()
        # ... (保留原 DahLoss 内容，假设你已有) ...
        # 为节省篇幅，此处省略具体实现，请务必保留原文件中的 DahLoss 类代码
        pass


# =========================================================================
#  核心修复：完全对齐 Repository B 的 DahLoss_Siam_Fastmoco
# =========================================================================

class DahLoss_Siam_Fastmoco(nn.Module):
    def __init__(self, cfg, max_iteration, training_domains, beta=0.8, scaling_factor=4, alpha=1.0, temperature=0.07,
                 fastmoco=1.0):
        super(DahLoss_Siam_Fastmoco, self).__init__()
        self.max_iteration = max_iteration
        self.training_domains = training_domains
        self.alpha = alpha
        self.beta = beta
        self.scaling_factor = scaling_factor
        self.temperature = temperature
        self.gamma = fastmoco
        self.cfg = cfg

        # 域数据统计 (硬编码于算法中，与 Repo B 一致)
        self.domain_num_dict = {'MESSIDOR': 1744, 'IDRID': 516, 'DEEPDR': 2000, 'FGADR': 1842, 'APTOS': 3662,
                                'RLDR': 1593}
        self.label_num_dict = {
            'MESSIDOR': [1016, 269, 347, 75, 35],
            'IDRID': [175, 26, 163, 89, 60],
            'DEEPDR': [917, 214, 402, 353, 113],
            'FGADR': [100, 211, 595, 646, 286],
            'APTOS': [1804, 369, 999, 192, 294],
            'RLDR': [165, 336, 929, 98, 62]
        }

        self.domain_prob, self.label_prob = self.get_domain_label_prob()
        self.domain_prob, self.label_prob = self.multinomial_soomthing(self.domain_prob, self.label_prob, self.beta)

        self.UnsupLoss = SupConLoss(temperature=self.temperature, reduction='none')
        self.SupLoss = nn.CrossEntropyLoss(reduction='none')

    def get_domain_label_prob(self):
        source_domain_num_list = torch.Tensor([self.domain_num_dict[domain] for domain in self.training_domains])
        source_domain_num = torch.sum(source_domain_num_list)
        domain_prob = source_domain_num_list / source_domain_num

        label_num_list = torch.Tensor([self.label_num_dict[domain] for domain in self.training_domains]).sum(dim=0)
        label_num = torch.sum(label_num_list)
        label_prob = label_num_list / label_num
        return domain_prob.cuda(), label_prob.cuda()

    def multinomial_soomthing(self, domain_prob, label_prob, beta=0.8):
        domain_prob = torch.pow(domain_prob, beta)
        label_prob = torch.pow(label_prob, beta)
        domain_prob = domain_prob / torch.sum(domain_prob)
        label_prob = label_prob / torch.sum(label_prob)
        return domain_prob, label_prob

    def get_weights(self, labels, domains):
        domain_prob = torch.index_select(self.domain_prob, 0, domains).cuda()
        domain_weight = 1 / domain_prob
        class_prob = torch.index_select(self.label_prob, 0, labels).cuda()
        class_weight = 1 / class_prob
        return domain_weight, class_weight

    def forward(self, output, features, features_masked, labels, domains, random_matrix):
        # 1. 解包 Label 和 Domain (适配 update 中传入的 [label, label_a])
        labels, labels_a = labels
        domains, domains_a = domains
        domain_weight, class_weight = self.get_weights(labels, domains)
        # domain_weight_a, class_weight_a = self.get_weights(labels_a, domains_a)

        loss_dict = {}

        # 2. 解包 Output
        # output: [output_new, output_new_a, output_new_masked, output_ori_masked_c(None)]
        output, output_new_a, output_new_masked, _ = output

        # 3. 解包 Features (FastMoCo 复杂特征)
        if self.gamma > 0:
            features_ori, features_new, z1, z2, z1_sup, z2_sup, p1, p2, \
                z1_orthmix, z2_orthmix, p1_orthmix, p2_orthmix, \
                features_x_mixed, mix_z_, mix_result, targets_b, lam = features
        else:
            z1, z2, z1_sup, z2_sup, p1, p2 = features

        loss_sup = 0

        # 4. Logit Adjust Loss (解决长尾问题)
        # 必须使用 training_domains[0] 对应的分布，或者动态选择
        self.SupLoss3 = LogitAdjust(self.label_num_dict[self.training_domains[0]], tau=0.6)

        # 计算主分类损失
        loss_sup += self.SupLoss3(output, labels)
        loss_sup += self.SupLoss(output_new_a, labels)  # 辅助增强视角的 Loss

        # 应用域权重和类别权重
        loss_sup = torch.mean(loss_sup * class_weight * domain_weight) / (
                    torch.mean(domain_weight) * torch.mean(class_weight))

        # 5. Mask 一致性 Loss
        temperature = 1.0
        # 强迫 Mask 后的预测分布 逼近 原始图像的预测分布
        loss_sup_mask = - 1 * (
                    F.softmax(output / temperature, 1).detach() * F.log_softmax(output_new_masked / temperature,
                                                                                1)).sum() / output.size()[0]

        # 6. Siam Loss (基础)
        SSL_loss = -0.5 * (torch.sum(p1 * z2.detach(), dim=-1) + torch.sum(p2 * z1.detach(), dim=-1))
        Sup_loss = -0.5 * (torch.sum(p1 * z2_sup, dim=-1) + torch.sum(p2 * z1_sup, dim=-1))
        loss_siam = 0.5 * SSL_loss.mean() + 0.5 * Sup_loss.mean()

        # 7. FastMoCo Loss (Orthogonal Mixing)
        loss_siam_fastmoco = torch.tensor(0.0).cuda()
        if self.gamma > 0:
            SSL_loss_fastmoco = 0.0
            Sup_loss_fastmoco = 0.0

            # 对每一组正交混合特征计算 SimSiam Loss
            for i in range(len(p1_orthmix)):
                p1_orthmix_, p2_orthmix_ = nn.functional.normalize(p1_orthmix[i], dim=-1), nn.functional.normalize(
                    p2_orthmix[i], dim=-1)

                # 使用 D 函数计算距离
                if self.cfg.DG_MODE == 'DG':  # DG 模式下使用随机投影增强
                    SSL_loss_fastmoco += D(p1_orthmix_, z2, random_matrix, version='original') / 2 + D(p2_orthmix_, z1,
                                                                                                       random_matrix,
                                                                                                       version='original') / 2
                    Sup_loss_fastmoco += D(p1_orthmix_, z2_sup, random_matrix, version='original') / 2 + D(p2_orthmix_,
                                                                                                           z1_sup,
                                                                                                           random_matrix,
                                                                                                           version='original') / 2
                else:
                    SSL_loss_fastmoco += D(p1_orthmix_, z2, random_matrix, version='random') / 2 + D(p2_orthmix_, z1,
                                                                                                     random_matrix,
                                                                                                     version='random') / 2
                    Sup_loss_fastmoco += D(p1_orthmix_, z2_sup, random_matrix, version='random') / 2 + D(p2_orthmix_,
                                                                                                         z1_sup,
                                                                                                         random_matrix,
                                                                                                         version='random') / 2

            loss_siam_fastmoco = 0.5 * SSL_loss_fastmoco / len(p1_orthmix) + 0.5 * Sup_loss_fastmoco / len(p1_orthmix)

            # 最终 Loss 组合: 监督 + Mask一致性 + FastMoCo Siam
            # self.cfg.MASKED 通常是权重系数
            mask_weight = getattr(self.cfg, 'MASKED', 1.0)
            sup_weight = getattr(self.cfg, 'SUP', 0.5)

            loss = sup_weight * loss_sup + mask_weight * loss_sup_mask + loss_siam_fastmoco * self.gamma
        else:
            print(" Not FastMOCO")
            loss = loss_sup + loss_siam * self.alpha

        loss_dict['loss'] = loss.item()
        loss_dict['loss_sup'] = loss_sup.item()
        loss_dict['loss_siam'] = loss_siam.item()
        loss_dict['loss_siam_fastmoco'] = loss_siam_fastmoco.item()

        return loss, loss_dict

    def update_alpha(self, iteration):
        self.alpha = 1 - iteration / self.max_iteration * 0.5
        return self.alpha


# 别名 (兼容旧代码引用)
DahLoss_Siam = DahLoss_Siam_Fastmoco