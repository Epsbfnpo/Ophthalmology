"""
This code is partially borrowed from https://github.com/HobbitLong/SupContrast
"""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ==============================================================================
# Helper Functions & Classes from MASK_SIAM (Must be included)
# ==============================================================================

def JS_Divergence(output_clean, output_aug1, output_aug2, output_aug3):
    p_clean, p_aug1 = F.softmax(output_clean, dim=1), F.softmax(output_aug1, dim=1)
    p_aug2, p_aug3 = F.softmax(output_aug2, dim=1), F.softmax(output_aug3, dim=1)
    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2 + p_aug3) / 4., 1e-7, 1).log()
    loss_ctr = (F.kl_div(p_mixture, p_clean, reduction='batchmean') + F.kl_div(p_mixture, p_aug1,
                                                                               reduction='batchmean') + F.kl_div(
        p_mixture, p_aug2, reduction='batchmean') + F.kl_div(p_mixture, p_aug3, reduction='batchmean')) / 4.
    return loss_ctr

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class LogitAdjust(nn.Module):
    def __init__(self, cls_num_list, tau=1, weight=None):
        super(LogitAdjust, self).__init__()
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = tau * torch.log(cls_p_list)
        self.m_list = m_list.view(1, -1)
        self.weight = weight

    def forward(self, x, target):
        x_m = x + self.m_list
        return F.cross_entropy(x_m, target, weight=self.weight)

# ==============================================================================
# Original Classes (Preserved for compatibility)
# ==============================================================================

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, reduction = 'mean'):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.reduction = reduction

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
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

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos

        if self.reduction == 'mean':
            loss = loss.view(anchor_count, batch_size).mean()
        else:
            loss = loss.view(anchor_count, batch_size)

        return loss

class DahLoss(nn.Module):
    def __init__(self, max_iteration, training_domains, beta = 0.8, scaling_factor = 4, alpha = 1, temperature = 0.07) -> None:
        super(DahLoss, self).__init__()
        self.max_iteration = max_iteration
        self.training_domains = training_domains
        self.alpha = alpha
        self.beta = beta
        self.scaling_factor = scaling_factor
        self.temperature = temperature

        self.domain_num_dict = {'MESSIDOR': 1744,
                                'IDRID': 516,
                                'DEEPDR': 2000,
                                'FGADR': 1842,
                                'APTOS': 3662,
                                'RLDR': 1593}

        self.label_num_dict = {'MESSIDOR': [1016, 269, 347, 75, 35],
                                'IDRID': [175, 26, 163, 89, 60],
                                'DEEPDR': [917, 214, 402, 353, 113],
                                'FGADR': [100, 211, 595, 646, 286],
                                'APTOS': [1804, 369, 999, 192, 294],
                                'RLDR': [165, 336, 929, 98, 62]}

        self.domain_prob, self.label_prob = self.get_domain_label_prob()
        self.domain_prob, self.label_prob = self.multinomial_soomthing(self.domain_prob, self.label_prob, self.beta)

        self.UnsupLoss = SupConLoss(temperature = self.temperature, reduction='none')
        self.SupLoss = nn.CrossEntropyLoss(reduction='none')

    def get_domain_label_prob(self):
        source_domain_num_list = torch.Tensor([self.domain_num_dict[domain] for domain in self.training_domains])
        source_domain_num = torch.sum(source_domain_num_list)
        domain_prob = source_domain_num_list / source_domain_num

        label_num_list = torch.Tensor([self.label_num_dict[domain] for domain in  self.training_domains]).sum(dim=0)
        label_num = torch.sum(label_num_list)
        label_prob = label_num_list / label_num

        return domain_prob.cuda(), label_prob.cuda()

    def multinomial_soomthing(self, domain_prob, label_prob, beta = 0.8):
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

    def forward(self, output, features, labels, domains):

        domain_weight, class_weight = self.get_weights(labels, domains)

        loss_dict = {}

        features_ori, features_new = features

        loss_sup = 0

        for op_item in output:
            loss_sup += self.SupLoss(op_item, labels)

        features_multi = torch.stack([features_ori, features_new], dim = 1)
        features_multi = F.normalize(features_multi, p=2, dim=2)

        loss_unsup = torch.mean(self.UnsupLoss(features_multi))
        loss_sup = torch.mean(loss_sup * class_weight * domain_weight) / (torch.mean(domain_weight) * torch.mean(class_weight))

        loss = (1 - self.alpha) * loss_sup + self.alpha * loss_unsup / self.scaling_factor

        loss_dict['loss'] = loss.item()
        loss_dict['loss_sup'] = loss_sup.item()
        loss_dict['loss_unsup'] = loss_unsup.item()

        return loss, loss_dict

    def update_alpha(self, iteration):
        self.alpha = 1 - iteration / self.max_iteration
        return self.alpha

# ==============================================================================
# New MASK_SIAM Algorithm Loss (The Critical Part)
# ==============================================================================

class DahLoss_Siam_Fastmoco(nn.Module):
    def __init__(self, cfg, max_iteration, training_domains, beta = 0.8, scaling_factor = 4, alpha = 1.0, temperature = 0.07,  fastmoco=1.0) -> None:
        super(DahLoss_Siam_Fastmoco, self).__init__()
        self.max_iteration = max_iteration
        self.training_domains = training_domains
        self.alpha = alpha
        self.beta = beta
        self.scaling_factor = scaling_factor
        self.temperature = temperature
        self.gamma = fastmoco
        self.cfg=cfg
        self.domain_num_dict = {'MESSIDOR': 1744,
                                'IDRID': 516,
                                'DEEPDR': 2000,
                                'FGADR': 1842,
                                'APTOS': 3662,
                                'RLDR': 1593}

        self.label_num_dict = {'MESSIDOR': [1016, 269, 347, 75, 35],
                                'IDRID': [175, 26, 163, 89, 60],
                                'DEEPDR': [917, 214, 402, 353, 113],
                                'FGADR': [100, 211, 595, 646, 286],
                                'APTOS': [1804, 369, 999, 192, 294],
                                'RLDR': [165, 336, 929, 98, 62]}

        self.domain_prob, self.label_prob = self.get_domain_label_prob()
        self.domain_prob, self.label_prob = self.multinomial_soomthing(self.domain_prob, self.label_prob, self.beta)

        self.UnsupLoss = SupConLoss(temperature = self.temperature, reduction='none')
        self.SupLoss = nn.CrossEntropyLoss(reduction='none')
        self.SupLoss2 = nn.CrossEntropyLoss()


    def get_domain_label_prob(self):
        source_domain_num_list = torch.Tensor([self.domain_num_dict[domain] for domain in self.training_domains])
        source_domain_num = torch.sum(source_domain_num_list)
        domain_prob = source_domain_num_list / source_domain_num

        label_num_list = torch.Tensor([self.label_num_dict[domain] for domain in  self.training_domains]).sum(dim=0)
        label_num = torch.sum(label_num_list)
        label_prob = label_num_list / label_num

        return domain_prob.cuda(), label_prob.cuda()

    def multinomial_soomthing(self, domain_prob, label_prob, beta = 0.8):
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

    def forward(self, output, features, labels, domains):

        domain_weight, class_weight = self.get_weights(labels, domains)

        loss_dict = {}

        output, output_new_a, output_new_masked = output

        # Unpacking features based on FastMoCo configuration
        if self.gamma > 0:
            features_ori, features_new, z1,z2, z1_sup, z2_sup, p1, p2, z1_orthmix, z2_orthmix, p1_orthmix, p2_orthmix, features_x_mixed, mix_z_, mix_result, targets_b, lam = features
        else:
             z1,z2, z1_sup, z2_sup, p1, p2 = features

        loss_sup = 0

        # 使用 Logit Adjustment 进行类别平衡 Loss 计算
        self.SupLoss3 = LogitAdjust(self.label_num_dict[self.training_domains[0]], tau=0.6)

        loss_sup += self.SupLoss3(output, labels)

        temperature = 1.0

        # Masked Distillation Loss
        loss_sup_mask = - 1  * (F.softmax( output / temperature, 1).detach() * F.log_softmax(output_new_masked / temperature, 1)).sum() / output.size()[0]

        # Standard Siam Loss
        SSL_loss = -0.5 * (torch.sum(p1 * z2.detach(), dim=-1) + torch.sum(p2 * z1.detach(), dim=-1))
        Sup_loss = -0.5 * (torch.sum(p1 * z2_sup, dim=-1) + torch.sum(p2 * z1_sup, dim=-1))
        loss_siam = 0.5 * SSL_loss.mean()  + 0.5 * Sup_loss.mean()


        if self.gamma > 0:
            # FastMoCo (Orthogonal Mixing) Loss Calculation
            SSL_loss_fastmoco = 0.0
            Sup_loss_fastmoco = 0.0

            for i in range(len(p1_orthmix)):
                p1_orthmix_, p2_orthmix_ = nn.functional.normalize(p1_orthmix[i], dim=-1), nn.functional.normalize(p2_orthmix[i], dim=-1)
                SSL_loss_fastmoco += -0.5 * (torch.sum(p1_orthmix_ * z2.detach(), dim=-1).mean() + torch.sum(p2_orthmix_ * z1.detach(), dim=-1).mean())
                Sup_loss_fastmoco += -0.5 * (torch.sum(p1_orthmix_ * z2_sup, dim=-1).mean() + torch.sum(p2_orthmix_ * z1_sup, dim=-1).mean())

            loss_siam_fastmoco = 0.5 * SSL_loss_fastmoco / len(p1_orthmix) + 0.5 * Sup_loss_fastmoco/ len(p1_orthmix)

            loss =  loss_siam_fastmoco * self.gamma
        else:
            print(" Not FastMOCO")
            loss =   loss_sup  +  loss_siam * self.alpha

        # Note: Mixup Loss logic is preserved below in case it's enabled in future versions
        # criterion = nn.CrossEntropyLoss()
        # loss_mix = F.mse_loss(features_x_mixed, mix_z_.detach())
        # mix_cate_loss = mixup_criterion(criterion,mix_result,labels,targets_b,lam)

        loss_dict['loss'] = loss.item()
        loss_dict['loss_sup'] = loss_sup.item()
        loss_dict['loss_siam'] = loss_siam.item()

        # Only populate fastmoco loss if gamma > 0, otherwise repeat siam loss or 0
        if self.gamma > 0:
            loss_dict['loss_siam_fastmoco'] = loss_siam_fastmoco.item()
        else:
            loss_dict['loss_siam_fastmoco'] = loss_siam.item()

        return loss, loss_dict

    def update_alpha(self, iteration):
        self.alpha = 1 - iteration / self.max_iteration * 0.5
        return self.alpha