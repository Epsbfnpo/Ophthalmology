"""
This code collected some methods from DomainBed (https://github.com/facebookresearch/DomainBed) and other SOTA methods.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, collections, logging
from collections import OrderedDict
from itertools import combinations
from torch.cuda.amp import GradScaler, autocast
import copy

import utils.misc as misc
from utils.validate import algorithm_validate, algorithm_validate_distill
import modeling.model_manager as models
from modeling.losses import DahLoss, DahLoss_Mask, DahLoss_Siam_Fastmoco
from modeling.nets import LossValley, AveragedModel
from dataset.data_manager import get_post_FundusAug

from backpack import backpack, extend
from backpack.extensions import BatchGrad

from guided_filter_pytorch.ModifiedHFCFilter import FourierButterworthHFCFilter

ALGORITHMS = [
    'ERM',
    'GDRNet',
    'GDRNet_Mask',
    'GDRNet_MASK_SIAM',
    'GREEN',
    'CABNet',
    'MixupNet',
    'MixStyleNet',
    'Fishr',
    'DRGen'
    ]

def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]

# ==============================================================================
# MASK_SIAM Helper Functions & Classes (The "Brain" Components)
# ==============================================================================

def init_weights_MLPHead(module, init_method='He'):
    for _, m in module.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if init_method == 'He':
                nn.init.kaiming_normal_(m.weight.data)
            elif init_method == 'Xavier':
                nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, val=0)

def hfc_mul_mask(hfc_filter, image, mask, do_norm=False):
    hfc = hfc_filter((image / 2 + 0.5), mask)
    if do_norm:
        hfc = 2 * hfc - 1
    return (hfc + 1) * mask - 1

def JS_Divergence(output_clean, output_aug1, output_aug2, output_aug3):
    p_clean, p_aug1 = F.softmax(output_clean, dim=1), F.softmax(output_aug1, dim=1)
    p_aug2, p_aug3 = F.softmax(output_aug2, dim=1), F.softmax(output_aug3, dim=1)
    p_mixture = torch.clamp((p_aug1 + p_aug2 + p_aug3) / 3., 1e-7, 1).log()
    loss_ctr = F.kl_div(p_mixture, p_clean, reduction='batchmean')
    return loss_ctr

class LogitAdjust(nn.Module):
    def __init__(self, cls_num_list, tau=1, weight=None):
        super(LogitAdjust, self).__init__()
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        m_list = torch.log(torch.pow(cls_num_list, tau) + 1e-9)
        self.m_list = m_list.view(1, -1)
        self.weight = weight

    def forward(self, x, target):
        x_m = x + self.m_list
        return F.cross_entropy(x_m, target, weight=self.weight)

class LogitAdjust_KD(nn.Module):
    def __init__(self, cls_num_list, tau=1, weight=None):
        super(LogitAdjust_KD, self).__init__()
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        m_list = torch.log(torch.pow(cls_num_list, tau) + 1e-9)
        self.m_list = m_list.view(1, -1)
        self.weight = weight

    def forward(self, x, target):
        x_m = x + self.m_list
        target = target + self.m_list
        target = F.softmax(target, 1)
        return F.cross_entropy(x_m, target, weight=self.weight)

class LogitAdjust_KD_V2(nn.Module):
    def __init__(self, cls_num_list, tau1=1, tau2=1, temperature=1.0, weight=None):
        super(LogitAdjust_KD_V2, self).__init__()
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        m_list1 = torch.log(torch.pow(cls_num_list, tau1) + 1e-9)
        m_list2 = torch.log(torch.pow(cls_num_list, tau2) + 1e-9)

        self.m_list1 = m_list1.view(1, -1)
        self.m_list2 = m_list2.view(1, -1)

        self.weight = weight
        self.temperature = temperature

    def forward(self, x, target, label):
        x_m = x + self.m_list1
        x_m = F.log_softmax(x_m / self.temperature, dim=1)
        target = target + self.m_list2
        target_t = F.softmax(target / self.temperature, 1).detach()
        output_target_max, output_target_index = torch.max(F.softmax(target, dim=1).detach(), dim=1)

        return -(target_t[(output_target_index == label)] * x_m[(output_target_index == label)]).sum() / target.size()[0]

# ==============================================================================
# Main Algorithm Class Definition
# ==============================================================================

class Algorithm(torch.nn.Module):
    def __init__(self, num_classes, cfg):
        super(Algorithm, self).__init__()
        self.cfg = cfg
        self.epoch = 0

    def update(self, minibatches):
        raise NotImplementedError

    def update_epoch(self, epoch):
        self.epoch = epoch
        return epoch

    def validate(self, val_loader, test_loader, writer):
        raise NotImplementedError

    def save_model(self, log_path):
        raise NotImplementedError

    def renew_model(self, log_path):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

class ERM(Algorithm):
    def __init__(self, num_classes, cfg):
        super(ERM, self).__init__(num_classes, cfg)
        self.network = models.get_net(cfg)
        self.classifier = models.get_classifier(self.network.out_features(), cfg)
        self.optimizer = torch.optim.SGD(
            [{"params":self.network.parameters()},
            {"params":self.classifier.parameters()}],
            lr = cfg.LEARNING_RATE,
            momentum = cfg.MOMENTUM,
            weight_decay = cfg.WEIGHT_DECAY,
            nesterov=True)

    def update(self, minibatch):
        image, mask, label, domain = minibatch
        self.optimizer.zero_grad()
        features = self.network(image)
        output = self.classifier(features)
        loss = F.cross_entropy(output, label)
        loss.backward()
        self.optimizer.step()
        return {'loss':loss}

    def validate(self, val_loader, test_loader, writer):
        val_auc = -1
        test_auc = -1
        if self.epoch <= self.cfg.EPOCHS:
            val_auc, val_loss = algorithm_validate(self, val_loader, writer, self.epoch, 'val')
            test_auc, test_loss = algorithm_validate(self, test_loader, writer, self.epoch, 'test')
            if self.epoch == self.cfg.EPOCHS:
                self.epoch += 1
        else:
            test_auc, test_loss = algorithm_validate(self, test_loader, writer, self.cfg.EPOCHS + self.cfg.VAL_EPOCH, 'test')
            logging.info('Best performance on test domain(s): {}'.format(test_auc))
        return val_auc, test_auc

    def save_model(self, log_path):
        logging.info("Saving best model...")
        torch.save(self.network.state_dict(), os.path.join(log_path, 'best_model.pth'))
        torch.save(self.classifier.state_dict(), os.path.join(log_path, 'best_classifier.pth'))

    def renew_model(self, log_path):
        net_path = os.path.join(log_path, 'best_model.pth')
        classifier_path = os.path.join(log_path, 'best_classifier.pth')
        self.network.load_state_dict(torch.load(net_path))
        self.classifier.load_state_dict(torch.load(classifier_path))

    def predict(self, x):
        return self.classifier(self.network(x))

class GDRNet(ERM):
    def __init__(self, num_classes, cfg):
        super(GDRNet, self).__init__(num_classes, cfg)
        self.fundusAug = get_post_FundusAug(cfg)
        self.criterion = DahLoss(beta= cfg.GDRNET.BETA, max_iteration = cfg.EPOCHS, \
                                training_domains = cfg.DATASET.SOURCE_DOMAINS, temperature = cfg.GDRNET.TEMPERATURE, \
                                scaling_factor = cfg.GDRNET.SCALING_FACTOR)

    def img_process(self, img_tensor, mask_tensor, fundusAug):
        img_tensor_new, mask_tensor_new = fundusAug['post_aug1'](img_tensor.clone(), mask_tensor.clone())
        img_tensor_new = img_tensor_new * mask_tensor_new
        img_tensor_new = fundusAug['post_aug2'](img_tensor_new)
        img_tensor_ori = fundusAug['post_aug2'](img_tensor)
        return img_tensor_new, img_tensor_ori

    def update(self, minibatch):
        image, mask, label, domain = minibatch
        self.optimizer.zero_grad()
        image_new, image_ori = self.img_process(image, mask, self.fundusAug)
        features_ori = self.network(image_ori)
        features_new = self.network(image_new)
        output_new = self.classifier(features_new)
        loss, loss_dict_iter = self.criterion([output_new], [features_ori, features_new], label, domain)
        loss.backward()
        self.optimizer.step()
        return loss_dict_iter

    def update_epoch(self, epoch):
        self.epoch = epoch
        return self.criterion.update_alpha(epoch)

class GDRNet_Mask(ERM):
    def __init__(self, num_classes, cfg):
        super(GDRNet_Mask, self).__init__(num_classes, cfg)
        self.cfg=cfg
        self.fundusAug = get_post_FundusAug(cfg)
        self.criterion = DahLoss_Mask(beta= cfg.GDRNET.BETA, max_iteration = cfg.EPOCHS, \
                                training_domains = cfg.DATASET.SOURCE_DOMAINS, temperature = cfg.GDRNET.TEMPERATURE, \
                                scaling_factor = cfg.GDRNET.SCALING_FACTOR)

    def img_process(self, img_tensor, mask_tensor, fundusAug):
        img_tensor_new, mask_tensor_new = fundusAug['post_aug1'](img_tensor.clone(), mask_tensor.clone())
        img_tensor_new = img_tensor_new * mask_tensor_new
        img_tensor_new = fundusAug['post_aug2'](img_tensor_new)
        img_tensor_ori = fundusAug['post_aug2'](img_tensor)
        return img_tensor_new, img_tensor_ori

    def patchify(self, imgs, block=32):
        p = block
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        h = w = imgs.shape[2] // p
        n = imgs.shape[0]
        x = imgs.reshape(shape=(n, 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        p = 32
        h = w = int(x.shape[1] ** .5)
        n = x.shape[0]
        assert h * w == x.shape[1]
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
        x_masked_c =  x * (1 - mask.unsqueeze(-1))
        return x_masked, x_masked_c

    def update(self, minibatch):
        image, mask, label, domain = minibatch
        self.optimizer.zero_grad()
        image_new, image_ori = self.img_process(image, mask, self.fundusAug)
        features_ori = self.network(image_ori)
        features_new = self.network(image_new)
        output_new = self.classifier(features_new)

        image_new_masked = self.patchify(image_new, block=self.cfg.BLOCK)
        image_new_masked, image_new_masked_c = self.random_masking(image_new_masked, self.cfg.MASK_RATIO)
        image_new_masked = self.unpatchify(image_new_masked)
        image_new_masked_c = self.unpatchify(image_new_masked_c)

        features_new_masked = self.network(image_new_masked)
        output_new_masked = self.classifier(features_new_masked)

        features_new_masked_c = self.network(image_new_masked_c)
        output_new_masked_c = self.classifier(features_new_masked_c)

        loss, loss_dict_iter = self.criterion([output_new, output_new_masked, output_new_masked_c], [features_ori, features_new], label, domain)
        loss.backward()
        self.optimizer.step()
        return loss_dict_iter

    def update_epoch(self, epoch):
        self.epoch = epoch
        return self.criterion.update_alpha(epoch)

# ==============================================================================
# THE NEW ALGORITHM CLASS (Strict Reconstruction)
# ==============================================================================

class GDRNet_MASK_SIAM(Algorithm):
    def __init__(self, num_classes, cfg):
        super(GDRNet_MASK_SIAM, self).__init__(num_classes, cfg)

        self.cfg = cfg
        self.network = models.get_net(cfg)

        # Determine dimensions for auxiliary classifiers
        # Usually: layer1=256, layer2=512, layer3=1024, layer4=2048 (for ResNet50)
        # We access network.out_features() which is usually the final dimension.
        final_dim = self.network.out_features()
        if final_dim == 2048: # ResNet 50/101/152
            dim_in1 = 2048
            feat_dim1 = 512
            dims = [256, 512, 1024]
        else: # ResNet 18/34 (final_dim 512)
            dim_in1 = 512
            feat_dim1 = 256
            dims = [64, 128, 256]

        self.classifier = models.get_classifier(final_dim, cfg)
        self.classifier1 = models.get_classifier(dims[0], cfg) # Output from layer 1 (x2)
        self.classifier2 = models.get_classifier(dims[1], cfg) # Output from layer 2 (x3)
        self.classifier3 = models.get_classifier(dims[2], cfg) # Output from layer 3 (x4)

        # Projector & Predictor
        self.projector = nn.Sequential(nn.Linear(dim_in1, dim_in1, bias=False),
                                       nn.BatchNorm1d(dim_in1),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(dim_in1, feat_dim1, bias=False),
                                    nn.BatchNorm1d(feat_dim1),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(feat_dim1, dim_in1, bias=False),
                                    nn.BatchNorm1d(dim_in1, affine=False)
                                    )

        self.predictor = nn.Sequential(nn.Linear(dim_in1, feat_dim1, bias=False),
                                        nn.BatchNorm1d(feat_dim1),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(feat_dim1, dim_in1)
                                        )

        init_weights_MLPHead(self.projector, init_method='He')
        init_weights_MLPHead(self.predictor, init_method='He')

        # Optimizer with fix_lr policy
        self.optimizer = torch.optim.AdamW(
            [{"params":self.network.parameters(), 'fix_lr': False},
            {"params":self.classifier.parameters(), 'fix_lr': False},
            {"params":self.classifier1.parameters(), 'fix_lr': False},
            {"params":self.classifier2.parameters(), 'fix_lr': False},
            {"params":self.classifier3.parameters(), 'fix_lr': False},
            {"params":self.projector.parameters(), 'fix_lr': False},
            {"params":self.predictor.parameters(), 'fix_lr': True},
            ],
            lr = cfg.LEARNING_RATE,
            weight_decay = 0.0001
            )

        # Memory Queue
        K= 1024
        dim= 2048
        self.K=K
        self.num_positive = 0
        self.register_buffer("queue", torch.randn(K, dim))
        self.queue = nn.functional.normalize(self.queue, dim=-1)
        self.register_buffer("queue_labels", -torch.ones(K, dtype=torch.long))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.fundusAug = get_post_FundusAug(cfg)
        self.alpha = 4.0
        self.beta = 2.0

        # Loss Initialization
        self.split_num = 2
        self.combs = 3
        self.criterion = DahLoss_Siam_Fastmoco(cfg=self.cfg, beta= cfg.GDRNET.BETA, max_iteration = cfg.EPOCHS, \
                                training_domains = cfg.DATASET.SOURCE_DOMAINS, temperature = cfg.GDRNET.TEMPERATURE, \
                                scaling_factor = cfg.GDRNET.SCALING_FACTOR, fastmoco=cfg.FASTMOCO)

        # HFC Filter
        self.hfc_filter_in = FourierButterworthHFCFilter(butterworth_d0_ratio=0.003, butterworth_n=1,
                                                         do_median_padding=False, image_size=(256, 256)).cuda()

        self.label_num_dict = {'MESSIDOR': [1016, 269, 347, 75, 35],
                                'IDRID': [175, 26, 163, 89, 60],
                                'DEEPDR': [917, 214, 402, 353, 113],
                                'FGADR': [100, 211, 595, 646, 286],
                                'APTOS': [1804, 369, 999, 192, 294],
                                'RLDR': [165, 336, 929, 98, 62]}

    def patchify(self, imgs, block=32):
        p = block
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
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
        assert h * w == x.shape[1]
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
        x_masked_c = x * (1 - mask.unsqueeze(-1))
        return x_masked, x_masked_c

    @torch.no_grad()
    def dequeue_and_enqueue(self, features, labels):
        batch_size = features.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0
        self.queue[ptr : ptr+batch_size, :] = features
        self.queue_labels[ptr : ptr+batch_size] = labels
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def sample_target(self, features, labels):
        neighbor = []
        for i, label in enumerate(labels):
            pos = torch.where(self.queue_labels == label)[0]
            if len(pos) != 0:
                if self.num_positive > 0:
                    choice = torch.multinomial(torch.ones_like(pos).type(torch.FloatTensor), self.num_positive, replacement=True)
                    idx = pos[choice]
                    neighbor.append(self.queue[idx].mean(0))
                else:
                    neighbor.append(self.queue[pos].mean(0))
            else:
                neighbor.append(features[i])
        neighbor = torch.stack(neighbor, dim=0)
        return neighbor

    def img_process(self, img_tensor, mask_tensor, fundusAug):
        img_tensor_new, mask_tensor_new = fundusAug['post_aug1'](img_tensor.clone(), mask_tensor.clone())
        img_tensor_new = img_tensor_new * mask_tensor_new
        img_tensor_new = fundusAug['post_aug2'](img_tensor_new)
        img_tensor_ori = fundusAug['post_aug2'](img_tensor)
        return img_tensor_new, img_tensor_ori

    def _local_split(self, x):
        _side_indent = x.size(2) // self.split_num, x.size(3) // self.split_num
        cols = x.split(_side_indent[1], dim=3)
        xs = []
        for _x in cols:
            xs += _x.split(_side_indent[0], dim=2)
        x = torch.cat(xs, dim=0)
        return x

    def update(self, minibatch):
        amp_grad_scaler = GradScaler()

        # Strict input unpacking based on expected DataLoader update in Phase 4
        # DataLoader MUST return 5 items including frequency augmented image
        image, image_feq, mask, label, domain = minibatch

        self.optimizer.zero_grad()

        image_new, image_ori = self.img_process(image, mask, self.fundusAug)

        with autocast():
            # 1. Forward Pass
            # Use distill=True to get feature list: [x2, x3, x4, final]
            features_ori_list = self.network(image_ori, distill=True)
            features_ori = features_ori_list[-1]
            features_ori_z1 = self.projector(features_ori)

            features_new_list = self.network(image_new, distill=True)
            features_new = features_new_list[-1]
            features_new_z2 = self.projector(features_new)

            p1, p2 = self.predictor(features_ori_z1), self.predictor(features_new_z2)

            features_ori_z1 = nn.functional.normalize(features_ori_z1, dim=-1)
            features_new_z2 = nn.functional.normalize(features_new_z2, dim=-1)
            p1, p2 = nn.functional.normalize(p1, dim=-1), nn.functional.normalize(p2, dim=-1)

            # Sample supervised targets
            z1_sup = self.sample_target(features_ori_z1.detach(), label)
            z2_sup = self.sample_target(features_new_z2.detach(), label)

            # Main outputs
            output_new = self.classifier(features_new)
            output_ori = self.classifier(features_ori)

            # Auxiliary outputs (Distillation)
            # features_new_list[0] -> layer1 out -> classifier1
            # features_new_list[1] -> layer2 out -> classifier2
            # features_new_list[2] -> layer3 out -> classifier3
            output_new1 = self.classifier1(features_new_list[0])
            output_new2 = self.classifier2(features_new_list[1])
            output_new3 = self.classifier3(features_new_list[2])

            # Masking Logic
            image_new_masked_ = self.patchify(image_new, block=self.cfg.BLOCK)
            image_new_masked, image_new_masked_c = self.random_masking(image_new_masked_, self.cfg.MASK_RATIO)
            image_new_masked = self.unpatchify(image_new_masked, block=self.cfg.BLOCK)
            image_new_masked_c = self.unpatchify(image_new_masked_c, block=self.cfg.BLOCK)

            # Masked Student Forward
            features_new_masked_list = self.network(image_new_masked, drop_rate=0.3, distill=True)
            features_new_masked = features_new_masked_list[-1]
            features_new_masked_c_list = self.network(image_new_masked_c, distill=True)
            features_new_masked_c = features_new_masked_c_list[-1]

            output_new_masked = self.classifier(features_new_masked)
            output_new_masked1 = self.classifier1(features_new_masked_list[0])
            output_new_masked2 = self.classifier2(features_new_masked_list[1])
            output_new_masked3 = self.classifier3(features_new_masked_list[2])

            output_new_masked_c1 = self.classifier1(features_new_masked_c_list[0])
            output_new_masked_c2 = self.classifier2(features_new_masked_c_list[1])
            output_new_masked_c3 = self.classifier3(features_new_masked_c_list[2])

            # Dequeue and enqueue
            self.dequeue_and_enqueue(features_ori_z1.detach(), label)

            # 2. FastMoCo Logic
            noise_std = 0.05
            index = torch.randperm(image_new.size(0)).cuda()
            lam = np.random.beta(1.0, 1.0)
            x_a = image_new
            x_b = copy.deepcopy(image_new[index,:])
            targets_b = copy.deepcopy(label[index])
            x_mixed = x_a * lam + x_b * (1-lam)

            features_x_mixed_list = self.network(x_mixed, distill=True)
            features_x_mixed = features_x_mixed_list[-1]
            features_x_a_list = self.network(x_a, distill=True)
            features_x_b_list = self.network(x_b, distill=True)

            mix_z = features_x_a_list[-1] * lam + features_x_b_list[-1] * (1-lam)
            mix_z_ = mix_z + torch.normal(mean=0., std=noise_std, size=(mix_z.size())).cuda()

            mix_result = self.classifier(mix_z) + self.classifier(features_x_mixed)

            # Local Split & Orthogonal Mixing
            x1_in_form = self._local_split(image_ori)
            x2_in_form = self._local_split(image_new)

            z1_pre_list = self.network(x1_in_form, distill=True)
            z2_pre_list = self.network(x2_in_form, distill=True)
            z1_pre = z1_pre_list[-1]
            z2_pre = z2_pre_list[-1]

            z1_splits = list(z1_pre.split(z1_pre.size(0) // self.split_num ** 2, dim=0))
            z2_splits = list(z2_pre.split(z2_pre.size(0) // self.split_num ** 2, dim=0))

            z1_orthmix_ = torch.cat(list(map(lambda x: sum(x) / self.combs, list(combinations(z1_splits, r=self.combs)))), dim=0)
            z2_orthmix_ = torch.cat(list(map(lambda x: sum(x) / self.combs, list(combinations(z2_splits, r=self.combs)))), dim=0)

            z1_orthmix_ = self.projector(z1_orthmix_)
            z2_orthmix_ = self.projector(z2_orthmix_)

            p1_orthmix_, p2_orthmix_ = self.predictor(z1_orthmix_), self.predictor(z2_orthmix_)

            z1_orthmix = z1_orthmix_.split(image_ori.size(0), dim=0)
            z2_orthmix = z2_orthmix_.split(image_new.size(0), dim=0)

            p1_orthmix = p1_orthmix_.split(image_ori.size(0), dim=0)
            p2_orthmix = p2_orthmix_.split(image_new.size(0), dim=0)

            # 3. Loss Calculation
            # Base Loss (Contrastive + Sup)
            loss_fastcoco, loss_dict_iter = self.criterion(
                [output_new, output_ori, output_new_masked],
                [features_new_list[-1], features_new, features_ori_z1, features_new_z2,
                 z1_sup, z2_sup, p1, p2, z1_orthmix, z2_orthmix, p1_orthmix, p2_orthmix,
                 features_x_mixed, mix_z_, mix_result, targets_b, lam],
                label, domain
            )

            # Logit Adjustment Logic (Restoring Critical Core)
            temperature = 2.0
            tau1, tau2, tau3, tau = -0.2, 1.0, 1.5, 0.55

            self.SupLoss3 = LogitAdjust_KD_V2(self.label_num_dict[self.cfg.DATASET.SOURCE_DOMAINS[0]], tau1=tau, tau2=tau3)
            self.SupLoss2 = LogitAdjust_KD_V2(self.label_num_dict[self.cfg.DATASET.SOURCE_DOMAINS[0]], tau1=tau, tau2=tau2)
            self.SupLoss1 = LogitAdjust_KD_V2(self.label_num_dict[self.cfg.DATASET.SOURCE_DOMAINS[0]], tau1=tau, tau2=tau1)
            self.SupLoss = LogitAdjust_KD(self.label_num_dict[self.cfg.DATASET.SOURCE_DOMAINS[0]], tau=tau)

            loss_sup = 0
            loss1_sup = 0
            loss2_sup = 0
            loss3_sup = 0

            # Calculate LA losses
            TEMP = self.cfg.GDRNET.MASKED * min(self.epoch / 25, 1.0) * 0.5

            loss3_sup += TEMP * self.SupLoss3(output_new, output_new3, label)
            loss2_sup += TEMP * self.SupLoss2(output_new, output_new2, label)
            loss1_sup += TEMP * self.SupLoss1(output_new, output_new1, label)

            # Masked Logit Distillation
            loss1_sup += - 0.5 * (F.softmax(output_new / temperature, 1).detach() * F.log_softmax(output_new_masked_c1 / temperature, 1)).sum() / output_new1.size()[0]
            loss2_sup += - 0.5 * (F.softmax(output_new / temperature, 1).detach() * F.log_softmax(output_new_masked_c2 / temperature, 1)).sum() / output_new2.size()[0]
            loss3_sup += - 0.5 * (F.softmax(output_new / temperature, 1).detach() * F.log_softmax(output_new_masked_c3 / temperature, 1)).sum() / output_new3.size()[0]

            # Main Sup Loss
            loss_sup += - 0.5 * (F.softmax(output_new / temperature, 1).detach() * F.log_softmax(output_new_masked / temperature, 1)).sum() / output_new.size()[0]
            loss_sup += TEMP * self.SupLoss(output_new_masked, output_new)

            # JS Divergence
            loss_js = 0.5 * JS_Divergence(output_new, output_new1, output_new2, output_new3)

            # Final Loss Combination
            loss = loss_fastcoco + loss_sup + loss1_sup + loss2_sup + loss3_sup + loss_js

            loss_dict_iter['loss'] = loss.item()
            loss_dict_iter['loss_sup'] = loss_sup.item()
            loss_dict_iter['loss1_sup'] = loss1_sup.item()
            loss_dict_iter['loss2_sup'] = loss2_sup.item()
            loss_dict_iter['loss3_sup'] = loss3_sup.item()

        # Backward Pass
        amp_grad_scaler.scale(loss).backward()
        amp_grad_scaler.step(self.optimizer)
        amp_grad_scaler.update()

        return loss_dict_iter

    def update_epoch(self, epoch):
        self.epoch = epoch
        return self.criterion.update_alpha(epoch)

    def validate(self, val_loader, test_loader, writer):
        val_auc = -1
        test_auc = -1
        if self.epoch <= self.cfg.EPOCHS:
            val_auc, val_loss = algorithm_validate_distill(self, val_loader, writer, self.epoch, 'val')
            test_auc, test_loss = algorithm_validate_distill(self, test_loader, writer, self.epoch, 'test')
            if self.epoch == self.cfg.EPOCHS:
                self.epoch += 1
        else:
            test_auc, test_loss = algorithm_validate_distill(self, test_loader, writer, self.cfg.EPOCHS + self.cfg.VAL_EPOCH, 'test')
            logging.info('Best performance on test domain(s): {}'.format(test_auc))

        return val_auc, test_auc

    def save_model(self, log_path):
        logging.info("Saving best model...")
        torch.save(self.network.state_dict(), os.path.join(log_path, 'best_model.pth'))
        torch.save(self.classifier.state_dict(), os.path.join(log_path, 'best_classifier.pth'))

    def renew_model(self, log_path):
        net_path = os.path.join(log_path, 'best_model.pth')
        classifier_path = os.path.join(log_path, 'best_classifier.pth')
        self.network.load_state_dict(torch.load(net_path))
        self.classifier.load_state_dict(torch.load(classifier_path))

    def predict(self, x):
        features_list = self.network(x, distill=True)
        # features_list: [x2, x3, x4, final]
        output = self.classifier(features_list[-1])
        output1 = self.classifier1(features_list[0])
        output2 = self.classifier2(features_list[1])
        output3 = self.classifier3(features_list[2])
        return [output1, output2, output3, output]

# Helper classes for other algorithms can remain or be added if needed
class CABNet(ERM):
    def __init__(self, num_classes, cfg):
        super(CABNet, self).__init__(num_classes, cfg)

class MixStyleNet(ERM):
    def __init__(self, num_classes, cfg):
        super(MixStyleNet, self).__init__(num_classes, cfg)

class MixupNet(ERM):
    def __init__(self, num_classes, cfg):
        super(MixupNet, self).__init__(num_classes, cfg)
        self.criterion_CE = torch.nn.CrossEntropyLoss()

    def update(self, minibatch, env_feats=None):
        image, mask, label, domain = minibatch
        self.optimizer.zero_grad()
        inputs, targets_a, targets_b, lam = self.mixup_data(image, label)
        outputs = self.predict(inputs)
        loss = self.mixup_criterion(self.criterion_CE, outputs, targets_a, targets_b, lam)
        loss.backward()
        self.optimizer.step()
        return {'loss':loss}

    def mixup_data(self, x, y, alpha=1.0, use_cuda=True):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        batch_size = x.size()[0]
        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class Fishr(ERM):
    def __init__(self, num_classes, cfg):
        super(Fishr, self).__init__(num_classes, cfg)
        self.num_groups = cfg.FISHR.NUM_GROUPS
        self.network = models.get_net(cfg)
        self.classifier = extend(
            models.get_classifier(self.network.out_features(), cfg)
        )
        self.optimizer = None
        self.register_buffer("update_count", torch.tensor([0]))
        self.bce_extended = extend(nn.CrossEntropyLoss(reduction='none'))
        self.ema_per_domain = [
            misc.MovingAverage(cfg.FISHR.EMA, oneminusema_correction=True)
            for _ in range(self.num_groups)
        ]
        self._init_optimizer()

    def _init_optimizer(self):
        self.optimizer = torch.optim.SGD(
            list(self.network.parameters()) + list(self.classifier.parameters()),
            lr = self.cfg.LEARNING_RATE,
            momentum = self.cfg.MOMENTUM,
            weight_decay = self.cfg.WEIGHT_DECAY,
            nesterov=True)

    def update(self, minibatch):
        image, mask, label, domain = minibatch
        all_x = image
        all_y = label
        len_minibatches = [image.shape[0]]
        all_z = self.network(all_x)
        all_logits = self.classifier(all_z)
        penalty = self.compute_fishr_penalty(all_logits, all_y, len_minibatches)
        all_nll = F.cross_entropy(all_logits, all_y)
        penalty_weight = 0
        if self.update_count >= self.cfg.FISHR.PENALTY_ANNEAL_ITERS:
            penalty_weight = self.cfg.FISHR.LAMBDA
            if self.update_count == self.cfg.FISHR.PENALTY_ANNEAL_ITERS != 0:
                self._init_optimizer()
        self.update_count += 1
        objective = all_nll + penalty_weight * penalty
        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()
        return {'loss': objective.item(), 'nll': all_nll.item(), 'penalty': penalty.item()}

    def compute_fishr_penalty(self, all_logits, all_y, len_minibatches):
        dict_grads = self._get_grads(all_logits, all_y)
        grads_var_per_domain = self._get_grads_var_per_domain(dict_grads, len_minibatches)
        return self._compute_distance_grads_var(grads_var_per_domain)

    def _get_grads(self, logits, y):
        self.optimizer.zero_grad()
        loss = self.bce_extended(logits, y).sum()
        with backpack(BatchGrad()):
            loss.backward(inputs=list(self.classifier.parameters()), retain_graph=True, create_graph=True)
        dict_grads = OrderedDict(
            [
                (name, weights.grad_batch.clone().view(weights.grad_batch.size(0), -1))
                for name, weights in self.classifier.named_parameters()
            ]
        )
        return dict_grads

    def _get_grads_var_per_domain(self, dict_grads, len_minibatches):
        grads_var_per_domain = [{} for _ in range(self.num_groups)]
        for name, _grads in dict_grads.items():
            all_idx = 0
            for domain_id, bsize in enumerate(len_minibatches):
                env_grads = _grads[all_idx:all_idx + bsize]
                all_idx += bsize
                env_mean = env_grads.mean(dim=0, keepdim=True)
                env_grads_centered = env_grads - env_mean
                grads_var_per_domain[domain_id][name] = (env_grads_centered).pow(2).mean(dim=0)
        for domain_id in range(self.num_groups):
            grads_var_per_domain[domain_id] = self.ema_per_domain[domain_id].update(
                grads_var_per_domain[domain_id]
            )
        return grads_var_per_domain

    def _compute_distance_grads_var(self, grads_var_per_domain):
        grads_var = OrderedDict(
            [
                (
                    name,
                    torch.stack(
                        [
                            grads_var_per_domain[domain_id][name]
                            for domain_id in range(self.num_groups)
                        ],
                        dim=0
                    ).mean(dim=0)
                )
                for name in grads_var_per_domain[0].keys()
            ]
        )
        penalty = 0
        for domain_id in range(self.num_groups):
            penalty += self.l2_between_dicts(grads_var_per_domain[domain_id], grads_var)
        return penalty / self.num_groups

    def l2_between_dicts(self, dict_1, dict_2):
        assert len(dict_1) == len(dict_2)
        dict_1_values = [dict_1[key] for key in sorted(dict_1.keys())]
        dict_2_values = [dict_2[key] for key in sorted(dict_1.keys())]
        return (
            torch.cat(tuple([t.view(-1) for t in dict_1_values])) -
            torch.cat(tuple([t.view(-1) for t in dict_2_values]))
        ).pow(2).mean()

class DRGen(Algorithm):
    def __init__(self, num_classes, cfg):
        super(DRGen, self).__init__(num_classes, cfg)
        algorithm_class = get_algorithm_class('Fishr')
        self.algorithm = algorithm_class(cfg.DATASET.NUM_CLASSES, cfg)
        self.optimizer = self.algorithm.optimizer
        self.swad_algorithm = AveragedModel(self.algorithm)
        self.swad_algorithm.cuda()
        self.swad = LossValley(None, cfg.DRGEN.N_CONVERGENCE, cfg.DRGEN.N_TOLERANCE, cfg.DRGEN.TOLERANCE_RATIO)

    def update(self, minibatch):
        loss_dict_iter = self.algorithm.update(minibatch)
        if self.swad:
            self.swad_algorithm.update_parameters(self.algorithm, step = self.epoch)
        return loss_dict_iter

    def validate(self, val_loader, test_loader, writer):
        swad_val_auc = -1
        swad_auc = -1
        if self.epoch <= self.cfg.EPOCHS:
            val_auc, val_loss = algorithm_validate(self.algorithm, val_loader, writer, self.epoch, 'val(Fishr)')
            test_auc, test_loss = algorithm_validate(self.algorithm, test_loader, writer, self.epoch, 'test(Fishr)')

            if self.swad:
                def prt_results_fn(results):
                    print(results)

                self.swad.update_and_evaluate(
                    self.swad_algorithm, val_auc, val_loss, prt_results_fn
                )

                if self.epoch != self.cfg.EPOCHS:
                    self.swad_algorithm = self.swad.get_final_model()
                    swad_val_auc, swad_val_loss = algorithm_validate(self.swad_algorithm, val_loader, writer, self.epoch, 'val')
                    swad_auc, swad_loss = algorithm_validate(self.swad_algorithm, test_loader, writer, self.epoch, 'test')

                    if hasattr(self.swad, "dead_valley") and self.swad.dead_valley:
                        logging.info("SWAD valley is dead -> not stop !")

                    self.swad_algorithm = AveragedModel(self.algorithm)

            if self.epoch == self.cfg.EPOCHS:
                self.epoch += 1
        else:
            self.swad_algorithm = self.swad.get_final_model()
            logging.warning("Evaluate SWAD ...")
            swad_auc, swad_loss = algorithm_validate(self.swad_algorithm, test_loader, writer, self.cfg.EPOCHS + self.cfg.VAL_EPOCH , 'test')
            logging.info('(last) swad test auc: {}  loss: {}'.format(swad_auc,swad_loss))
            
        return swad_val_auc, swad_auc    
        
    def save_model(self, log_path):
        self.algorithm.save_model(log_path)
    
    def renew_model(self, log_path):
        self.algorithm.renew_model(log_path)
    
    def predict(self, x):
        return self.swad_algorithm.predict(x)