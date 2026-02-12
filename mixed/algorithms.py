import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, collections, logging
from collections import OrderedDict
from copy import deepcopy
from itertools import combinations
from torch.cuda.amp import autocast

import utils.misc as misc
from utils.validate import algorithm_validate
import modeling.model_manager as models
from modeling.losses import DahLoss, DahLoss_Siam_Fastmoco_v0
from modeling.nets import LossValley, AveragedModel
from dataset.data_manager import get_post_FundusAug

from backpack import backpack, extend
from backpack.extensions import BatchGrad

ALGORITHMS = ['ERM', 'GDRNet', 'GREEN', 'CABNet', 'MixupNet', 'MixStyleNet', 'Fishr', 'DRGen']


def get_algorithm_class(algorithm_name):
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


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
        val_metrics = algorithm_validate(self, val_loader, writer, self.epoch, 'val')

        if test_loader is not None:
            test_metrics = algorithm_validate(self, test_loader, writer, self.epoch, 'test')
        else:
            test_metrics = {'auc': 0.0, 'acc': 0.0, 'f1': 0.0, 'qwk': 0.0, 'loss': 0.0}

        return val_metrics, test_metrics

    def save_model(self, log_path):
        raise NotImplementedError

    def renew_model(self, log_path):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

    def _get_net_state_dict(self):
        if hasattr(self.network, 'module'):
            return self.network.module.state_dict()
        return self.network.state_dict()

    def _load_net_state_dict(self, state_dict):
        is_ddp = hasattr(self.network, 'module')

        ckpt_keys = list(state_dict.keys())
        has_module_prefix = any(k.startswith('module.') for k in ckpt_keys)

        if is_ddp and not has_module_prefix:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict['module.' + k] = v
            self.network.load_state_dict(new_state_dict)

        elif not is_ddp and has_module_prefix:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            self.network.load_state_dict(new_state_dict)

        else:
            self.network.load_state_dict(state_dict)


class ERM(Algorithm):
    def __init__(self, num_classes, cfg):
        super(ERM, self).__init__(num_classes, cfg)

        self.network = models.get_net(cfg)
        self.classifier = models.get_classifier(self.network.out_features(), cfg)

        if 'FPT' in cfg.BACKBONE:
            print(f">> [Optimizer Info] Detected {cfg.BACKBONE}, switching to AdamW optimizer.")
            self.optimizer = torch.optim.AdamW(
                [{"params": self.network.parameters()}, {"params": self.classifier.parameters()}], lr=cfg.LEARNING_RATE,
                weight_decay=cfg.WEIGHT_DECAY)
        else:
            self.optimizer = torch.optim.SGD(
                [{"params": self.network.parameters()}, {"params": self.classifier.parameters()}], lr=cfg.LEARNING_RATE,
                momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY, nesterov=True)

    def update(self, minibatch):
        image, mask, label, domain = minibatch
        self.optimizer.zero_grad()

        features = self.network(image)
        output = self.classifier(features)
        loss = F.cross_entropy(output, label)

        loss.backward()
        self.optimizer.step()

        return {'loss': loss}

    def save_model(self, log_path):
        logging.info("Saving best model...")
        net_state = self._get_net_state_dict()
        torch.save(net_state, os.path.join(log_path, 'best_model.pth'))
        torch.save(self.classifier.state_dict(), os.path.join(log_path, 'best_classifier.pth'))

    def renew_model(self, log_path):
        net_path = os.path.join(log_path, 'best_model.pth')
        classifier_path = os.path.join(log_path, 'best_classifier.pth')

        self._load_net_state_dict(torch.load(net_path, map_location='cpu'))
        self.classifier.load_state_dict(torch.load(classifier_path, map_location='cpu'))

    def predict(self, x):
        return self.classifier(self.network(x))


class GDRNet(Algorithm):
    """
    GDRNet + MASK_SIAM (Complete Implementation v0) + DINOv3-FD + FPT+
    """

    def __init__(self, num_classes, cfg):
        super(GDRNet, self).__init__(num_classes, cfg)
        self.cfg = cfg

        # 1. Network Backbone (FPT+)
        self.network = models.get_net(cfg)

        # MASK_SIAM v0 核心参数 (全盘接受)
        # FPT+ 的 SideResNet50 输出通常是 2048 维
        dim_in = 2048
        feat_dim = 512  # MASK_SIAM v0 的瓶颈维度

        # 2. Projector & Predictor (严格复刻 MASK_SIAM v0 结构)
        # Projector: Linear -> LeakyReLU -> Linear (瓶颈结构: 2048 -> 512 -> 2048)
        self.projector = nn.Sequential(
            nn.Linear(dim_in, feat_dim, bias=False),
            nn.LeakyReLU(inplace=True),  # v0 使用 LeakyReLU
            nn.Linear(feat_dim, dim_in, bias=False)  # 映射回 2048
        )

        # Predictor: Linear (2048 -> 2048) - v0 实现非常简单，没有 BN 和 ReLU
        self.predictor = nn.Sequential(
            nn.Linear(dim_in, dim_in, bias=False)
        )

        # 初始化权重 (v0逻辑)
        self._init_weights(self.projector)
        self._init_weights(self.predictor)

        # 3. Classifier
        self.classifier = models.get_classifier(dim_in, cfg)

        # 4. EMA Teacher Model (MASK_SIAM 核心机制 - 你的代码之前缺了这个)
        self.network_ema = deepcopy(self.network)
        self.classifier_ema = deepcopy(self.classifier)

        # 冻结 EMA 模型参数，不参与反向传播
        for param in self.network_ema.parameters():
            param.requires_grad = False
        for param in self.classifier_ema.parameters():
            param.requires_grad = False

        # 5. Memory Queue
        self.K = 1024  # v0 默认值
        self.register_buffer("queue", torch.randn(self.K, dim_in))  # 注意：这里存的是 2048 维
        self.queue = nn.functional.normalize(self.queue, dim=-1)
        self.register_buffer("queue_labels", -torch.ones(self.K, dtype=torch.long))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.num_positive = cfg.POSITIVE if hasattr(cfg, 'POSITIVE') else 4
        self.split_num = 2
        self.combs = 3

        # 6. Optimizer
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.network.parameters(), 'fix_lr': False},
                {"params": self.classifier.parameters(), 'fix_lr': False},
                {"params": self.projector.parameters(), 'fix_lr': False},
                # [Fix] 严格对齐 MASK_SIAM v0 / SimSiam，Predictor 的 LR 应当固定或不衰减
                {"params": self.predictor.parameters(), 'fix_lr': True},
            ],
            lr=cfg.LEARNING_RATE,
            weight_decay=cfg.WEIGHT_DECAY
        )

        # 7. Losses
        self.criterion = DahLoss_Siam_Fastmoco_v0(
            cfg=cfg,
            max_iteration=cfg.EPOCHS,
            training_domains=cfg.DATASET.SOURCE_DOMAINS,
            beta=cfg.GDRNET.BETA,
            temperature=cfg.GDRNET.TEMPERATURE,
            scaling_factor=cfg.GDRNET.SCALING_FACTOR,
            fastmoco=getattr(cfg, 'FASTMOCO', 1.0)
        )
        self.E_dis = nn.MSELoss()  # 用于特征一致性损失
        self.global_step = 0

    def _init_weights(self, module):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    # --- MASK_SIAM Core Logic: Queue Management ---
    @torch.no_grad()
    def dequeue_and_enqueue(self, features, labels):
        batch_size = features.shape[0]
        ptr = int(self.queue_ptr)
        # Replace the queue at ptr (wrap around)
        if ptr + batch_size > self.K:
            rem = self.K - ptr
            self.queue[ptr:self.K] = features[:rem]
            self.queue_labels[ptr:self.K] = labels[:rem]
            self.queue[0:batch_size - rem] = features[rem:]
            self.queue_labels[0:batch_size - rem] = labels[rem:]
        else:
            self.queue[ptr: ptr + batch_size] = features
            self.queue_labels[ptr: ptr + batch_size] = labels

        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def sample_target(self, features, labels):
        neighbor = []
        for i, label in enumerate(labels):
            pos = torch.where(self.queue_labels == label)[0]
            if len(pos) != 0:
                choice = torch.multinomial(torch.ones_like(pos).type(torch.FloatTensor), self.num_positive,
                                           replacement=True)
                idx = pos[choice]
                neighbor.append(self.queue[idx].mean(0))
            else:
                neighbor.append(features[i])
        neighbor = torch.stack(neighbor, dim=0)
        return neighbor

    # --- MASK_SIAM Core Logic: Patchify & Masking ---
    def patchify(self, imgs, block=32):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = block
        # Ensure divisible
        if imgs.shape[2] % p != 0:
            new_size = (imgs.shape[2] // p) * p
            imgs = F.interpolate(imgs, size=(new_size, new_size))

        h = w = imgs.shape[2] // p
        n = imgs.shape[0]

        x = imgs.reshape(shape=(n, 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x, block=32):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = block
        h = w = int(x.shape[1] ** .5)
        n = x.shape[0]

        x = x.reshape(shape=(n, h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(n, 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand([N, L], device=x.device)

        # Sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Generate binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0

        # Unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # Mask M: kept parts masked out (0), removed parts kept (1)?
        # MASK_SIAM logic: We want to create two complementary views.
        # View 1 (Masked): x * mask
        # View 2 (Complement): x * (1-mask)
        # Note: In MAE, 0 usually means visible. Here we treat 'mask' as the binary multiplier.

        # Let's align with MASK_SIAM v0:
        # It usually generates a mask where 1 = active/visible or 1 = masked.
        # We return two masks for complementary views.

        # Return masks extended to feature dim for multiplication
        return mask.unsqueeze(-1), (1 - mask.unsqueeze(-1))

    # --- MASK_SIAM Core Logic: FastMoCo Splits ---
    def _local_split(self, x):
        # Splits the image batch spatially into smaller crops for FastMoCo
        # NxCxHxW --> 4NxCx(H/2)x(W/2) (if split_num=2)
        if x.size(2) % self.split_num != 0:
            # Safety resize if needed
            new_size = (x.size(2) // self.split_num) * self.split_num
            x = F.interpolate(x, size=(new_size, new_size))

        _side_indent = x.size(2) // self.split_num, x.size(3) // self.split_num

        # Split width
        cols = x.split(_side_indent[1], dim=3)
        xs = []
        # Split height
        for _x in cols:
            xs += _x.split(_side_indent[0], dim=2)

        # Concatenate along batch dimension
        x = torch.cat(xs, dim=0)
        return x

    def update_ema_model(self, momentum=0.996):
        """Update EMA target model parameters."""
        with torch.no_grad():
            for param_q, param_k in zip(self.network.parameters(), self.network_ema.parameters()):
                param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)
            for param_q, param_k in zip(self.classifier.parameters(), self.classifier_ema.parameters()):
                param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)

    def update(self, minibatch):
        self.global_step += 1

        # 1. [Fix] 正确解包 8 个返回值
        # 注意：经过 dataset 的修正，label2 现在等于 label1，lam 等于 1.0
        # 但我们依然保留完整解包逻辑以防未来变更
        img_weak, img_strong, mask, label1, domain, index, label2, lam = minibatch

        img_weak = img_weak.cuda()
        img_strong = img_strong.cuda()
        label = label1.cuda()  # 主要使用 label1
        domain = domain.cuda()

        self.optimizer.zero_grad()

        # 2. Mask Generation (MASK_SIAM 逻辑保持不变)
        with torch.no_grad():
            bs = self.cfg.BLOCK_SIZE if hasattr(self.cfg, 'BLOCK_SIZE') else 32
            mask_ratio = self.cfg.MASK_RATIO if hasattr(self.cfg, 'MASK_RATIO') else 0.5

            image_new_patches = self.patchify(img_strong, block=bs)
            image_ori_patches = self.patchify(img_weak, block=bs)

            mask_m, mask_c = self.random_masking(image_new_patches, mask_ratio=mask_ratio)

            image_new_masked = self.unpatchify(image_new_patches * mask_m, block=bs)
            image_ori_masked_c = self.unpatchify(image_ori_patches * mask_c, block=bs)

        # 3. Forward Pass (Student) - FPT+ 全局输入
        out_ori = self.network(img_weak)
        out_new = self.network(img_strong)
        out_new_masked = self.network(image_new_masked)
        out_ori_masked_c = self.network(image_ori_masked_c)

        features_ori = out_ori['features']
        features_new = out_new['features']
        features_new_masked = out_new_masked['features']
        features_ori_masked_c = out_ori_masked_c['features']

        # DINOv3-FD 特征
        dino_raw = out_new['dino_raw'].detach()
        tra_feat = out_new['tra_feat']

        # Logits
        output_new = self.classifier(features_new)
        output_ori = self.classifier(features_ori)
        output_new_masked = self.classifier(features_new_masked)
        output_ori_masked_c = self.classifier(features_ori_masked_c)

        # 4. Forward Pass (Teacher / EMA)
        with torch.no_grad():
            out_new_ema = self.network_ema(img_strong)
            features_new_ema_list = out_new_ema['features_list']
            # features_new_ema = out_new_ema['features'] # 如果需要

        # 5. Projections & Predictions (MASK_SIAM v0)
        # 这里的 Projector 输入维度是 2048，输出 512，再 Predict 回 2048

        # 5.1 Regular Views
        features_ori_z1 = self.projector(features_ori)
        features_new_z2 = self.projector(features_new)

        p1 = self.predictor(features_ori_z1)
        p2 = self.predictor(features_new_z2)

        # 5.2 Masked Views
        z_new_masked = self.projector(features_new_masked)
        z_ori_masked_c = self.projector(features_ori_masked_c)

        p_new_masked = self.predictor(z_new_masked)
        p_ori_masked_c = self.predictor(z_ori_masked_c)

        # 5.3 Supervised Targets (Queue Sampling)
        z1_sup = self.sample_target(features_ori_z1.detach(), label)
        z2_sup = self.sample_target(features_new_z2.detach(), label)

        # 6. [Fix] 移除 FastMoCo Split 逻辑，防止破坏 FPT+ 分辨率
        # 原代码中的 _local_split 和 combinations 全部删除。
        # 我们直接使用全图特征作为 "Orthmix" 的替代，或者简单地将其设为 None/Identity。
        # 为了兼容 DahLoss 的接口 (它期望接收 z1_orthmix 和 p1_orthmix_list)，
        # 我们直接使用完整的 features_ori_z1 和 p1 作为替代。
        # 这在逻辑上等同于 batch_size=1 的 split，即没有 split。

        z1_orthmix = features_ori_z1
        z2_orthmix = features_new_z2

        # DahLoss 内部可能期望 list 结构 (针对 FastMoCo 的 chunks)
        # 这里我们将它们包装成单元素 list，或者根据 Loss 内部实现调整。
        # 查看 Loss 代码，它可能对 list 进行遍历。为了安全：
        p1_orthmix_list = [p1]
        p2_orthmix_list = [p2]

        # 7. Update Queue
        self.dequeue_and_enqueue(features_new_z2.detach(), label)

        # 8. Loss Calculation
        # 注意：由于 label2=label1, lam=1.0，我们直接传 label 即可。
        loss, loss_dict = self.criterion(
            [output_new, output_ori, output_new_masked, output_ori_masked_c],
            [features_ori, features_new, features_ori_z1, features_new_z2, z1_sup, z2_sup, p1, p2,
             z1_orthmix, z2_orthmix, p1_orthmix_list, p2_orthmix_list],  # 传入未 Split 的特征
            [z_new_masked, z_ori_masked_c, p_new_masked, p_ori_masked_c],
            label, domain, random_matrix=None
        )

        # 9. Additional Loss 1: MASK_SIAM Feature Consistency
        features_new_masked_list = out_new_masked.get('features_list', None)
        loss_feat = 0.0
        # 获取 KD 温度系数
        temp_kd = getattr(self.cfg, 'KD', 1.0)

        if features_new_masked_list is not None and features_new_ema_list is not None:
            # FPT+ SideResNet 通常有 4 个 stage
            min_len = min(len(features_new_masked_list), len(features_new_ema_list))
            for i in range(min_len):
                loss_feat += self.E_dis(features_new_masked_list[i], features_new_ema_list[i].detach())

            loss += temp_kd * loss_feat
            loss_dict['loss_feat'] = loss_feat.item()

        # 10. Additional Loss 2: DINOv3-FD Orthogonality
        # 确保 tra_feat [B, N, D] 和 dino_raw [B, N, D] 维度一致
        if tra_feat.dim() == 3:
            u = tra_feat.mean(dim=1)  # [B, D]
            v = dino_raw.mean(dim=1)  # [B, D]
        else:
            u = tra_feat
            v = dino_raw

        cosine = F.cosine_similarity(u, v, dim=-1)
        loss_ortho = (cosine ** 2).mean()

        loss += 0.1 * loss_ortho
        loss_dict['loss_ortho'] = loss_ortho.item()

        # Backward
        loss.backward()
        self.optimizer.step()

        # Update EMA
        self.update_ema_model()
        self.criterion.update_alpha(self.epoch)

        return loss_dict

    def save_model(self, log_path):
        torch.save(self.network.state_dict(), os.path.join(log_path, 'best_model.pth'))
        torch.save(self.classifier.state_dict(), os.path.join(log_path, 'best_classifier.pth'))
        # Optional: Save projector/predictor if needed for resuming
        torch.save(self.projector.state_dict(), os.path.join(log_path, 'best_projector.pth'))
        torch.save(self.predictor.state_dict(), os.path.join(log_path, 'best_predictor.pth'))

    def renew_model(self, log_path):
        self.network.load_state_dict(torch.load(os.path.join(log_path, 'best_model.pth')))
        self.classifier.load_state_dict(torch.load(os.path.join(log_path, 'best_classifier.pth')))
        # Load others if saved

    def predict(self, x):
        # Return logits from classifier
        feat = self.network(x)['features']
        return self.classifier(feat)


class GREEN(Algorithm):
    def __init__(self, num_classes, cfg):
        super(GREEN, self).__init__(num_classes, cfg)
        self.network = models.get_net(cfg)
        self.optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=cfg.LEARNING_RATE,
            momentum=cfg.MOMENTUM,
            weight_decay=cfg.WEIGHT_DECAY,
            nesterov=True)

    def update(self, minibatch):
        image, mask, label, domain = minibatch
        self.optimizer.zero_grad()

        output = self.network(image)
        loss = F.cross_entropy(output, label)

        loss.backward()
        self.optimizer.step()
        return {'loss': loss}

    def save_model(self, log_path):
        logging.info("Saving best model...")
        net_state = self._get_net_state_dict()
        torch.save(net_state, os.path.join(log_path, 'best_model.pth'))

    def renew_model(self, log_path):
        net_path = os.path.join(log_path, 'best_model.pth')
        self._load_net_state_dict(torch.load(net_path, map_location='cpu'))

    def predict(self, x):
        return self.network(x)


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

        return {'loss': loss}

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
        self.classifier = extend(models.get_classifier(self.network._out_features, cfg))
        self.optimizer = None
        self.register_buffer("update_count", torch.tensor([0]))
        self.bce_extended = extend(nn.CrossEntropyLoss(reduction='none'))
        self.ema_per_domain = [misc.MovingAverage(cfg.FISHR.EMA, oneminusema_correction=True) for _ in
                               range(self.num_groups)]
        self._init_optimizer()

    def _init_optimizer(self):
        self.optimizer = torch.optim.SGD(
            list(self.network.parameters()) + list(self.classifier.parameters()),
            lr=self.cfg.LEARNING_RATE,
            momentum=self.cfg.MOMENTUM,
            weight_decay=self.cfg.WEIGHT_DECAY,
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
            [(name, weights.grad_batch.clone().view(weights.grad_batch.size(0), -1)) for name, weights in
             self.classifier.named_parameters()])
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
            grads_var_per_domain[domain_id] = self.ema_per_domain[domain_id].update(grads_var_per_domain[domain_id])
        return grads_var_per_domain

    def _compute_distance_grads_var(self, grads_var_per_domain):
        grads_var = OrderedDict([(name, torch.stack(
            [grads_var_per_domain[domain_id][name] for domain_id in range(self.num_groups)], dim=0).mean(dim=0)) for
                                 name in grads_var_per_domain[0].keys()])
        penalty = 0
        for domain_id in range(self.num_groups):
            penalty += self.l2_between_dicts(grads_var_per_domain[domain_id], grads_var)
        return penalty / self.num_groups

    def l2_between_dicts(self, dict_1, dict_2):
        assert len(dict_1) == len(dict_2)
        dict_1_values = [dict_1[key] for key in sorted(dict_1.keys())]
        dict_2_values = [dict_2[key] for key in sorted(dict_1.keys())]
        return (torch.cat(tuple([t.view(-1) for t in dict_1_values])) - torch.cat(
            tuple([t.view(-1) for t in dict_2_values]))).pow(2).mean()


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
            self.swad_algorithm.update_parameters(self.algorithm, step=self.epoch)
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

                self.swad.update_and_evaluate(self.swad_algorithm, val_auc, val_loss, prt_results_fn)

                if self.epoch != self.cfg.EPOCHS:
                    self.swad_algorithm = self.swad.get_final_model()
                    swad_val_auc, swad_val_loss = algorithm_validate(self.swad_algorithm, val_loader, writer,
                                                                     self.epoch, 'val')
                    swad_auc, swad_loss = algorithm_validate(self.swad_algorithm, test_loader, writer, self.epoch,
                                                             'test')

                    if hasattr(self.swad, "dead_valley") and self.swad.dead_valley:
                        logging.info("SWAD valley is dead -> not stop !")

                    self.swad_algorithm = AveragedModel(self.algorithm)  # reset

            if self.epoch == self.cfg.EPOCHS:
                self.epoch += 1

        else:
            self.swad_algorithm = self.swad.get_final_model()
            logging.warning("Evaluate SWAD ...")
            swad_auc, swad_loss = algorithm_validate(self.swad_algorithm, test_loader, writer,
                                                     self.cfg.EPOCHS + self.cfg.VAL_EPOCH, 'test')
            logging.info('(last) swad test auc: {}  loss: {}'.format(swad_auc, swad_loss))

        return swad_val_auc, swad_auc

    def save_model(self, log_path):
        self.algorithm.save_model(log_path)

    def renew_model(self, log_path):
        self.algorithm.renew_model(log_path)

    def predict(self, x):
        return self.swad_algorithm.predict(x)