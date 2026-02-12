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
    GDRNet + MASK_SIAM (Complete Implementation)
    Integrates FPT+ Backbone with MASK_SIAM Logic (Masking, FastMoCo, Consistency).
    """

    def __init__(self, num_classes, cfg):
        super(GDRNet, self).__init__(num_classes, cfg)
        self.cfg = cfg

        # 1. Network Backbone (FPT+)
        self.network = models.get_net(cfg)

        # Get output feature dimension from FPT+
        # Note: FPT+ typically returns a dict, but out_features() should give the dim of 'features'
        dim_in = self.network.out_features()
        feat_dim = 512  # Fixed feature dimension for projection space as per MASK_SIAM v0

        # 2. Projector & Predictor (MASK_SIAM Architecture)
        # Projector: Linear -> BN -> ReLU -> Linear
        self.projector = nn.Sequential(
            nn.Linear(dim_in, dim_in, bias=False),
            nn.BatchNorm1d(dim_in),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in, feat_dim, bias=False)
        )

        # Predictor: Linear -> BN -> ReLU -> Linear
        self.predictor = nn.Sequential(
            nn.Linear(feat_dim, feat_dim, bias=False),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim, bias=False)
        )

        # 3. Independent Classifier (For MASK_SIAM logic separation)
        # Although FPT+ might have a head, we define one explicitly to control the flow
        self.classifier = models.get_classifier(dim_in, cfg)

        # 4. Memory Queue (MoCo Style)
        self.K = cfg.MOCO_QUEUE_K if hasattr(cfg, 'MOCO_QUEUE_K') else 1024
        self.register_buffer("queue", torch.randn(self.K, feat_dim))
        self.queue = nn.functional.normalize(self.queue, dim=-1)
        self.register_buffer("queue_labels", -torch.ones(self.K, dtype=torch.long))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # MASK_SIAM specific params
        self.num_positive = cfg.POSITIVE if hasattr(cfg, 'POSITIVE') else 4
        self.split_num = 2  # For FastMoCo split
        self.combs = 3  # For FastMoCo combinations

        # 5. Optimizer (Restored to Adam as requested)
        # Predictor usually requires higher LR or specific handling in Siamese networks
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.network.parameters(), 'fix_lr': False},
                {"params": self.classifier.parameters(), 'fix_lr': False},
                {"params": self.projector.parameters(), 'fix_lr': False},
                {"params": self.predictor.parameters(), 'fix_lr': False},
            ],
            lr=cfg.LEARNING_RATE,
            weight_decay=cfg.WEIGHT_DECAY
        )

        # 6. Loss Function (DahLoss_Siam_Fastmoco_v0)
        self.criterion = DahLoss_Siam_Fastmoco_v0(
            cfg=cfg,
            max_iteration=cfg.EPOCHS,  # Or total steps if available
            training_domains=cfg.DATASET.SOURCE_DOMAINS,
            beta=cfg.GDRNET.BETA,
            temperature=cfg.GDRNET.TEMPERATURE,
            scaling_factor=cfg.GDRNET.SCALING_FACTOR,
            fastmoco=getattr(cfg, 'FASTMOCO', 1.0)
        )

        self.global_step = 0

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

    def update(self, minibatch):
        self.global_step += 1

        # 1. Prepare Data
        # img_ori: Weak Augmentation
        # img_strong: Strong Augmentation (potentially with SPMix applied in DataLoader)
        img_weak = minibatch[0].cuda()
        img_strong = minibatch[1].cuda()
        label = minibatch[3].cuda()
        domain = minibatch[5].cuda() if len(minibatch) > 5 else None

        self.optimizer.zero_grad()

        # 2. MASK GENERATION (The Missing Core Mechanism)
        with torch.no_grad():
            # Patchify Strong Aug Image
            # Note: Using cfg.BLOCK_SIZE, default 32
            bs = self.cfg.BLOCK_SIZE if hasattr(self.cfg, 'BLOCK_SIZE') else 32
            mask_ratio = self.cfg.MASK_RATIO if hasattr(self.cfg, 'MASK_RATIO') else 0.5

            image_new_patches = self.patchify(img_strong, block=bs)
            image_ori_patches = self.patchify(img_weak, block=bs)

            # Generate Random Masks
            mask_m, mask_c = self.random_masking(image_new_patches, mask_ratio=mask_ratio)

            # Apply Masks
            # image_new_masked: Masked version of Strong Aug
            image_new_masked = self.unpatchify(image_new_patches * mask_m, block=bs)

            # image_ori_masked_c: Complementary Masked version of Weak Aug (or Strong, depending on v0 logic)
            # v0 usually masks 'ori' with the complement of 'new' mask to enforce cross-view consistency
            image_ori_masked_c = self.unpatchify(image_ori_patches * mask_c, block=bs)

        # 3. FORWARD PASS
        # We need output dictionaries from FPT+

        # 3.1 Standard Views
        out_ori = self.network(img_weak)
        features_ori = out_ori['features']  # (B, dim_in)
        output_ori = self.classifier(features_ori)

        out_new = self.network(img_strong)
        features_new = out_new['features']
        output_new = self.classifier(features_new)

        # 3.2 Masked Views
        out_new_masked = self.network(image_new_masked)
        features_new_masked = out_new_masked['features']
        output_new_masked = self.classifier(features_new_masked)

        out_ori_masked_c = self.network(image_ori_masked_c)
        features_ori_masked_c = out_ori_masked_c['features']
        output_ori_masked_c = self.classifier(features_ori_masked_c)

        # 4. PROJECTION & PREDICTION
        # Siamese Logic: z = Projector(h), p = Predictor(z)

        features_ori_z1 = self.projector(features_ori)
        features_new_z2 = self.projector(features_new)

        p1 = self.predictor(features_ori_z1)
        p2 = self.predictor(features_new_z2)

        # Normalize for internal calculations if needed, though loss handles it
        # MASK_SIAM v0 normalizes before passing to some logic
        features_ori_z1_norm = F.normalize(features_ori_z1, dim=-1)
        features_new_z2_norm = F.normalize(features_new_z2, dim=-1)
        p1_norm = F.normalize(p1, dim=-1)
        p2_norm = F.normalize(p2, dim=-1)

        # Masked Projections
        z_new_masked = self.projector(features_new_masked)
        z_ori_masked_c = self.projector(features_ori_masked_c)
        p_new_masked = self.predictor(z_new_masked)
        p_ori_masked_c = self.predictor(z_ori_masked_c)

        z_new_masked = F.normalize(z_new_masked, dim=-1)
        z_ori_masked_c = F.normalize(z_ori_masked_c, dim=-1)
        p_new_masked = F.normalize(p_new_masked, dim=-1)
        p_ori_masked_c = F.normalize(p_ori_masked_c, dim=-1)

        # 5. FASTMOCO LOGIC (Split & OrthMix)
        # This is the "S-Class" complexity part

        # Split images
        x1_split = self._local_split(img_weak)
        x2_split = self._local_split(img_strong)

        # Forward splits (B*4 batch size)
        # FPT+ handles variable size inputs usually, but we must ensure it's safe
        out_z1_pre = self.network(x1_split)['features']
        out_z2_pre = self.network(x2_split)['features']

        # Split features back into list of chunks
        # batch size increased by factor of split_num^2 (e.g. 4)
        chunks = self.split_num ** 2
        z1_splits = list(out_z1_pre.split(out_z1_pre.size(0) // chunks, dim=0))
        z2_splits = list(out_z2_pre.split(out_z2_pre.size(0) // chunks, dim=0))

        # Orthogonal Mix (OrthMix)
        # Combinations of splits averaged
        z1_orthmix_list = list(map(lambda x: sum(x) / self.combs, list(combinations(z1_splits, r=self.combs))))
        z2_orthmix_list = list(map(lambda x: sum(x) / self.combs, list(combinations(z2_splits, r=self.combs))))

        z1_orthmix_ = torch.cat(z1_orthmix_list, dim=0)
        z2_orthmix_ = torch.cat(z2_orthmix_list, dim=0)

        # Project & Predict Mixed Features
        z1_orthmix_ = self.projector(z1_orthmix_)
        z2_orthmix_ = self.projector(z2_orthmix_)
        p1_orthmix_ = self.predictor(z1_orthmix_)
        p2_orthmix_ = self.predictor(z2_orthmix_)

        # Normalize
        z1_orthmix = F.normalize(z1_orthmix_, dim=-1)
        z2_orthmix = F.normalize(z2_orthmix_, dim=-1)
        p1_orthmix = F.normalize(p1_orthmix_, dim=-1)
        p2_orthmix = F.normalize(p2_orthmix_, dim=-1)

        # Since cat produced a large batch, we split it back to list for Loss compatibility
        # The Loss expects a list of tensors for orthmix if multiple combinations exist
        # Or we pass the concatenated tensor and handle it in Loss (Loss v0 handles tensor)
        # However, to be safe with `DahLoss_Siam_Fastmoco_v0` which iterates:
        # "for i in range(len(p1_orthmix))" -> implies p1_orthmix should be a list or iterable of tensors
        num_combs = len(list(combinations(range(chunks), self.combs)))
        p1_orthmix_list = list(p1_orthmix.chunk(num_combs, dim=0))
        p2_orthmix_list = list(p2_orthmix.chunk(num_combs, dim=0))

        # 6. MEMORY QUEUE SAMPLING
        z1_sup = self.sample_target(features_ori_z1_norm.detach(), label)
        z2_sup = self.sample_target(features_new_z2_norm.detach(), label)

        # Update Queue
        self.dequeue_and_enqueue(features_new_z2_norm.detach(), label)

        # 7. LOSS CALCULATION
        # Pack everything for DahLoss_Siam_Fastmoco_v0

        logits_group = [output_new, output_ori, output_new_masked, output_ori_masked_c]

        # Note: The Loss expects specific order in features list:
        # features_ori, features_new, z1, z2, z1_sup, z2_sup, p1, p2, z1_orthmix, z2_orthmix, p1_orthmix, p2_orthmix
        feats_group = [
            features_ori, features_new,
            features_ori_z1_norm, features_new_z2_norm,
            z1_sup, z2_sup,
            p1_norm, p2_norm,
            z1_orthmix, z2_orthmix,  # Passed as full tensor or list? Loss v0 seems to use p1_orthmix for loop
            p1_orthmix_list, p2_orthmix_list  # Passing lists to match "range(len(p1_orthmix))" in Loss
        ]

        siam_group = [z_new_masked, z_ori_masked_c, p_new_masked, p_ori_masked_c]

        # DINOv3-FD (Optional): If FPT+ returns 'dino_raw' and 'tra_feat', we can add orthogonality loss here
        # But MASK_SIAM logic is primary. We let the backbone do its job.

        loss, loss_dict = self.criterion(logits_group, feats_group, siam_group, label, domain, random_matrix=None)

        loss.backward()
        self.optimizer.step()

        # Update alpha for loss scaling if needed
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