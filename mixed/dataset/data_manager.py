from .GDRBench import GDRBench
from . import fundusaug as FundusAug
from torchvision import transforms
from torch.utils.data import DataLoader, DistributedSampler, Dataset
import torch
import numpy as np
import cv2
from transformers import AutoImageProcessor
import os
import copy


def get_dataset(args, cfg):
    # 1. 获取基础变换和眼底增强变换
    if cfg.ALGORITHM != 'GDRNet':
        train_ts, test_ts, tra_fundus = get_transform(cfg)
    else:
        # GDRNet 模式下，tra_fundus 将作为 Strong Augmentation 的基础
        train_ts, test_ts, tra_fundus = get_pre_FundusAug(cfg)

    batch_size = cfg.BATCH_SIZE
    drop_last = getattr(cfg, 'DROP_LAST', True)
    num_worker = getattr(cfg, 'WORKERS', 4)

    # 2. 初始化原始数据集
    # trans_basic -> Weak Augmentation (img_ori)
    # trans_mask  -> Strong Augmentation Base (img_strong_base)
    train_dataset_base = GDRBench(
        root=cfg.DATASET.ROOT,
        source_domains=cfg.DATASET.SOURCE_DOMAINS,
        target_domains=cfg.DATASET.TARGET_DOMAINS,
        mode='train',
        trans_basic=train_ts,
        trans_mask=tra_fundus
    )

    # 3. [核心修改] 引入 SPMix 包装器
    # 如果是 GDRNet，我们需要对 Strong Augmentation 分支应用 SPMix 策略
    if cfg.ALGORITHM == 'GDRNet':
        print(">> [Data Manager] Wrapping Dataset with Saliency-Guided Patch-Based Mixup (SPMix).")
        # 传入 cfg 以获取可能的 grid 设置，默认 16x16
        train_dataset = SPMixDatasetWrapper(train_dataset_base, cfg)
    else:
        train_dataset = train_dataset_base

    # 4. 构建 DataLoader
    train_sampler = None
    train_shuffle = True
    if args.local_rank != -1:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        train_shuffle = False

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=train_shuffle, num_workers=num_worker,
                              drop_last=drop_last, pin_memory=True, sampler=train_sampler)

    val_dataset = GDRBench(root=cfg.DATASET.ROOT, source_domains=cfg.DATASET.SOURCE_DOMAINS,
                           target_domains=cfg.DATASET.TARGET_DOMAINS, mode='val', trans_basic=test_ts)

    val_sampler = None
    if args.local_rank != -1:
        val_sampler = DistributedSampler(val_dataset, shuffle=False)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker, pin_memory=True,
                            sampler=val_sampler)

    test_dataset = GDRBench(root=cfg.DATASET.ROOT, source_domains=cfg.DATASET.SOURCE_DOMAINS,
                            target_domains=cfg.DATASET.TARGET_DOMAINS, mode='test', trans_basic=test_ts)

    test_sampler = None
    if args.local_rank != -1:
        test_sampler = DistributedSampler(test_dataset, shuffle=False)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker,
                             pin_memory=True, sampler=test_sampler)

    dataset_size = [len(train_dataset), len(val_dataset), len(test_dataset)]

    return train_loader, val_loader, test_loader, dataset_size, train_sampler


class SPMixDatasetWrapper(Dataset):
    """
    SPMix 包装器：实现原汁原味的 Saliency-Guided Patch-Based Mixup。
    内置了基于 Spectral Residual 的显著性检测算法，无需额外模型权重即可运行。
    """

    def __init__(self, dataset, cfg):
        self.dataset = dataset
        self.cfg = cfg
        self.prob = 0.5  # SPMix 触发概率
        self.beta = 1.0  # Beta 分布参数
        self.grid_size = 16  # 将图片划分为 16x16 的网格

    def __len__(self):
        return len(self.dataset)

    def get_saliency_sr(self, img_tensor):
        """
        使用 Spectral Residual (SR) 算法计算显著性图。
        这是一种高效的、无监督的显著性检测方法，不需要预训练模型。
        """
        # Tensor (C, H, W) -> Numpy (H, W, C)
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        # 反归一化并转为 uint8 (假设输入经过了 Normalize)
        # 这里为了速度，简单缩放即可，SR 对绝对颜色不敏感
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        img_uint8 = (img_np * 255).astype(np.uint8)

        # 转灰度并缩放到小尺寸以加快 FFT 速度 (例如 64x64)
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        small_size = 64
        img_small = cv2.resize(gray, (small_size, small_size))

        # FFT
        f = np.fft.fft2(img_small)
        fshift = np.fft.fftshift(f)
        amp_spectrum = np.abs(fshift)
        log_amp = np.log(amp_spectrum + 1e-8)

        # Spectral Residual = Log Amplitude - Average Log Amplitude
        # 使用 Box Filter 模拟局部平均
        avg_log_amp = cv2.blur(log_amp, (3, 3))
        spectral_residual = log_amp - avg_log_amp

        # IFFT 重建显著性图
        phase_spectrum = np.angle(fshift)
        fshift_sr = np.exp(spectral_residual + 1j * phase_spectrum)
        f_sr = np.fft.ifftshift(fshift_sr)
        img_sr = np.abs(np.fft.ifft2(f_sr))

        # 后处理：平滑与归一化
        saliency_map = img_sr * img_sr
        saliency_map = cv2.GaussianBlur(saliency_map, (3, 3), 0)
        saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)

        # 恢复到原图大小
        H, W = img_tensor.shape[1], img_tensor.shape[2]
        saliency_map_full = cv2.resize(saliency_map, (W, H))

        return saliency_map_full

    def apply_spmix(self, img_source, img_target):
        """
        SPMix 核心逻辑：
        1. 计算 Source 的 Saliency Map。
        2. 将 Source 划分为 Patches。
        3. 根据 Saliency 选择 Top-K 个最重要的 Patches。
        4. 将这些 Patches 粘贴到 Target 的对应位置。
        """
        # 1. 获取 Source 的显著性图
        saliency_map = self.get_saliency_sr(img_source)

        # 2. 准备网格参数
        C, H, W = img_source.shape
        grid_h = H // self.grid_size
        grid_w = W // self.grid_size

        # 如果无法整除，简单 resize 处理
        if H % self.grid_size != 0 or W % self.grid_size != 0:
            return img_source  # 尺寸不匹配时跳过，防止报错

        # 3. 计算每个 grid 的显著性分数
        # 将 saliency_map view 成 (grid_size, grid_h, grid_size, grid_w)
        # 然后求和
        sal_tensor = torch.from_numpy(saliency_map)
        sal_patches = sal_tensor.view(self.grid_size, grid_h, self.grid_size, grid_w)
        patch_scores = sal_patches.sum(dim=(1, 3))  # (grid_size, grid_size)

        # 摊平分数以便排序
        flat_scores = patch_scores.view(-1)
        num_patches = flat_scores.shape[0]

        # 4. 确定要混合的 Patch 数量
        # 根据 Beta 分布采样混合比例 lambda
        lam = np.random.beta(self.beta, self.beta)
        num_mix = int(lam * num_patches)
        if num_mix == 0: num_mix = 1

        # 5. 获取 Top-K 显著的 Patch 索引
        _, top_indices = torch.topk(flat_scores, num_mix)

        # 6. 执行混合
        img_mixed = copy.deepcopy(img_target)

        # 遍历选中的索引，将 Source 的 patch 贴到 Target 上
        # 这种方式比循环快：创建一个 mask
        mask = torch.zeros((H, W), dtype=torch.float32)

        # 这里需要将平铺的 index 映射回 (h_idx, w_idx)
        # 既然我们只要粘贴，可以先在 mask 上操作
        mask_patches = mask.view(self.grid_size, grid_h, self.grid_size, grid_w)

        # 将 top_indices 对应的 mask 区域置 1
        # 这种 tensor 操作需要一点技巧，我们可以简单循环 top_indices (数量不多，256以内)
        for idx in top_indices:
            h_idx = idx // self.grid_size
            w_idx = idx % self.grid_size
            mask_patches[h_idx, :, w_idx, :] = 1.0

        # 恢复 mask 形状
        mask = mask_patches.view(H, W)

        # 混合操作: Mask 区域用 Source，其他区域保持 Target (即 img_mixed 初始值)
        # img_mixed = mask * img_source + (1-mask) * img_target
        # 因为 img_mixed 已经是 img_target，我们只要把 mask 为 1 的地方换成 source
        img_mixed = img_mixed * (1 - mask) + img_source * mask

        return img_mixed

    def __getitem__(self, index):
        # 1. 获取当前样本 (Weak, Strong_Base, Label, Domain)
        # 注意：GDRBench 返回的是 (img, mask, label, domain)
        # 这里 trans_mask 产生的是 Strong Base (含 FundusAug)
        data = self.dataset[index]
        img_weak = data[0]
        img_strong = data[1]
        label = data[2]
        domain = data[3]

        # 2. SPMix 处理 (仅对 Strong Aug 进行混合)
        # 以一定的概率触发 SPMix
        if np.random.rand() < self.prob:
            # 随机选取另一个样本作为 "Source" (提供 Patch 的一方)
            # 当前样本 img_strong 作为 "Target" (底图)
            rand_idx = np.random.randint(0, len(self.dataset))
            data_partner = self.dataset[rand_idx]
            img_strong_source = data_partner[1]

            # 执行 Saliency-Guided Patch-Based Mixup
            # 将 partner 中最显著的区域贴到当前图片上
            img_strong = self.apply_spmix(img_source=img_strong_source, img_target=img_strong)

            # 注意：在 MASK_SIAM 架构下，我们通常不混合 Label，
            # 而是将这种混合视为一种强力的数据增强 (Strong Augmentation)，
            # 目的是让模型在输入分布剧烈变化（被遮挡、被替换）时仍能提取一致的特征。

        # 重新打包
        return img_weak, img_strong, label, domain


def get_transform(cfg):
    re_size = getattr(cfg.INPUT, 'SIZE', 224)

    normalize = get_normalize(cfg)

    tra_train = transforms.Compose(
        [transforms.RandomResizedCrop(re_size, scale=(0.7, 1.0)), transforms.RandomHorizontalFlip(),
         transforms.ColorJitter(0.3, 0.3, 0.3, 0.3), transforms.RandomGrayscale(), transforms.ToTensor(), normalize])

    tra_test = transforms.Compose([transforms.Resize((re_size, re_size)), transforms.ToTensor(), normalize])

    tra_mask = transforms.Compose([transforms.Resize((re_size, re_size)), transforms.ToTensor()])

    return tra_train, tra_test, tra_mask


def get_pre_FundusAug(cfg):
    # 配置 Strong Augmentation 的基础参数
    jitter_b = getattr(cfg.TRANSFORM, 'COLORJITTER_B', 0.2)
    jitter_c = getattr(cfg.TRANSFORM, 'COLORJITTER_C', 0.2)
    jitter_s = getattr(cfg.TRANSFORM, 'COLORJITTER_S', 0.2)
    jitter_h = getattr(cfg.TRANSFORM, 'COLORJITTER_H', 0.1)
    aug_prob = getattr(cfg.TRANSFORM, 'AUGPROB', 0.5)

    re_size = getattr(cfg.INPUT, 'SIZE', 224)
    size = int(re_size * (256 / 224))

    normalize = get_normalize(cfg)

    # 1. Weak Augmentation (img_ori)
    tra_train = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomCrop(re_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    # 2. Strong Augmentation Base (img_strong) - 包含 FundusAug
    # SPMix 将在此基础上进一步混合
    tra_mask = transforms.Compose([
        transforms.Resize((size, size)),
        # FundusAug 增强 (Halo, Hole, Spot, Blur)
        FundusAug.Compose([
            FundusAug.Sharpness(prob=aug_prob),
            FundusAug.Halo(size, prob=aug_prob),
            FundusAug.Hole(size, prob=aug_prob),
            FundusAug.Spot(size, prob=aug_prob),
            FundusAug.Blur(prob=aug_prob)
        ]),
        transforms.RandomCrop(re_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=jitter_b, contrast=jitter_c, saturation=jitter_s, hue=jitter_h),
        transforms.ToTensor(),
        normalize
    ])

    tra_test = transforms.Compose(
        [transforms.Resize((size, size)), transforms.CenterCrop(re_size), transforms.ToTensor(), normalize])

    return tra_train, tra_test, tra_mask


def get_post_FundusAug(cfg):
    # 此函数保留以备不时之需，但在 GDRNet 流程中已被 get_pre_FundusAug + SPMixDatasetWrapper 替代
    aug_prob = getattr(cfg.TRANSFORM, 'AUGPROB', 0.5)

    re_size = getattr(cfg.INPUT, 'SIZE', 224)
    size = int(re_size * (256 / 224))

    normalize = get_normalize(cfg)

    tra_fundus_1 = FundusAug.Compose(
        [FundusAug.Sharpness(prob=aug_prob), FundusAug.Halo(size, prob=aug_prob), FundusAug.Hole(size, prob=aug_prob),
         FundusAug.Spot(size, prob=aug_prob), FundusAug.Blur(prob=aug_prob)])

    tra_fundus_2 = transforms.Compose(
        [transforms.RandomCrop(re_size), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), normalize])

    return {'post_aug1': tra_fundus_1, 'post_aug2': tra_fundus_2}


def get_normalize(cfg=None):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if cfg is not None and hasattr(cfg.MODEL, 'FPT') and hasattr(cfg.MODEL.FPT, 'LPM_PATH'):
        try:
            processor = AutoImageProcessor.from_pretrained(cfg.MODEL.FPT.LPM_PATH)
            if hasattr(processor, 'image_mean') and hasattr(processor, 'image_std'):
                if isinstance(processor.image_mean, (list, tuple)):
                    mean = processor.image_mean
                elif isinstance(processor.image_mean, (float, int)):
                    mean = [processor.image_mean] * 3

                if isinstance(processor.image_std, (list, tuple)):
                    std = processor.image_std
                elif isinstance(processor.image_std, (float, int)):
                    std = [processor.image_std] * 3
        except Exception:
            pass

    return transforms.Normalize(mean=mean, std=std)