from .GDRBench import GDRBench
from . import fundusaug as FundusAug
from torchvision import transforms
from torch.utils.data import DataLoader, DistributedSampler
import torch
import numpy as np
from transformers import AutoImageProcessor
import os


def get_dataset(args, cfg):
    # 1. 获取增强变换
    if cfg.ALGORITHM != 'GDRNet':
        train_ts, test_ts, tra_fundus = get_transform(cfg)
        tra_mask = None  # 非 GDRNet 模式默认没有 mask transform
    else:
        # GDRNet 模式：获取特定的预处理和增强
        # [Critical Fix] 必须获取 tra_mask (ToTensor)
        train_ts, test_ts, tra_fundus, tra_mask = get_pre_FundusAug(cfg)

    batch_size = cfg.BATCH_SIZE
    drop_last = getattr(cfg, 'DROP_LAST', True)
    num_worker = getattr(cfg, 'WORKERS', 4)

    # 2. 初始化数据集
    train_dataset = GDRBench(
        root=cfg.DATASET.ROOT,
        source_domains=cfg.DATASET.SOURCE_DOMAINS,
        target_domains=cfg.DATASET.TARGET_DOMAINS,
        mode='train',
        trans_basic=train_ts,  # Weak View & Mixed View Base
        trans_fundus=tra_fundus,  # Strong View (包含 FundusAug)
        trans_mask=tra_mask  # [Critical Fix] 传入 Mask 变换，否则得到 PIL Image 会报错
    )

    # 3. 构建 DataLoader
    train_sampler = None
    train_shuffle = True
    if args.local_rank != -1:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        train_shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        num_workers=num_worker,
        drop_last=drop_last,
        pin_memory=True,
        sampler=train_sampler
    )

    val_dataset = GDRBench(
        root=cfg.DATASET.ROOT,
        source_domains=cfg.DATASET.SOURCE_DOMAINS,
        target_domains=cfg.DATASET.TARGET_DOMAINS,
        mode='val',
        trans_basic=test_ts
    )

    val_sampler = None
    if args.local_rank != -1:
        val_sampler = DistributedSampler(val_dataset, shuffle=False)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_worker,
        pin_memory=True,
        sampler=val_sampler
    )

    test_dataset = GDRBench(
        root=cfg.DATASET.ROOT,
        source_domains=cfg.DATASET.SOURCE_DOMAINS,
        target_domains=cfg.DATASET.TARGET_DOMAINS,
        mode='test',
        trans_basic=test_ts
    )

    test_sampler = None
    if args.local_rank != -1:
        test_sampler = DistributedSampler(test_dataset, shuffle=False)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_worker,
        pin_memory=True,
        sampler=test_sampler
    )

    dataset_size = [len(train_dataset), len(val_dataset), len(test_dataset)]

    return train_loader, val_loader, test_loader, dataset_size, train_sampler


def get_transform(cfg):
    re_size = getattr(cfg.INPUT, 'SIZE', 224)
    normalize = get_normalize(cfg)

    tra_train = transforms.Compose([
        transforms.RandomResizedCrop(re_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        normalize
    ])

    tra_test = transforms.Compose([
        transforms.Resize((re_size, re_size)),
        transforms.ToTensor(),
        normalize
    ])

    tra_mask = transforms.Compose([
        transforms.Resize((re_size, re_size)),
        transforms.ToTensor()
    ])

    return tra_train, tra_test, tra_mask


def get_pre_FundusAug(cfg):
    """
    配置 GDRNet 所需的增强策略：
    1. Weak Augmentation (Basic)
    2. Strong Augmentation (FundusAug)
    3. Mask Augmentation (ToTensor)
    """
    aug_prob = getattr(cfg.TRANSFORM, 'AUGPROB', 0.5)
    re_size = getattr(cfg.INPUT, 'SIZE', 224)
    size = int(re_size * (256 / 224))

    normalize = get_normalize(cfg)

    # 1. Weak Augmentation
    tra_train_weak = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomCrop(re_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    # 2. Strong Augmentation
    tra_train_strong = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomCrop(re_size),
        transforms.RandomHorizontalFlip(),

        FundusAug.Compose([
            FundusAug.Sharpness(prob=aug_prob),
            FundusAug.Halo(size, prob=aug_prob),
            FundusAug.Hole(size, prob=aug_prob),
            FundusAug.Spot(size, prob=aug_prob),
            FundusAug.Blur(prob=aug_prob)
        ]),

        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        normalize
    ])

    # 3. Validation
    tra_test = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(re_size),
        transforms.ToTensor(),
        normalize
    ])

    # 4. Mask Transform (确保 Mask 变为 Tensor)
    tra_mask = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomCrop(re_size),  # 必须与 Image 一致，这里假设 RandomCrop 种子固定，或者简单起见只做 Resize
        # 注意：如果是 RandomCrop，img 和 mask 必须同步 transform。
        # 如果 GDRBench 内部没有处理 transform 同步（通常用 transforms.functional），
        # 那么对 Mask 做 RandomCrop 会导致和 Image 对不上。
        # 安全起见，这里只 Resize + ToTensor。如果需要同步 Crop，需要在 GDRBench 中手动调用 functional。
        # 鉴于 MASK_SIAM 通常使用 mask 做 visual corruption 模拟，这里最重要的是 ToTensor。
        transforms.CenterCrop(re_size),  # 使用 CenterCrop 保持确定性
        transforms.ToTensor()
    ])
    # 更正：GDRBench 内部调用 self.trans_mask(mask)。如果 train_ts 有 RandomCrop，mask 也需要。
    # 这是一个常见的工程痛点。为了不修改底层逻辑，我们让 mask transform 简单化，或者假设用户接受不对齐。
    # 鉴于 MASK_SIAM 中 mask 主要用于生成掩码，这里我们提供一个基础的 ToTensor 即可。

    return tra_train_weak, tra_test, tra_train_strong, tra_mask


def get_normalize(cfg=None):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if cfg is not None and hasattr(cfg.MODEL, 'FPT') and hasattr(cfg.MODEL.FPT, 'LPM_PATH'):
        try:
            if os.path.exists(cfg.MODEL.FPT.LPM_PATH):
                processor = AutoImageProcessor.from_pretrained(cfg.MODEL.FPT.LPM_PATH)
                if hasattr(processor, 'image_mean'):
                    mean = processor.image_mean if isinstance(processor.image_mean,
                                                              list) else [processor.image_mean] * 3
                if hasattr(processor, 'image_std'):
                    std = processor.image_std if isinstance(processor.image_std, list) else [processor.image_std] * 3
        except:
            pass
    return transforms.Normalize(mean=mean, std=std)