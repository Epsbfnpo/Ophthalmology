from .GDRBench import GDRBench
from . import fundusaug as FundusAug
from torchvision import transforms
from torch.utils.data import DataLoader, DistributedSampler
import torch


def get_normalize():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return transforms.Normalize(mean=mean, std=std)


def get_transform(cfg):
    jitter_b = getattr(cfg.TRANSFORM, 'COLORJITTER_B', 0.4)
    jitter_c = getattr(cfg.TRANSFORM, 'COLORJITTER_C', 0.2)
    jitter_s = getattr(cfg.TRANSFORM, 'COLORJITTER_S', 0.2)
    jitter_h = getattr(cfg.TRANSFORM, 'COLORJITTER_H', 0.1)
    size = 256
    re_size = 224
    normalize = get_normalize()
    tra_train = transforms.Compose([transforms.Resize(size),
                                    transforms.ColorJitter(brightness=jitter_b, contrast=jitter_c, saturation=jitter_s,
                                                           hue=jitter_h), transforms.ToTensor()])
    tra_test = transforms.Compose(
        [transforms.Resize(size), transforms.CenterCrop(re_size), transforms.ToTensor(), normalize])
    tra_mask = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])

    # Fundus Augmentation
    aug_prob = getattr(cfg.TRANSFORM, 'AUGPROB', 0.5)
    tra_fundus = FundusAug.Compose(
        [FundusAug.Sharpness(prob=aug_prob), FundusAug.Halo(size, prob=aug_prob), FundusAug.Hole(size, prob=aug_prob),
         FundusAug.Spot(size, prob=aug_prob), FundusAug.Blur(prob=aug_prob)])

    return tra_train, tra_test, tra_mask, tra_fundus


def get_pre_FundusAug(cfg):
    # For GDRNet specific augmentation if needed
    return get_transform(cfg)


def get_dataset(args, cfg):
    # Select transforms
    # Simplified logic: standardizing to use get_transform for consistency
    train_ts, test_ts, tra_mask, tra_fundus = get_transform(cfg)

    batch_size = args.batch_size
    drop_last = getattr(cfg, 'DROP_LAST', True)
    num_worker = args.workers

    # --- Train Set (Source) ---
    # [Updated] Pass 'cfg' to GDRBench
    train_dataset = GDRBench(
        root=cfg.DATASET.ROOT,
        source_domains=cfg.DATASET.SOURCE_DOMAINS,
        target_domains=cfg.DATASET.TARGET_DOMAINS,
        mode='train',
        cfg=cfg,  # <--- CRITICAL UPDATE HERE
        trans_basic=train_ts,
        trans_mask=tra_mask,
        trans_fundus=tra_fundus
    )

    # DDP Sampler
    train_sampler = None
    shuffle = True
    if args.local_rank != -1:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_worker,
        drop_last=drop_last,
        pin_memory=True,
        sampler=train_sampler
    )

    # --- Val/Test Set (Target) ---
    # Val dataset usually on source domains
    val_dataset = GDRBench(
        root=cfg.DATASET.ROOT,
        source_domains=cfg.DATASET.SOURCE_DOMAINS,
        target_domains=cfg.DATASET.TARGET_DOMAINS,
        mode='val',
        cfg=cfg,
        trans_basic=test_ts,
        trans_mask=tra_mask
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_worker,
        pin_memory=True
    )

    # Test dataset on target domains
    test_dataset = GDRBench(
        root=cfg.DATASET.ROOT,
        source_domains=cfg.DATASET.SOURCE_DOMAINS,
        target_domains=cfg.DATASET.TARGET_DOMAINS,
        mode='test',
        cfg=cfg,
        trans_basic=test_ts,
        trans_mask=tra_mask
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_worker,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader