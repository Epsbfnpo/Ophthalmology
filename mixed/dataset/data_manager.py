from .GDRBench import GDRBench
from . import fundusaug as FundusAug
from torchvision import transforms
from torch.utils.data import DataLoader, DistributedSampler
import torch


def get_dataset(args, cfg):
    # 预处理选择
    if cfg.ALGORITHM != 'GDRNet':
        train_ts, test_ts, tra_fundus = get_transform(cfg)
    else:
        train_ts, test_ts, tra_fundus = get_pre_FundusAug(cfg)

    batch_size = args.batch_size
    drop_last = getattr(cfg, 'DROP_LAST', True)
    num_worker = args.workers

    # --- 训练集 (Source) ---
    train_dataset = GDRBench(
        root=cfg.DATASET.ROOT,
        source_domains=cfg.DATASET.SOURCE_DOMAINS,
        target_domains=cfg.DATASET.TARGET_DOMAINS,
        mode='train',
        trans_basic=train_ts,
        trans_mask=tra_fundus
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

    # --- 验证/测试集 (Target) ---
    val_dataset = GDRBench(root=cfg.DATASET.ROOT, source_domains=cfg.DATASET.SOURCE_DOMAINS,
                           target_domains=cfg.DATASET.TARGET_DOMAINS, mode='val', trans_basic=test_ts)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker)

    test_dataset = GDRBench(root=cfg.DATASET.ROOT, source_domains=cfg.DATASET.SOURCE_DOMAINS,
                            target_domains=cfg.DATASET.TARGET_DOMAINS, mode='test', trans_basic=test_ts)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker)

    dataset_size = [len(train_dataset), len(val_dataset), len(test_dataset)]

    return train_loader, val_loader, test_loader, dataset_size, train_sampler


def get_transform(cfg):
    """
    ERM, FPT 等常规方法的 Transform
    根据 cfg.INPUT.SIZE 动态调整分辨率
    """
    # 尝试从配置中获取输入尺寸，如果没有定义则默认为 224
    re_size = getattr(cfg.INPUT, 'SIZE', 224)

    normalize = get_normalize()

    # 训练集：随机裁剪 resize 到指定大小 (e.g., 512)
    tra_train = transforms.Compose([
        transforms.RandomResizedCrop(re_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        normalize
    ])

    # 测试集：强制 resize 到指定大小
    tra_test = transforms.Compose([
        transforms.Resize((re_size, re_size)),
        transforms.ToTensor(),
        normalize
    ])

    # Mask：调整到相同大小
    tra_mask = transforms.Compose([
        transforms.Resize((re_size, re_size)),
        transforms.ToTensor()
    ])

    return tra_train, tra_test, tra_mask


def get_pre_FundusAug(cfg):
    """
    GDRNet 专用的 Transform (Pre-Augmentation)
    需要保持 size (大) 和 re_size (小) 的比例关系
    """
    jitter_b = getattr(cfg.TRANSFORM, 'COLORJITTER_B', 0.2)
    jitter_c = getattr(cfg.TRANSFORM, 'COLORJITTER_C', 0.2)
    jitter_s = getattr(cfg.TRANSFORM, 'COLORJITTER_S', 0.2)
    jitter_h = getattr(cfg.TRANSFORM, 'COLORJITTER_H', 0.1)

    # 目标输入尺寸 (例如 224 或 512)
    re_size = getattr(cfg.INPUT, 'SIZE', 224)
    # 原始增强尺寸 (通常比输入大一点，保持约 1.14 倍比例)
    # 例如 224 -> 256; 512 -> 585
    size = int(re_size * (256 / 224))

    normalize = get_normalize()

    # 注意：GDRNet 这里输出的是 size (大图)，后续会在 get_post_FundusAug 中裁剪
    tra_train = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ColorJitter(brightness=jitter_b, contrast=jitter_c, saturation=jitter_s,
                               hue=jitter_h),
        transforms.ToTensor()
    ])

    # 验证集直接 CenterCrop 到 re_size
    tra_test = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(re_size),
        transforms.ToTensor(),
        normalize
    ])

    tra_mask = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])

    return tra_train, tra_test, tra_mask


def get_post_FundusAug(cfg):
    """
    GDRNet 专用的 Post-Augmentation (处理伪影和最终裁剪)
    """
    aug_prob = getattr(cfg.TRANSFORM, 'AUGPROB', 0.5)

    re_size = getattr(cfg.INPUT, 'SIZE', 224)
    size = int(re_size * (256 / 224))

    normalize = get_normalize()

    # 伪影增强在较大的 size 上进行
    tra_fundus_1 = FundusAug.Compose([
        FundusAug.Sharpness(prob=aug_prob),
        FundusAug.Halo(size, prob=aug_prob),
        FundusAug.Hole(size, prob=aug_prob),
        FundusAug.Spot(size, prob=aug_prob),
        FundusAug.Blur(prob=aug_prob)
    ])

    # 最终随机裁剪到 re_size
    tra_fundus_2 = transforms.Compose([
        transforms.RandomCrop(re_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        normalize
    ])

    return {'post_aug1': tra_fundus_1, 'post_aug2': tra_fundus_2}


def get_normalize():
    return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])