from .GDRBench import GDRBench
from . import fundusaug as FundusAug
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageFilter
from .mask import FrequencyMaskGenerator
from .spm import ShufflePatchMixOverlap_all

import random


import sys
# sys.path.append('/home/aleksandrmatsun/CCSDG')
from ccsdg.utils.fourier import FDA_source_to_target_np
from ccsdg.datasets.utils.normalize import normalize_image
from ccsdg.datasets.utils.slaug import LocationScaleAugmentation




def get_dataset(cfg):
    if cfg.ALGORITHM != 'GDRNet' and cfg.ALGORITHM != 'GDRNet_DUAL'  and cfg.ALGORITHM != 'GDRNet_DUAL_MASK' and cfg.ALGORITHM != "GDRNet_Mask" and  cfg.ALGORITHM !="GDRNet_DUAL_MASK_SIAM" and cfg.ALGORITHM != 'GDRNet_MASK_SIAM':  ### 
        train_ts, test_ts, tra_fundus = get_transform(cfg)
    else:
        train_ts, test_ts, tra_fundus, tra_train_freq, tra_train_freq2 = get_pre_FundusAug(cfg)

    batch_size = cfg.BATCH_SIZE
    drop_last = cfg.DROP_LAST
    num_worker = min (batch_size // 4, 4)
    
    # def __init__(self, root, source_domains, target_domains, mode, trans_basic=None, trans_mask = None, trans_fundus=None):

    train_dataset = GDRBench(root = cfg.DATASET.ROOT, source_domains= cfg.DATASET.SOURCE_DOMAINS, target_domains = cfg.DATASET.TARGET_DOMAINS,  mode = 'train', trans_basic=train_ts, trans_mask=tra_fundus,trans_basic_freq=tra_train_freq, trans_basic_freq2=tra_train_freq2, class_balance=cfg.DATASET.IMBALANCE)
    
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=num_worker, drop_last=drop_last, pin_memory=True)

    val_dataset = GDRBench(root = cfg.DATASET.ROOT, source_domains= cfg.DATASET.SOURCE_DOMAINS, target_domains = cfg.DATASET.TARGET_DOMAINS, mode = 'val', trans_basic=test_ts)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle=False, num_workers=num_worker)

    test_dataset = GDRBench(root = cfg.DATASET.ROOT, source_domains= cfg.DATASET.SOURCE_DOMAINS, target_domains = cfg.DATASET.TARGET_DOMAINS,   mode = 'test', trans_basic=test_ts)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False, num_workers=num_worker)        

    dataset_size = [len(train_dataset), len(val_dataset), len(test_dataset)]
    print('dataset size:', dataset_size)
    return train_loader, val_loader, test_loader, dataset_size

def get_transform(cfg):

    size_train = 512
    size = 256
    re_size = 224
    normalize = get_normalize()
    tra_train = transforms.Compose([
        transforms.RandomResizedCrop(re_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        normalize,
    ])
    
    tra_test = transforms.Compose([
        transforms.Resize((re_size, re_size)),
        transforms.ToTensor(),
        normalize
    ])
    
    tra_mask= transforms.Compose([
                transforms.Resize(re_size),
                transforms.ToTensor()])
    
    return tra_train, tra_test, tra_mask


    
# FundusAug contains ["RandomCrop", "RandomHorizontalFlip", "RandomVerticalFlip", "ColorJitter", "Sharpness", "Halo", "Hole", "Spot", "Blur"],
# these operations are splited into pre_FundusAug and post_FundusAug

def get_pre_FundusAug(cfg):

    size = 256
    size_train = 256
    re_size = 224
    normalize = get_normalize()
    mask_generator = FrequencyMaskGenerator(ratio=0.15, band='high')  ##  # 'low', 'mid', 'high', 'all'
    tra_train = transforms.Compose([
        # transforms.Lambda(lambda img: mask_generator.transform(img)),
        transforms.Resize(size_train),
        transforms.ColorJitter( brightness=cfg.TRANSFORM.COLORJITTER_B, \
                                contrast=cfg.TRANSFORM.COLORJITTER_C, \
                                saturation=cfg.TRANSFORM.COLORJITTER_S, \
                                hue=cfg.TRANSFORM.COLORJITTER_H),
        # transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),  ## Add
        transforms.ToTensor()
    ])
    
    tra_train_freq = transforms.Compose([
        transforms.Lambda(lambda img: mask_generator.transform(img)),
        transforms.Resize(size),
        transforms.ColorJitter( brightness=cfg.TRANSFORM.COLORJITTER_B, \
                                contrast=cfg.TRANSFORM.COLORJITTER_C, \
                                saturation=cfg.TRANSFORM.COLORJITTER_S, \
                                hue=cfg.TRANSFORM.COLORJITTER_H),
        transforms.ToTensor()
    ])
    
    
    tra_train_v2 = transforms.Compose([
        # transforms.RandomResizedCrop(size_train, scale=(0.2, 1.)),
        transforms.RandomResizedCrop(size_train, scale=(0.7, 1.0)),
        # transforms.RandomGrayscale(p=0.2),
        # transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),  ## Add
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
        transforms.RandomGrayscale(),
        # transforms.ColorJitter( brightness=cfg.TRANSFORM.COLORJITTER_B, \
        #                         contrast=cfg.TRANSFORM.COLORJITTER_C, \
        #                         saturation=cfg.TRANSFORM.COLORJITTER_S, \
        #                         hue=cfg.TRANSFORM.COLORJITTER_H),                                
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        # normalize,
    ])
    
    tra_train_v3 = transforms.Compose([
        # transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomResizedCrop(re_size, scale=(0.7, 1.0)),
        # transforms.RandomResizedCrop(re_size, scale=(0.7, 1.0)),
        transforms.RandomGrayscale(p=0.2),
        # transforms.RandomGrayscale(),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),  ## Add
        transforms.ColorJitter( brightness=cfg.TRANSFORM.COLORJITTER_B, \
                                contrast=cfg.TRANSFORM.COLORJITTER_C, \
                                saturation=cfg.TRANSFORM.COLORJITTER_S, \
                                hue=cfg.TRANSFORM.COLORJITTER_H),
        # transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    
    
    
    tra_train_mix = transforms.Compose([
        transforms.Lambda(lambda img: ShufflePatchMixOverlap_all(img)),
        transforms.Resize(size),
        transforms.ColorJitter( brightness=cfg.TRANSFORM.COLORJITTER_B, \
                                contrast=cfg.TRANSFORM.COLORJITTER_C, \
                                saturation=cfg.TRANSFORM.COLORJITTER_S, \
                                hue=cfg.TRANSFORM.COLORJITTER_H),
        
        transforms.ToTensor()
    ])
    
    tra_test = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(re_size),
            transforms.ToTensor(),
            normalize])
    
    tra_mask= transforms.Compose([
                transforms.Resize(size_train),
                transforms.ToTensor()])
    
    if cfg.TRANSFORM.FREQ:
        return tra_train, tra_test, tra_mask, tra_train_v2 , tra_train_v3
    else:
        return tra_train, tra_test, tra_mask, None, None

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
    
    
def get_post_FundusAug(cfg):
    aug_prob = cfg.TRANSFORM.AUGPROB
    size = 256
    re_size = 224
    normalize = get_normalize()

    tra_fundus_1 = FundusAug.Compose([
        FundusAug.Sharpness(prob = aug_prob),
        FundusAug.Halo(size, prob=aug_prob),
        FundusAug.Hole(size, prob=aug_prob),
        FundusAug.Spot(size, prob=aug_prob),
        FundusAug.Blur(prob=aug_prob)
    ])
    
    tra_fundus_2 = transforms.Compose([
                transforms.RandomCrop(re_size),  
                # transforms.RandomResizedCrop(re_size, scale=(0.7, 1.0)),  
                # transforms.RandomGrayscale(p=0.2),  ## Add 
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                normalize])
    
    
    tra_fundus_3 = transforms.Compose([
                # transforms.Resize(256),
                # transforms.RandomCrop(re_size),
                transforms.RandomResizedCrop(re_size, scale=(0.7, 1.0)),  
                # transforms.RandomCrop(re_size),  #### 不进行随机裁剪，保全全图
                # transforms.RandomGrayscale(p=0.2),  ## Add 
                transforms.RandomGrayscale(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                normalize])
    
    tra_train_4 = transforms.Compose([
            transforms.RandomResizedCrop(re_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            normalize,
        ])

    # tra_fundus_2 = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.2, 1.)), 
    #                                         transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    #                                         transforms.RandomGrayscale(p=0.2),
    #                                         transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
    #                                         transforms.RandomHorizontalFlip(p=0.5),
    #                                         transforms.ToTensor(),
    #                                         normalize])
    
    
    # tra_fundus_3 = transforms.Compose([
    #             # transforms.RandomCrop(re_size),
    #             # transforms.RandomHorizontalFlip(),
    #             # transforms.RandomVerticalFlip(),
    #             normalize])
    
    return {'post_aug1':tra_fundus_1, 'post_aug2':tra_fundus_2, 'post_aug3':tra_fundus_3, 'post_aug4':tra_train_4}

def get_normalize():

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # mean = [0.5, 0.5, 0.5]
    # std = [0.5, 0.5, 0.5]
    
    normalize = transforms.Normalize(mean=mean, std=std)
    
    return normalize