"""
This code collected some methods from DomainBed (https://github.com/facebookresearch/DomainBed) and other SOTA methods.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, collections, logging
from collections import OrderedDict

import utils.misc as misc
from utils.validate import algorithm_validate,algorithm_validate_distill,algorithm_validate_distill_v2
import modeling.model_manager as models
from modeling.losses import DahLoss,DahLoss_Dual, DahLoss_Dual_BalSCL,DahLoss_Mask, DahLoss_Dual_Siam, DahLoss_Siam,DahLoss_Siam_Fastmoco
from modeling.nets import LossValley, AveragedModel
from dataset.data_manager import get_post_FundusAug

from backpack import backpack, extend
from backpack.extensions import BatchGrad

from utils.optimizers import get_optimizer
from masking import Masking
from itertools import combinations
import copy

from dataset.mask import FrequencyMaskGenerator,FrequencyMaskGenerator_Tensor
from torch.cuda.amp import GradScaler, autocast
from torchvision.utils import save_image
from torchvision import transforms


from guided_filter_pytorch.HFC_filter import HFCFilter
from guided_filter_pytorch.ModifiedHFCFilter import FourierButterworthHFCFilter
from modeling import networks
from copy import deepcopy


ALGORITHMS = [
    'ERM',
    'GDRNet',
    'GDRNet_DUAL',
    'GDRNet_MASK_SIAM',
    'GDRNet_DUAL_MASK',
    'GDRNet_DUAL_MASK_SIAM',
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

class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - validate()
    - save_model()
    - renew_model()
    - predict()
    """
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
    """Empirical Risk Minimization (ERM)"""
    def __init__(self, num_classes, cfg):
        super(ERM, self).__init__(num_classes, cfg)
        
        self.network = models.get_net(cfg)

        if cfg.BACKBONE == "LaDeDa33":
            self.classifier = models.get_classifier(2048, cfg)

        else:
            self.classifier = models.get_classifier(self.network.out_features(), cfg)
            
            
        # self.classifier = models.get_classifier(self.network.out_features(), cfg)

        self.optimizer = torch.optim.SGD(
        # self.optimizer = torch.optim.Adam(
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
    
# Our method
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




def init_weights_MLPHead(module, init_method='He'):
    for _, m in module.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if init_method == 'He':
                nn.init.kaiming_normal_(m.weight.data)
            elif init_method == 'Xavier':
                nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, val=0)
                

def create_feather_mask(height, width, feather_size=4):
    """
    Create a 2D mask of shape (height x width) that smoothly transitions
    from 1.0 in the interior to 0.0 at the edges over 'feather_size' pixels.
    """
    mask = np.ones((height, width), dtype=np.float32)
    ramp = np.linspace(0, 1, feather_size, dtype=np.float32)

    # Top fade
    mask[ :feather_size, :]    *= ramp[:, None]
    # Bottom fade
    mask[ -feather_size:, :]   *= ramp[::-1, None]
    # Left fade
    mask[ :, :feather_size]    *= ramp[None, :]
    # Right fade
    mask[ :, -feather_size:]   *= ramp[None, ::-1]

    return mask

def edgelogic(i, j, patch_height, patch_width, num_patches_h, num_patches_w, overlap):
    """
    Example 'edgelogic' that extends patch size in the middle,
    but does not exceed (patch_height+2*overlap, patch_width+2*overlap).
    Modify as needed for your scenario.
    """
    # Base top-left (no overlap):
    start_h = i * patch_height
    start_w = j * patch_width
    end_h   = start_h + patch_height
    end_w   = start_w + patch_width

    # If i == 0, we add overlap only at the bottom. If i == last, only top, etc.
    # This is just one possible logic:
    if i == 0:
        end_h += 2 * overlap
    elif i == num_patches_h - 1:
        start_h -= 2 * overlap
    else:
        start_h -= overlap
        end_h   += overlap

    if j == 0:
        end_w += 2 * overlap
    elif j == num_patches_w - 1:
        start_w -= 2 * overlap
    else:
        start_w -= overlap
        end_w   += overlap

    # Make sure we don't go negative or beyond image dimension here if needed

    return start_h, end_h, start_w, end_w



class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        if len(self.sigma) == 2:
            sigma = random.uniform(self.sigma[0], self.sigma[1])
        else:
            sigma = self.sigma[0]
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class HighPassFilter(object):
    """High pass filter augmentation: original image minus image after low pass filter (GaussianBlur)"""
    def __init__(self, kernel_size, sigma):
        self.kernel_size = kernel_size
        self.sigma = sigma
    
    def __call__(self,x):
        T = transforms.GaussianBlur(kernel_size=self.kernel_size, sigma=self.sigma)
        x_lp = T(x)
        x_hp = x - x_lp
        return x_hp
    
    
    
def hfc_mul_mask(hfc_filter, image, mask, do_norm=False):
    # print('image', image.min(), image.max())
    hfc = hfc_filter((image / 2 + 0.5), mask)

    if do_norm:
        hfc = 2 * hfc - 1
    # return hfc
    return (hfc + 1) * mask - 1
    # return image


def dkd_loss(logits_student, logits_teacher, target, alpha=1, beta=8, temperature=4):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    # pdb.set_trace()

    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
            F.kl_div(log_pred_student, pred_teacher, reduction='batchmean')
            * (temperature ** 2)
        # / target.shape[0]
    )

    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
            F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='batchmean')
            * (temperature ** 2)
        # / target.shape[0]
    )
    return alpha * tckd_loss + beta * nckd_loss


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return t2


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


class LogitAdjust_KD(nn.Module):

    def __init__(self, cls_num_list, tau=1, weight=None):
        super(LogitAdjust_KD, self).__init__()
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = tau * torch.log(cls_p_list)
        self.m_list = m_list.view(1, -1)
        self.weight = weight

    def forward(self, x, target):
        x_m = x + self.m_list
        # target = target + self.m_list
        target = F.softmax( target, 1)
        return F.cross_entropy(x_m, target, weight=self.weight)

    
class LogitAdjust_KD_V2(nn.Module):

    def __init__(self, cls_num_list, tau1=1, tau2=1,  temperature=1.0, weight=None):
        super(LogitAdjust_KD_V2, self).__init__()
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list1 = tau1 * torch.log(cls_p_list)
        m_list2 = tau2 * torch.log(cls_p_list)

        self.m_list1 = m_list1.view(1, -1)
        self.m_list2 = m_list2.view(1, -1)

        self.weight = weight
        self.temperature =  temperature

    def forward(self, x, target, label):
        x_m = x + self.m_list1
        target = target + self.m_list2
        target = F.softmax( target / self.temperature , 1).detach()
        output_target_max, output_target_index = torch.max(F.softmax((target), dim=1).detach(), dim=1)
        
        return  -(target[(output_target_index == label)] * torch.log(x_m[(output_target_index == label)])).sum() / target.size()[0]
    
    

class LogitAdjust(nn.Module):

    def __init__(self, cls_num_list, tau=1, weight=None):
        super(LogitAdjust, self).__init__()
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list =  torch.log(torch.pow(cls_num_list,  tau) + 1e-9)  # tau * torch.log(cls_p_list)
        self.m_list = m_list.view(1, -1)
        self.weight = weight

    def forward(self, x, target):
        x_m = x + self.m_list
        return F.cross_entropy(x_m, target, weight=self.weight)


class LogitAdjust_KD(nn.Module):

    def __init__(self, cls_num_list, tau=1, weight=None):
        super(LogitAdjust_KD, self).__init__()
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = torch.log(torch.pow(cls_num_list,  tau) + 1e-9) #  tau * torch.log(cls_p_list)
        self.m_list = m_list.view(1, -1)
        self.weight = weight

    def forward(self, x, target):
        x_m = x + self.m_list
        target = target + self.m_list
        target = F.softmax( target, 1)
        return F.cross_entropy(x_m, target, weight=self.weight)

class LogitAdjust_KD_V2(nn.Module):

    def __init__(self, cls_num_list, tau1=1, tau2=1,  temperature=1.0, weight=None):
        super(LogitAdjust_KD_V2, self).__init__()
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list1 =   torch.log(torch.pow(cls_num_list,  tau1) + 1e-9) # tau1 * torch.log(cls_p_list)
        m_list2 =  torch.log(torch.pow(cls_num_list,  tau2) + 1e-9) #  tau2 * torch.log(cls_p_list)

        self.m_list1 = m_list1.view(1, -1)
        self.m_list2 = m_list2.view(1, -1)

        self.weight = weight
        self.temperature =  temperature

    def forward(self, x, target, label):
        x_m = x + self.m_list1
        x_m = F.log_softmax(x_m / self.temperature, dim=1)
        target = target + self.m_list2
        target_t = F.softmax( target / self.temperature , 1).detach()
        output_target_max, output_target_index = torch.max(F.softmax((target), dim=1).detach(), dim=1)
        
        return  F.kl_div(x_m[(output_target_index == label)], target_t[(output_target_index == label)], reduction='batchmean') * (self.temperature ** 2) # # -(target_t[(output_target_index == label)] * torch.log(x_m[(output_target_index == label)])).sum() / target.size()[0] #
    
    
    
    
    
class GDRNet_MASK_SIAM(Algorithm):
    def __init__(self, num_classes, cfg):
        super(GDRNet_MASK_SIAM, self).__init__(num_classes, cfg)
        
        self.cfg = cfg
        self.network = models.get_net(cfg)
        # self.classifier = models.get_classifier(self.network.out_features(), cfg)
        self.classifier = models.get_classifier(self.network.n_outputs, cfg)

        self.classifier1 = models.get_classifier(self.network.n_outputs, cfg)
        self.classifier2 = models.get_classifier(self.network.n_outputs, cfg)
        # self.classifier3 = models.get_classifier(self.network.n_outputs3, cfg)


        # self.network_ema = deepcopy(self.network)
        # self.classifier_ema = deepcopy(self.classifier)

        self.model = nn.Sequential(self.network, self.classifier)
        # self.model_ema = nn.Sequential(self.network_ema, self.classifier_ema)

        
        
        if cfg.BACKBONE == 'resnet34' or cfg.BACKBONE =='resnet18':
            dim_in1=512 
            feat_dim1=512  ## Resnet 18/34
        else:
            dim_in1=2048 ## Resnet 50/101
            feat_dim1=512

        
        
        
        # # #         # build a 3-layer projector
        self.projector = nn.Sequential(nn.Linear(dim_in1, dim_in1, bias=False), 
                                       nn.BatchNorm1d(dim_in1),
                                    nn.ReLU(inplace=True), 
                                    nn.Linear(dim_in1, feat_dim1, bias=False), 
                                    nn.BatchNorm1d(feat_dim1),
                                    nn.ReLU(inplace=True), 
                                    nn.Linear(feat_dim1, dim_in1, bias=False), 
                                    nn.BatchNorm1d(dim_in1, affine=False)
                                    )

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim_in1, feat_dim1, bias=False),
                                        nn.BatchNorm1d(feat_dim1),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(feat_dim1, dim_in1)
                                        ) # output layer

        
        init_weights_MLPHead(self.projector, init_method='He')
        init_weights_MLPHead(self.predictor, init_method='He')


        # self.optimizer = torch.optim.SGD(
        self.optimizer = torch.optim.AdamW(
            [{"params":self.network.parameters(), 'fix_lr': False},
            {"params":self.classifier.parameters(), 'fix_lr': True},
            {"params":self.classifier1.parameters(), 'fix_lr': True},
            {"params":self.classifier2.parameters(), 'fix_lr': True},
            # {"params":self.classifier3.parameters(), 'fix_lr': True},
            {"params":self.projector.parameters(), 'fix_lr': True},
            {"params":self.predictor.parameters(), 'fix_lr': True},
            ],
            lr = cfg.LEARNING_RATE,
            # momentum = cfg.MOMENTUM,
            weight_decay = cfg.WEIGHT_DECAY,
            # nesterov=True
            )
    

        # self.optimizer1 = torch.optim.SGD([
        # # self.optimizer = torch.optim.Adam(
        #     # [{"params":self.network.parameters(), 'fix_lr': False},
        #     # {"params":self.classifier.parameters(), 'fix_lr': False},
        #     # {"params":self.projector.parameters(), 'fix_lr': False},
        #     {"params":self.predictor.parameters(), 'fix_lr': True}
        #     ],
        #     lr = cfg.LEARNING_RATE*5,
        #     momentum = cfg.MOMENTUM,
        #     weight_decay = cfg.WEIGHT_DECAY,
        #     nesterov=True)
        
                                    
        K= 1024 # 1536 # 1024
        dim= 2048 # 2048
        self.K=K
        self.num_positive = 0

        # create queue for keeping neighbor
        self.register_buffer("queue", torch.randn(K, dim))
        self.queue = nn.functional.normalize(self.queue, dim=-1)
        self.register_buffer("queue_labels", -torch.ones(K, dtype=torch.long))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        self.fundusAug = get_post_FundusAug(cfg)
        self.alpha = 4.0
        self.beta = 2.0
        
        if self.cfg.FASTMOCO > 0:
            self.split_num = 2
            self.combs = 3
            self.criterion =  DahLoss_Siam_Fastmoco(cfg=self.cfg, beta= cfg.GDRNET.BETA, max_iteration = cfg.EPOCHS, \
                                    training_domains = cfg.DATASET.SOURCE_DOMAINS, temperature = cfg.GDRNET.TEMPERATURE, \
                                    scaling_factor = cfg.GDRNET.SCALING_FACTOR, fastmoco=cfg.FASTMOCO)
        else:
            self.criterion =  DahLoss_Siam(beta= cfg.GDRNET.BETA, max_iteration = cfg.EPOCHS, \
                                    training_domains = cfg.DATASET.SOURCE_DOMAINS, temperature = cfg.GDRNET.TEMPERATURE, \
                                    scaling_factor = cfg.GDRNET.SCALING_FACTOR)
                                    
        self.hp_transform = transforms.Compose([HighPassFilter(kernel_size=(11,11), sigma=5)])

                        
        self.hfc_filter_in = FourierButterworthHFCFilter(butterworth_d0_ratio=0.003, butterworth_n=1,
                                                                            do_median_padding=False, image_size=(256, 256)).cuda()
            
            
            
        # self.mask_freq=FrequencyMaskGenerator_Tensor(ratio=0.3, band='all')
        
        
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
        
    def patchify(self, imgs, block=32):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = block
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        n = imgs.shape[0]
        x = imgs.reshape(shape=(n, 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))

        return x
    
    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = 32
        h = w = int(x.shape[1] ** .5)
        n = x.shape[0]
        assert h * w == x.shape[1]

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
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        # Normalize mask_noise to [0, 1]
        noise = torch.rand([N, L], device=x.device)

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # Zero-out the masked regions
        x_masked = x * mask.unsqueeze(-1)
        x_masked_c =  x * (1 - mask.unsqueeze(-1))

        return x_masked, x_masked_c



    @torch.no_grad()
    def dequeue_and_enqueue(self, features, labels):
        # gather features and labels before updating queue
        # features = concat_all_gather(features)
        # labels = concat_all_gather(labels)

        batch_size = features.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0 # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr : ptr+batch_size, :] = features
        self.queue_labels[ptr : ptr+batch_size] = labels
        ptr = (ptr + batch_size) % self.K  # move pointer

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
        
        # img_tensor_new_freq = hfc_mul_mask(self.hfc_filter_in, img_tensor_new, mask_tensor_new, do_norm=True)
 
        img_tensor_new = img_tensor_new * mask_tensor_new
        
        # img_tensor_new_freq = hfc_mul_mask(self.hfc_filter_in, img_tensor_new, mask_tensor_new, do_norm=True )

        img_tensor_new = fundusAug['post_aug2'](img_tensor_new)
        img_tensor_ori = fundusAug['post_aug2'](img_tensor)
        # img_tensor_new_freq  = fundusAug['post_aug2'](img_tensor_new_freq)

        return img_tensor_new, img_tensor_ori #, img_tensor_new_freq
    
    
    def img_process_freq(self, img_tensor, mask_tensor, fundusAug):
        
        
        
        # img_tensor_new, mask_tensor_new = fundusAug['post_aug1'](img_tensor.clone(), mask_tensor.clone())
        # img_tensor_new = img_tensor_new * mask_tensor_new
        img_tensor_new = fundusAug['post_aug2'](img_tensor)
        # img_tensor_ori = fundusAug['post_aug2'](img_tensor)

        return img_tensor_new #, img_tensor_ori

    
    
    def ShufflePatchMixOverlap_all(self, imgs, block=32):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        _, _, h, w = imgs.shape
        # 1) Randomly skip
        patch_height_options = [7, 14, 28, 56, 112]
        random_number = np.random.randint(1, len(patch_height_options))
        
        patch_height = patch_height_options[random_number]  ## 随机选取不同的patch块
        patch_width = patch_height_options[random_number] 
        overlap = int(patch_height/7)

        # 3) Compute # of patches in each dimension
        num_patches_h = h // patch_height
        num_patches_w = w // patch_width
        

        # 4) Precompute a large feather mask for the largest possible patch 
        #    (patch_height + 2*overlap x patch_width + 2*overlap)
        #    We'll slice it down for boundary patches if needed
        
        
        feather_mask_full = create_feather_mask(
            patch_height + 2*overlap,
            patch_width  + 2*overlap,
            feather_size=overlap
        )
    
        # 5) Extract patches
        patches = []
        coords = []
        for i_patch in range(num_patches_h):
            for j_patch in range(num_patches_w):
                start_h, end_h, start_w, end_w = edgelogic(
                    i_patch, j_patch,
                    patch_height, patch_width,
                    num_patches_h, num_patches_w,
                    overlap
                )
                # Clip to image boundaries if needed
                start_h = max(0, start_h)
                start_w = max(0, start_w)
                end_h   = min(h, end_h)
                end_w   = min(w, end_w)

                patch = imgs[:, :, start_h:end_h, start_w:end_w]
                patches.append(patch)
                coords.append((start_h, end_h, start_w, end_w))
        
        
        # 6) Shuffle & mix patches
        N = len(patches)
        indices = np.random.permutation(N)
        mixed_patches = []
        for i_patch in range(N):
            lam = np.random.beta(self.alpha, self.beta)
            patchA = patches[i_patch]
            patchB = patches[indices[i_patch]]
            print(patchA.shape, patchB.shape)  ## torch.Size([32, 3, 72, 72]) torch.Size([32, 3, 72, 72])
            mixed_patch = lam * patchA + (1 - lam) * patchB
            mixed_patches.append(mixed_patch)


        # 7) Prepare output & weight arrays for soft blending
        output = np.zeros_like(imgs.cpu(), dtype=np.float32)
        weight = np.zeros((h, w), dtype=np.float32)

        # 8) Blend each patch with the feather mask
        for i_patch, (sh, eh, sw, ew) in enumerate(coords):
            patch_mixed = mixed_patches[i_patch]
            _,_, ph, pw = patch_mixed.shape

            # Extract the corresponding portion of the big feather mask
            # if patch is smaller near boundaries, slice the mask
            # mask_2d = feather_mask_full[:ph, :pw]
            
            mask_3d = feather_mask_full[ :ph, :pw]

            # # Convert mask to 3 channels if needed
            # if c == 1:
            #     mask_3d = mask_2d[..., None]
            # else:
            #     mask_3d = np.repeat(mask_2d[..., None], c, axis=2)

            # Feathered patch
            patch_feathered = patch_mixed[:,:, ...].cpu() * mask_3d

            # Accumulate in output
            output[:, :, sh:eh, sw:ew] += patch_feathered
            weight[:, :, sh:eh, sw:ew] += mask_3d

        # 9) Final divide by weight -> smooth blend
        weight = np.clip(weight, 1e-8, None)
        output /= weight[..., None]

        final_output = imgs
        final_output[:,:, 1:-1,1:-1] = output[:,:,1:-1,1:-1]
        # 10) Convert back to uint8
        # final_output = np.clip(final_output, 0, 255).astype(np.uint8)
        return final_output
    

    
    
    def patchify(self, imgs, block=32):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = block
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        n = imgs.shape[0]
        x = imgs.reshape(shape=(n, 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)  ### 
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))  ### 分块


        return x
    
    def unpatchify(self, x,block=32):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = block
        h = w = int(x.shape[1] ** .5)
        n = x.shape[0]
        assert h * w == x.shape[1]

        x = x.reshape(shape=(n, h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(n, 3, h * p, h * p))

        return imgs


    def random_patch_mix(self, x, mask_ratio):

        N, L, D = x.shape

        x= x.reshape(shape=(N, L, int(D/3), 3))  ### 分块

        # 6) Shuffle & mix patches
        # N = len(patches)
        # indices = np.random.permutation(D)
        
        index = torch.randperm(int(D/3)).cuda()
        
        lam = np.random.beta(self.alpha, self.beta)
        # patchA = x[:,:, i_patch]
        patchA = x
        patchB = x[:,:, index,:]
        # print(patchA.shape, patchB.shape)  ## torch.Size([32, 3, 72, 72]) torch.Size([32, 3, 72, 72])
        mixed_patch = lam * patchA + (1 - lam) * patchB
        mixed_patch=mixed_patch.reshape(shape=(N, L, D))  ### 分块
        # mixed_patches.append(mixed_patch)
        return mixed_patch
        
        # index = torch.randperm(image_new.size(0)).cuda()
        # lam = np.random.beta(1.0, 1.0)
        # x_a = image_new
        # targets_a = label
        # x_b = copy.deepcopy(image_new[index,:])
            
            
        # mixed_patches = []
        # for i_patch in range(D):
        #     lam = np.random.beta(self.alpha, self.beta)
        #     patchA = x[:,:, i_patch]
        #     patchB = x[:,:, indices[i_patch]]
        #     print(patchA.shape, patchB.shape)  ## torch.Size([32, 3, 72, 72]) torch.Size([32, 3, 72, 72])
        #     mixed_patch = lam * patchA + (1 - lam) * patchB
        #     mixed_patches.append(mixed_patch)
            

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        # Normalize mask_noise to [0, 1]
        noise = torch.rand([N, L], device=x.device)

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # Zero-out the masked regions
        x_masked = x * mask.unsqueeze(-1)

        return x_masked
    
    def _local_split(self, x):     # NxCxHxW --> 4NxCx(H/2)x(W/2)
        _side_indent = x.size(2) // self.split_num, x.size(3) // self.split_num
        cols = x.split(_side_indent[1], dim=3)
        xs = []
        for _x in cols:
            xs += _x.split(_side_indent[0], dim=2)
        x = torch.cat(xs, dim=0)
        return x
    
    
    
    def update(self, minibatch):
        
        amp_grad_scaler = GradScaler()

        if self.cfg.TRANSFORM.FREQ:
            image, image_feq,  mask, label, domain = minibatch
        else:
            image, mask, label, domain = minibatch

        # expert = torch.zeros(all_y.shape[0], 
        # self.num_classes).to('cuda')

        # print(image.shape)  ## torch.Size([32, 3, 256, 256])
        self.optimizer.zero_grad()
        # self.optimizer1.zero_grad()

        image_new, image_ori = self.img_process(image, mask, self.fundusAug)


                    
        with autocast():  #### 
            
                # for i in range(len(z1_splits)):
            features_ori = self.network(image_ori)
            features_ori_z1 = self.projector(features_ori[3])  ## z1
            features_new = self.network(image_new)
            features_new_z2 = self.projector(features_new[3])
            
            
            
            p1, p2 = self.predictor(features_ori_z1), self.predictor(features_new_z2) 

            features_ori_z1, features_new_z2 = nn.functional.normalize(features_ori_z1, dim=-1), nn.functional.normalize(features_new_z2, dim=-1)
            p1, p2 = nn.functional.normalize(p1, dim=-1), nn.functional.normalize(p2, dim=-1)
            
            # sample supervised targets
            z1_sup = self.sample_target(features_ori_z1.detach(), label)
            z2_sup = self.sample_target(features_new_z2.detach(), label)
            
            output_new = self.classifier(features_new[3])
            output_ori = self.classifier(features_ori[3])
            
            output_new1 = self.classifier1(features_new[3])
            output_new2 = self.classifier2(features_new[3])
            # output_new3 = self.classifier3(features_new[2])

            # output_new_ema = self.model_ema(image_new)

            image_new_masked_ = self.patchify(image_new, block=self.cfg.BLOCK)

            
            image_new_masked = self.random_masking(image_new_masked_, self.cfg.MASK_RATIO)
            image_new_masked = self.unpatchify(image_new_masked, block=self.cfg.BLOCK)
            # else:
            
            
            # features_new_masked = self.network(image_new_masked,drop_rate=0.5)
            features_new_masked = self.network(image_new_masked)
            # features_new_masked_freq = self.network(image_new_feq)
            # print(features_new_masked_freq)


            output_new_masked = self.classifier(features_new_masked[3])
            # output_new_patch_mix = self.classifier(features_new_patch_mix)

            output_new_masked1 = self.classifier1(features_new_masked[3])
            output_new_masked2 = self.classifier2(features_new_masked[3])
            


            # dequeue and enqueue
            # self.dequeue_and_enqueue(features_new_z2.detach(), label)
                        
            self.dequeue_and_enqueue(features_ori_z1.detach(), label)

            if self.cfg.FASTMOCO > 0:
        
        
                # # if self.cfg.MIXUP:
                noise_std=0.05
                index = torch.randperm(image_new.size(0)).cuda()
                lam = np.random.beta(1.0, 1.0)
                x_a = image_new
                targets_a = label
                x_b = copy.deepcopy(image_new[index,:])
                targets_b = copy.deepcopy(label[index])
                x_mixed = x_a * lam + x_b * (1-lam)
                features_x_mixed = self.network(x_mixed)
                features_x_a=self.network(x_a)
                features_x_b=self.network(x_b)
                mix_z = features_x_a[3] * lam  + features_x_b[3] * (1-lam)
                mix_z_ = mix_z + torch.normal(mean = 0., std = noise_std, size= (mix_z.size())).cuda()
                
                mix_result = self.classifier(mix_z) 
                + self.classifier(features_x_mixed[3])
            

                x1_in_form = self._local_split(image_ori)   
                x2_in_form = self._local_split(image_new)
                
                
                # print(image_new.shape, x2_in_form.shape)
                z1_pre = self.network(x1_in_form)
                z2_pre = self.network(x2_in_form)

                z1_splits = list(z1_pre[3].split(z1_pre[3].size(0) // self.split_num ** 2, dim=0))  # 4b x c x
                z2_splits = list(z2_pre[3].split(z2_pre[3].size(0) // self.split_num ** 2, dim=0))

                z1_orthmix_ = torch.cat(list(map(lambda x: sum(x) / self.combs, list(combinations(z1_splits, r=self.combs)))), dim=0) # 6 of 2combs / 4 of 3combs
                z2_orthmix_ = torch.cat(list(map(lambda x: sum(x) / self.combs, list(combinations(z2_splits, r=self.combs)))), dim=0) # 6 of 2combs / 4 of 3combs

                # print(x2_in_form.shape, z1_pre.shape,z1_orthmix.shape)  ## torch.Size([64, 3, 112, 112]) torch.Size([64, 2048]) torch.Size([96, 2048])
                z1_orthmix_ = self.projector(z1_orthmix_)
                z2_orthmix_ = self.projector(z2_orthmix_)
                    
                p1_orthmix_, p2_orthmix_ = self.predictor(z1_orthmix_), self.predictor(z2_orthmix_)
                
                z1_orthmix = z1_orthmix_.split(image_ori.size(0), dim=0)
                z2_orthmix = z2_orthmix_.split(image_new.size(0), dim=0)

                p1_orthmix = p1_orthmix_.split(image_ori.size(0), dim=0)
                p2_orthmix = p2_orthmix_.split(image_new.size(0), dim=0)
                
                # print(len(p1_orthmix),p1_orthmix[0].shape, z1_pre.shape)  ## 6 torch.Size([64, 2048]) torch.Size([64, 2048]) batch_size=16
                    
                    
                # z1_orthmix, z2_orthmix = nn.functional.normalize(z1_orthmix, dim=-1), nn.functional.normalize(z2_orthmix, dim=-1)
                # p1_orthmix, p2_orthmix = nn.functional.normalize(p1_orthmix, dim=-1), nn.functional.normalize(p2_orthmix, dim=-1)
            
                # loss, loss_dict_iter = self.criterion([output_new, output_new_masked], [features_ori_z1, features_new_z2, z1_sup, z2_sup, p1, p2, z1_orthmix, z2_orthmix,p1_orthmix, p2_orthmix], label, domain)
                loss, loss_dict_iter = self.criterion([output_new, output_ori, output_new_masked], [features_ori, features_new,features_ori_z1, features_new_z2, z1_sup, z2_sup, p1, p2, z1_orthmix, z2_orthmix, p1_orthmix, p2_orthmix, features_x_mixed, mix_z_, mix_result,targets_b,lam],  label, domain)

                temperature=2.0
    
                tau1, tau2,  tau=-0.3, 1.5, 0.7


                # self.SupLoss3 = LogitAdjust(self.label_num_dict[self.cfg.DATASET.SOURCE_DOMAINS[0]],tau=1.0)    #### 使用逻辑补偿loss
                self.SupLoss2 = LogitAdjust(self.label_num_dict[self.cfg.DATASET.SOURCE_DOMAINS[0]],tau=tau2)    #### 使用逻辑补偿loss
                self.SupLoss1 = LogitAdjust(self.label_num_dict[self.cfg.DATASET.SOURCE_DOMAINS[0]],tau=tau1)    #### 使用逻辑补偿loss
                self.SupLoss = LogitAdjust(self.label_num_dict[self.cfg.DATASET.SOURCE_DOMAINS[0]],tau=tau)    #### 使用逻辑补偿loss

                TEMP=0.5
                # loss += TEMP * self.SupLoss3(output_new3, label)
                loss += TEMP * self.SupLoss2(output_new2, label)
                loss += TEMP * self.SupLoss1(output_new1, label)

                loss += TEMP * self.SupLoss(output_new, label)

                
                
                # teacher_expert1_softmax = F.softmax((output_new1) / temperature, dim=1).detach()
                
                

                # loss += - 0.5  * (F.softmax(output_new / temperature, 1).detach() * F.log_softmax(output_new1 / temperature, 1)).sum() / output_new1.size()[0]
                # loss += - 0.5  * (F.softmax(output_new / temperature, 1).detach() * F.log_softmax(output_new2 / temperature, 1)).sum() / output_new2.size()[0]
                # loss += - 0.5  * (F.softmax(output_new / temperature, 1).detach() * F.log_softmax(output_new3 / temperature, 1)).sum() / output_new3.size()[0]

                # loss += - 0.5  * (F.softmax(output_new / temperature, 1).detach() * F.log_softmax(output_new_masked1 / temperature, 1)).sum() / output_new1.size()[0]
                # loss += - 0.5  * (F.softmax(output_new / temperature, 1).detach() * F.log_softmax(output_new_masked2 / temperature, 1)).sum() / output_new2.size()[0]
                # loss += - 0.5  * (F.softmax(output_new / temperature, 1).detach() * F.log_softmax(output_new_masked3 / temperature, 1)).sum() / output_new3.size()[0]



                # loss += - 0.5  * (F.softmax(output_new / temperature, 1).detach() * F.log_softmax(output_new1 / temperature, 1)).sum() / output_new1.size()[0]
                # loss += - 0.5  * (F.softmax(output_new / temperature, 1).detach() * F.log_softmax(output_new2 / temperature, 1)).sum() / output_new2.size()[0]
                # loss += - 0.5  * (F.softmax(output_new / temperature, 1).detach() * F.log_softmax(output_new3 / temperature, 1)).sum() / output_new3.size()[0]

                # loss += 1.0 * self.SupLoss1(output_new_masked1, F.softmax( output_new / temperature, 1).detach())
                # loss += 1.0 * self.SupLoss2(output_new_masked2, F.softmax( output_new / temperature, 1).detach())
                # loss += 1.0 * self.SupLoss3(output_new_masked3, F.softmax( output_new / temperature, 1).detach())

                # self.SupLoss3 = LogitAdjust_KD(self.label_num_dict[self.cfg.DATASET.SOURCE_DOMAINS[0]],tau=1.0)    #### 使用逻辑补偿loss
                self.SupLoss2 = LogitAdjust_KD(self.label_num_dict[self.cfg.DATASET.SOURCE_DOMAINS[0]],tau=tau2)    #### 使用逻辑补偿loss
                self.SupLoss1 = LogitAdjust_KD(self.label_num_dict[self.cfg.DATASET.SOURCE_DOMAINS[0]],tau=tau1)    #### 使用逻辑补偿loss
                self.SupLoss = LogitAdjust_KD(self.label_num_dict[self.cfg.DATASET.SOURCE_DOMAINS[0]],tau=tau)    #### 使用逻辑补偿loss

                # loss += TEMP * self.SupLoss1(output_new1, output_new.detach())
                # loss += TEMP * self.SupLoss2(output_new2, output_new.detach())
                # loss += TEMP * self.SupLoss3(output_new3, output_new.detach())   

                

                self.SupLoss2 = LogitAdjust_KD_V2(self.label_num_dict[self.cfg.DATASET.SOURCE_DOMAINS[0]],tau1=tau2,tau2=tau2)    #### 使用逻辑补偿loss
                self.SupLoss1 = LogitAdjust_KD_V2(self.label_num_dict[self.cfg.DATASET.SOURCE_DOMAINS[0]],tau1=tau1,tau2=tau1)    #### 使用逻辑补偿loss
                self.SupLoss = LogitAdjust_KD_V2(self.label_num_dict[self.cfg.DATASET.SOURCE_DOMAINS[0]],tau1=tau,tau2=tau)    #### 使用逻辑补偿loss


                
                TEMP=0.5 * min( self.epoch / 10, 1.0)

                loss += TEMP * self.SupLoss2(output_new_masked2, output_new2, label)
                loss += TEMP * self.SupLoss1(output_new_masked1, output_new1, label)
                loss += TEMP * self.SupLoss(output_new_masked, output_new, label)

                # loss += TEMP * self.SupLoss1(output_new_masked1, output_new1.detach())
                # loss += TEMP * self.SupLoss2(output_new_masked2, output_new2.detach())
                # # loss += TEMP * self.SupLoss3(output_new_masked3, output_new3.detach())   
                # loss += TEMP * self.SupLoss(output_new_masked, output_new.detach())

                # loss += 0.5 * JS_Divergence(output_new, output_new1,output_new2, output_new3)

            else:
                loss, loss_dict_iter = self.criterion([output_new, output_new_masked], [features_ori_z1, features_new_z2, z1_sup, z2_sup, p1, p2], label, domain)

                # loss, loss_dict_iter = self.criterion([output_new1, output_new2], [features_ori1, features_new1, features_ori2, features_new2], label, domain)

        # loss.backward()
        # self.optimizer.step()
        amp_grad_scaler.scale(loss).backward()
        amp_grad_scaler.step(self.optimizer)
        # amp_grad_scaler.step(self.optimizer1)
        amp_grad_scaler.update()

        # iters = epoch * len(trainloader_u) + i
        # ema_ratio = min(1 - 1 / (iters + 1), 0.996)
        
        # for param, param_ema in zip(model.parameters(), model_ema.parameters()):
        #     param_ema.copy_(param_ema * ema_ratio + param.detach() * (1 - ema_ratio))
                
                
                
        return loss_dict_iter
    


    def update_ema_model(self, iters):
        # self.epoch = epoch
        # iters = epoch * len(trainloader_u) + i
        ema_ratio = min(1 - 1 / (iters + 1), 0.996)
        
        for param, param_ema in zip(self.model.parameters(), self.model_ema.parameters()):
            param_ema.copy_(param_ema * ema_ratio + param.detach() * (1 - ema_ratio))
            
        # return self.criterion.update_alpha(epoch)

    def update_epoch(self, epoch):
        self.epoch = epoch
        return self.criterion.update_alpha(epoch)


    def validate(self, val_loader, test_loader, writer):
        val_auc = -1
        test_auc = -1
        if self.epoch <= self.cfg.EPOCHS:
            val_auc, val_loss = algorithm_validate_distill_v2(self, val_loader, writer, self.epoch, 'val')
            test_auc, test_loss = algorithm_validate_distill_v2(self, test_loader, writer, self.epoch, 'test')
            if self.epoch == self.cfg.EPOCHS:
                self.epoch += 1
        else:
            test_auc, test_loss = algorithm_validate_distill_v2(self, test_loader, writer, self.cfg.EPOCHS + self.cfg.VAL_EPOCH, 'test')
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
        
        # return self.classifier(self.network(x)[3])
    
        features= self.network(x)
        output = self.classifier(features[3])
        output1 = self.classifier1(features[3])
        output2 = self.classifier2(features[3])
        # output3 = self.classifier3(features[2])

        return [output, output1, output2]
    

def JS_Divergence(output_clean, output_aug1, output_aug2, output_aug3):
    p_clean, p_aug1 = F.softmax(output_clean, dim=1), F.softmax(output_aug1, dim=1)
    p_aug2, p_aug3 = F.softmax(output_aug2, dim=1), F.softmax(output_aug3, dim=1)
    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2 + p_aug3) / 4., 1e-7, 1).log()
    # loss_ctr = (F.kl_div(p_mixture, p_clean, reduction='batchmean') + F.kl_div(p_mixture, p_aug1,
    #                                                                            reduction='batchmean') + F.kl_div(
    #     p_mixture, p_aug2, reduction='batchmean') + F.kl_div(p_mixture, p_aug3, reduction='batchmean')) / 4.

    # loss_ctr = (F.kl_div(p_mixture, p_aug1, reduction='batchmean') + F.kl_div(
    #     p_mixture, p_aug2, reduction='batchmean') + F.kl_div(p_mixture, p_aug3, reduction='batchmean')) / 3.
    loss_ctr = (F.kl_div(p_mixture, p_clean, reduction='batchmean') )
    
    return loss_ctr


class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.s = 30

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return self.s * out