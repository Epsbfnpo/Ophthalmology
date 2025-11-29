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
from utils.validate import algorithm_validate, algorithm_validate_class, algorithm_eval_mu, algorithm_eval_tsne, algorithm_eval_heat
import modeling.model_manager as models
from modeling.losses import DahLoss,DahLoss_Dual, DahLoss_Dual_BalSCL,DahLoss_Mask, DahLoss_Dual_Siam, DahLoss_Siam,DahLoss_Siam_Fastmoco,DahLoss_Siam_Fastmoco_v0
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

from sam import SAM
from grams import Grams

def adjust_rho(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    epoch = epoch + 1
    if args.rho_schedule == 'step':
      if epoch <= 5:
          rho = args.rho_steps[0]
      elif epoch > 75:
          rho = args.rho_steps[3]
      elif epoch > 60:
          rho = args.rho_steps[2]

      else:
          rho = args.rho_steps[1]
      for param_group in optimizer.param_groups:
          param_group['rho'] = rho
    if args.rho_schedule == 'linear':
      X = [1, args.epochs]
      Y = [args.min_rho, args.max_rho]
      y_interp = interp1d(X, Y)
      rho = y_interp(epoch)

      for param_group in optimizer.param_groups:

          param_group['rho'] = np.float16(rho)
    if args.rho_schedule == 'none':
      rho = args.rho
      for param_group in optimizer.param_groups:
          param_group['rho'] = rho
          
ALGORITHMS = [
    'ERM',
    'GDRNet',
    'GDRNet_DUAL',
    'GDRNet_MASK_SIAM',   ####自己的方法
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

# Our method
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


    def img_process_freq(self, img_tensor, mask_tensor, fundusAug):
        
        img_tensor_new, mask_tensor_new = fundusAug['post_aug1'](img_tensor.clone(), mask_tensor.clone())
        img_tensor_new = img_tensor_new * mask_tensor_new
        img_tensor_new = fundusAug['post_aug2'](img_tensor_new)
        # img_tensor_ori = fundusAug['post_aug2'](img_tensor)

        return img_tensor_new #, img_tensor_ori
    
    

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
    
    
class GDRNet_MASK_SIAM(Algorithm):
    def __init__(self, num_classes, cfg):
        super(GDRNet_MASK_SIAM, self).__init__(num_classes, cfg)
        
        self.cfg = cfg
        self.network = models.get_net(cfg)
        if cfg.BACKBONE == 'resnet50' or cfg.BACKBONE == 'resnet101':
            self.classifier = models.get_classifier(self.network.out_features(), cfg)
        elif cfg.BACKBONE == 'resnet50_distill':
            self.classifier = models.get_classifier(self.network.n_outputs, cfg)

        # self.classifier1 = models.get_classifier(self.network.n_outputs1, cfg)
        # self.classifier2 = models.get_classifier(self.network.n_outputs2, cfg)
        # self.classifier3 = models.get_classifier(self.network.n_outputs3, cfg)


        self.network_ema = deepcopy(self.network)
        self.classifier_ema = deepcopy(self.classifier)

        self.model = nn.Sequential(self.network, self.classifier)
        self.model_ema = nn.Sequential(self.network_ema, self.classifier_ema)

        
        
        if cfg.BACKBONE == 'resnet34' or cfg.BACKBONE =='resnet18':
            dim_in1=512 
            feat_dim1=512  ## Resnet 18/34
        else:
            dim_in1=2048 ## Resnet 50/101
            feat_dim1=512

        # # # # #         # build a 3-layer projector
        self.projector = nn.Sequential(nn.Linear(dim_in1, feat_dim1, bias=False), 
                                    #    nn.BatchNorm1d(feat_dim1),
                                    nn.LeakyReLU(inplace=True), 
                                    nn.Linear(feat_dim1, dim_in1, bias=False),
                                    # nn.BatchNorm1d(feat_dim1),
                                    # nn.ReLU(inplace=True), 
                                    # nn.Linear(feat_dim1, dim_in1, bias=False), 
                                    # nn.BatchNorm1d(dim_in1, affine=False)
                                    )

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim_in1, dim_in1, bias=False),
                                        # nn.BatchNorm1d(feat_dim1),
                                        # nn.ReLU(inplace=True), # hidden layer
                                        # nn.Linear(feat_dim1, dim_in1)
                                        ) # output layer


        #         # build a 3-layer projector
        # self.projector = nn.Sequential(nn.Linear(dim_in1, feat_dim1, bias=False), 
        #                             #    nn.BatchNorm1d(dim_in1),
        #                             nn.ReLU(inplace=True), 
        #                             nn.Linear(feat_dim1, feat_dim1, bias=False), 
        #                             # nn.BatchNorm1d(feat_dim1),
        #                             # nn.ReLU(inplace=True), 
        #                             # nn.Linear(feat_dim1, dim_in1, bias=False), 
        #                             # nn.BatchNorm1d(dim_in1, affine=False)
        #                             )

        # # build a 2-layer predictor
        # self.predictor = nn.Sequential(nn.Linear(feat_dim1, feat_dim1, bias=False),
        #                                 # nn.BatchNorm1d(feat_dim1),
        #                                 # nn.ReLU(inplace=True), # hidden layer
        #                                 # nn.Linear(feat_dim1, dim_in1)
        #                                 ) # output layer
        
        
        
        # #         # build a 3-layer projector
        # self.projector = nn.Sequential(nn.Linear(dim_in1, dim_in1, bias=False), 
        #                                nn.BatchNorm1d(dim_in1),
        #                             nn.LeakyReLU(0.2, True),
        #                             nn.Linear(dim_in1, feat_dim1, bias=False), 
        #                             nn.BatchNorm1d(feat_dim1),
        #                             nn.LeakyReLU(0.2, True),
        #                             nn.Linear(feat_dim1, dim_in1, bias=False), 
        #                             nn.BatchNorm1d(dim_in1, affine=False)
        #                             )

        # # build a 2-layer predictor
        # self.predictor = nn.Sequential(nn.Linear(dim_in1, feat_dim1, bias=False),
        #                                 nn.BatchNorm1d(feat_dim1),
        #                                 nn.ReLU(inplace=True), # hidden layer
        #                                 nn.Linear(feat_dim1, dim_in1)
        #                                 ) # output layer
        
        
        
        
        init_weights_MLPHead(self.projector, init_method='He')
        init_weights_MLPHead(self.predictor, init_method='He')


        # self.optimizer = torch.optim.SGD(
        # # self.optimizer = torch.optim.Adam(
        #     [{"params":self.network.parameters(), 'fix_lr': False},
        #     {"params":self.classifier.parameters(), 'fix_lr': False},
        #     # {"params":self.classifier1.parameters(), 'fix_lr': True},
        #     # {"params":self.classifier2.parameters(), 'fix_lr': True},
        #     # {"params":self.classifier3.parameters(), 'fix_lr': True},
        #     {"params":self.projector.parameters(), 'fix_lr': False},
        #     {"params":self.predictor.parameters(), 'fix_lr': True},
        #     ],
        #     lr = cfg.LEARNING_RATE,
        #     momentum = cfg.MOMENTUM,
        #     weight_decay = cfg.WEIGHT_DECAY,
        #     nesterov=True,
        #     )
        
        
        # self.optimizer = torch.optim.SGD(
        self.optimizer = torch.optim.Adam(
        # self.optimizer = Grams(
            [{"params":self.network.parameters(), 'fix_lr': False},
            {"params":self.classifier.parameters(), 'fix_lr': False},
            # {"params":self.classifier1.parameters(), 'fix_lr': True},
            # {"params":self.classifier2.parameters(), 'fix_lr': True},
            # {"params":self.classifier3.parameters(), 'fix_lr': True},
            # {"params":self.random_matrix.parameters(), 'fix_lr': False},

            {"params":self.projector.parameters(), 'fix_lr': False},
            {"params":self.predictor.parameters(), 'fix_lr': True},
            ],
            lr = cfg.LEARNING_RATE,
            # betas=(0.5, 0.99),
            # momentum = cfg.MOMENTUM,
            weight_decay = 0.0001, #cfg.WEIGHT_DECAY,
            # nesterov=True
            )
    
        # self.optimizer = SAM(base_optimizer=torch.optim.SGD, rho=0.05, 
        #                      params=[{"params":self.network.parameters(), 'fix_lr': False},
        #     {"params":self.classifier.parameters(), 'fix_lr': False},
        #     # {"params":self.classifier1.parameters(), 'fix_lr': True},
        #     # {"params":self.classifier2.parameters(), 'fix_lr': True},
        #     # {"params":self.classifier3.parameters(), 'fix_lr': True},
        #     {"params":self.projector.parameters(), 'fix_lr': False},
        #     {"params":self.predictor.parameters(), 'fix_lr': True},
        #     ], lr=cfg.LEARNING_RATE, momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)

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
        dim = 2048 #2048 #2048 # 2048  #################### 此处应与projector保持一致
        self.K=K
        self.num_positive = self.cfg.POSITIVE #4

        # create queue for keeping neighbor
        self.register_buffer("queue", torch.randn(K, dim))
        self.queue = nn.functional.normalize(self.queue, dim=-1)
        self.register_buffer("queue_labels", -torch.ones(K, dtype=torch.long))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        self.fundusAug = get_post_FundusAug(cfg)
        self.alpha = 4.0
        self.beta = 2.0
        
        self.random_matrix = None
        self.random_in_dim = 2048 #2048
        self.random_out_dim = 2048 #2048
        
        self.E_dis = nn.MSELoss()
        
        if self.cfg.FASTMOCO > -1.0:

            self.split_num = 2
            if self.cfg.DG_MODE =='DG':
                self.combs = 3
            else:
                self.combs = 4

            self.criterion =  DahLoss_Siam_Fastmoco_v0(cfg=self.cfg, beta= cfg.GDRNET.BETA, max_iteration = cfg.EPOCHS, \
                                    training_domains = cfg.DATASET.SOURCE_DOMAINS, temperature = cfg.GDRNET.TEMPERATURE, \
                                    scaling_factor = cfg.GDRNET.SCALING_FACTOR, fastmoco=cfg.FASTMOCO)
        else:
            self.criterion =  DahLoss_Siam(beta= cfg.GDRNET.BETA, max_iteration = cfg.EPOCHS, \
                                    training_domains = cfg.DATASET.SOURCE_DOMAINS, temperature = cfg.GDRNET.TEMPERATURE, \
                                    scaling_factor = cfg.GDRNET.SCALING_FACTOR)
                                    
        self.hp_transform = transforms.Compose([HighPassFilter(kernel_size=(11,11), sigma=5)])

                        
        self.hfc_filter_in = FourierButterworthHFCFilter(butterworth_d0_ratio=0.001, butterworth_n=4,
                                                                            do_median_padding=False, image_size=(256, 256)).cuda()
            
            
            
        # self.mask_freq=FrequencyMaskGenerator_Tensor(ratio=0.3, band='all')
        
        
    def change_random_matrix(self):
        random_matrix = torch.randn(self.random_in_dim, self.random_out_dim).cuda()
        self.random_matrix = random_matrix
        # if dist.is_initialized():
        #     dist.broadcast(random_matrix, src=0)

        # if dist.is_initialized() and dist.get_rank() == 0:
        #     print('change random matrix')
            
        # 伯努利分布
        # bernoulli_seed = torch.empty(self.random_in_dim, self.random_out_dim).uniform_(0, 1)
        # self.random_matrix = torch.bernoulli(bernoulli_seed).cuda()

        # [-1, 1]均匀分布
        # self.random_matrix = torch.Tensor(self.random_in_dim, self.random_out_dim).uniform_(-1,1).cuda()
        
        


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
        # print(ptr)
        # print(ptr + batch_size, self.K)  %%%% 
        ptr = (ptr + batch_size) % self.K  # move pointer


        self.queue_ptr[0] = ptr
        
    @torch.no_grad()
    def sample_target(self, features, labels):  ### 只有当前的labele
        neighbor = []
        for i, label in enumerate(labels):
            pos = torch.where(self.queue_labels == label)[0]
            if len(pos) != 0:
                if self.num_positive > 0:
                    choice = torch.multinomial(torch.ones_like(pos).type(torch.FloatTensor), self.num_positive, replacement=True)
                    idx = pos[choice]
                    neighbor.append(self.queue[idx].mean(0))
                else:
                    neighbor.append(self.queue[pos].mean(0))  ###
            else:
                neighbor.append(features[i])
        neighbor = torch.stack(neighbor, dim=0)

        return neighbor
    
    def img_process(self, img_tensor, mask_tensor, fundusAug):
        
        
        img_tensor_new, mask_tensor_new = fundusAug['post_aug1'](img_tensor.clone(), mask_tensor.clone())
        
        # img_tensor_new_freq = hfc_mul_mask(self.hfc_filter_in, img_tensor_new, mask_tensor_new, do_norm=True)
 
        img_tensor_new = img_tensor_new * mask_tensor_new
        
        
        
        # img_tensor_new_freq = hfc_mul_mask(self.hfc_filter_in, img_tensor_new, mask_tensor_new, do_norm=True )
        # img_tensor_new_freq = hfc_mul_mask(self.hfc_filter_in, img_tensor_new, mask_tensor_new, do_norm=True )

        img_tensor_new = fundusAug['post_aug2'](img_tensor_new)
        
        
        r = np.random.rand(1)

        if r > 2.0 :
            
            img_tensor_ori, mask_tensor_ori = fundusAug['post_aug1'](img_tensor.clone(), mask_tensor.clone())
            img_tensor_ori = img_tensor_ori * mask_tensor_ori
            
            img_tensor_ori = fundusAug['post_aug2'](img_tensor_ori)  #### 
            
        else: 
            img_tensor_ori = fundusAug['post_aug2'](img_tensor.clone())  #### 
            


        # img_tensor_new = fundusAug['post_aug4'](img_tensor.clone())  #### 
            
            
        # img_tensor_all, mask_tensor_all = fundusAug['post_aug1'](img_tensor.clone(), mask_tensor.clone())
        # img_tensor_all = img_tensor_all * mask_tensor_all
        
        # img_tensor_all = fundusAug['post_aug3'](img_tensor_all)
        

        # img_tensor_all = fundusAug['post_aug2'](img_tensor_new)

        # img_tensor_ori, mask_tensor_ori = fundusAug['post_aug1'](img_tensor.clone(), mask_tensor.clone())
        # img_tensor_ori = img_tensor_ori * mask_tensor_ori
        
        # img_tensor_ori = fundusAug['post_aug2'](img_tensor_ori)
        
        
        # img_tensor_new_freq  = fundusAug['post_aug2'](img_tensor_new_freq)

        return img_tensor_new, img_tensor_ori # , img_tensor_new
    
    
    def img_process_freq(self, img_tensor, mask_tensor, fundusAug):
        
        # img_tensor_new, mask_tensor_new = fundusAug['post_aug1'](img_tensor.clone(), mask_tensor.clone())
        # img_tensor_new = img_tensor_new * mask_tensor_new
        # img_tensor_new = fundusAug['post_aug2'](img_tensor)
        # img_tensor_ori = fundusAug['post_aug2'](img_tensor)


        img_tensor_freq = hfc_mul_mask(self.hfc_filter_in, img_tensor, mask_tensor, do_norm=True )


        img_tensor_freq = fundusAug['post_aug2'](img_tensor_freq)

        return img_tensor_freq #, img_tensor_ori
    
    
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
        x_masked_c = x * (1 -mask.unsqueeze(-1))

        return mask.unsqueeze(-1), (1 -mask.unsqueeze(-1))
    
    def _local_split(self, x):     # NxCxHxW --> 4NxCx(H/2)x(W/2)
        _side_indent = x.size(2) // self.split_num, x.size(3) // self.split_num
        cols = x.split(_side_indent[1], dim=3)
        xs = []
        for _x in cols:
            xs += _x.split(_side_indent[0], dim=2)
        x = torch.cat(xs, dim=0)
        return x
    
    
    def update(self, minibatch):
        

        for param in self.model_ema.parameters():
            param.detach_()   # ema_model set


        amp_grad_scaler = GradScaler()
        
        # self.change_random_matrix()   ##### 

        if self.cfg.TRANSFORM.FREQ:
            image, image_feq, image_feq2,  mask, label, domain = minibatch
        else:
            image, mask, label, domain = minibatch


        self.optimizer.zero_grad()

        image_new, image_ori = self.img_process(image, mask, self.fundusAug)
        
        
        if self.cfg.TRANSFORM.FREQ:
            image_ori = image_feq
        
            image_ori =   self.img_process_freq(image_ori, mask, self.fundusAug)   ### 
            
            

        # sample_path1 = f'./samples/image_new_crop'
        # os.makedirs(sample_path1, exist_ok=True)
        # for i, image_ in enumerate(image_new):
        #     # Move the image tensor to CUDA and save it with the index as the filename
        #     image_ = image_.to("cuda")
        #     # save_image(image, f"output/visualize/{folder_name}/image_{i}.png")
        #     save_image(image_, f"{sample_path1}/image_{i}.png")            


        # sample_path2 = f'./samples/image_ori_crop'
        # os.makedirs(sample_path2, exist_ok=True)
        # for i, image_ in enumerate(image_ori):
        #     # Move the image tensor to CUDA and save it with the index as the filename
        #     image_ = image_.to("cuda")
        #     # save_image(image, f"output/visualize/{folder_name}/image_{i}.png")
        #     save_image(image_, f"{sample_path2}/image_{i}.png") 
            
        

        # sample_path3 = f'./samples/image_new_resize'
        # os.makedirs(sample_path3, exist_ok=True)
        # for i, image_ in enumerate(image_all):
        #     # Move the image tensor to CUDA and save it with the index as the filename
        #     image_ = image_.to("cuda")
        #     # save_image(image, f"output/visualize/{folder_name}/image_{i}.png")
        #     save_image(image_, f"{sample_path3}/image_{i}.png") 
            
        
                    
        # print(image_ori.shape,self.network1(image_ori).shape )  ## torch.Size([16, 3, 224, 224]) torch.Size([16, 2048])
        with autocast():  #### 
            
            
                # for i in range(len(z1_splits)):
            features_ori = self.network(image_ori, perturb=False, distill=True)
            # print(len(features_ori))
            features_ori_z1 = self.projector(features_ori[3])  ## z1
            features_new = self.network(image_new, perturb=True, distill=True)
            features_new_z2 = self.projector(features_new[3])
            # features_all = self.network(image_all)
            # features_all_z = self.projector(features_all)sss
            
            # features_feq2 = self.network(image_feq2)
            # featur_freq2_z = self.projector(features_feq2)
            
            with torch.no_grad():

                features_new_ema = self.network_ema(image_new, perturb=True, distill=True)
                output_new_ema = self.classifier_ema(features_new_ema[3])

            p1, p2 = self.predictor(features_ori_z1), self.predictor(features_new_z2)   ####
            # p_all = self.predictor(features_all_z)
            

            
            
            # features_ori_z1, features_new_z2 = nn.functional.normalize(features_ori_z1, dim=-1), nn.functional.normalize(features_new_z2, dim=-1)
            
            # p1, p2 = nn.functional.normalize(p1, dim=-1), nn.functional.normalize(p2, dim=-1)
            

            # features_all_z = nn.functional.normalize(features_all_z, dim=-1)
            # p_all = nn.functional.normalize(p_all, dim=-1)
        
            # sample supervised targets
            z1_sup = self.sample_target(features_ori_z1.detach(), label)   ###
            z2_sup = self.sample_target(features_new_z2.detach(), label)  ### 

            # z_all_sup = self.sample_target(features_all_z.detach(), label)
            
            output_new = self.classifier(features_new[3])
            output_ori = self.classifier(features_ori[3])
            # output_all = self.classifier(features_all)
            # output_new1 = self.classifier1(features_new[0])
            # output_new2 = self.classifier2(features_new[1])
            # output_new3 = self.classifier3(features_new[2])

            # output_new_ema = self.model_ema(image_new)

            image_new_masked_ = self.patchify(image_new, block=self.cfg.BLOCK)
            image_ori_masked_c_ = self.patchify(image_ori, block=self.cfg.BLOCK)


            masked, masked_c = self.random_masking(image_new_masked_, self.cfg.MASK_RATIO)
            
            image_new_masked = self.unpatchify(masked * image_new_masked_, block=self.cfg.BLOCK)
            image_ori_masked_c = self.unpatchify(masked_c * image_ori_masked_c_, block=self.cfg.BLOCK)
            

            # image_new_masked_c = self.unpatchify(image_new_masked_c, block=self.cfg.BLOCK)


            
            
            # features_new_masked = self.network(image_new_masked,drop_rate=0.5)
            features_new_masked = self.network(image_new_masked, drop_rate=0.3, perturb=False, distill=True)
            
            # features_new_masked_freq = self.network(image_new_feq)
            # print(features_new_masked_freq)


            features_ori_masked_c = self.network(image_ori_masked_c,drop_rate=0.3)



            output_new_masked = self.classifier(features_new_masked[3])
            output_ori_masked_c = self.classifier(features_ori_masked_c)


            # output_new_patch_mix = self.classifier(features_new_patch_mix)

            
            # features_new_masked = self.network(image_new_patch_mix,drop_rate=0.3)
            # features_new_masked_freq = self.network(image_freq)


            # output_new_masked_freq = self.classifier(features_new_masked_freq)
            
            # print(output_new_masked_freq )
            
            
            # output_new_ori = self.classifier(features_ori)


            z_new_masked = self.projector(features_new_masked[3])
            z_ori_masked_c = self.projector(features_ori_masked_c)
                
            p_new_masked, p_ori_masked_c = self.predictor(z_new_masked), self.predictor(z_ori_masked_c)
                
                
            z_new_masked, z_ori_masked_c = nn.functional.normalize(z_new_masked, dim=-1), nn.functional.normalize(z_ori_masked_c, dim=-1)
            p_new_masked, p_ori_masked_c = nn.functional.normalize(p_new_masked, dim=-1), nn.functional.normalize(p_ori_masked_c, dim=-1)


            # dequeue and enqueue
            self.dequeue_and_enqueue(features_new_z2.detach(), label)
            # self.dequeue_and_enqueue(features_all_z.detach(), label)
            
            
            # self.dequeue_and_enqueue(features_ori_z1.detach(), label)

            if self.cfg.FASTMOCO > -1.0:
        
        
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
                mix_z = features_x_a * lam  + features_x_b * (1-lam)
                mix_z_ = mix_z + torch.normal(mean = 0., std = noise_std, size= (mix_z.size())).cuda()
                
                mix_result = self.classifier(mix_z) 
                + self.classifier(features_x_mixed[3])
            
        
                x1_in_form = self._local_split(image_ori)   
                x2_in_form = self._local_split(image_new)
                # x_all_in_form = self._local_split(image_all)
                
                
                # sample_path3 = f'./samples/image_ori_split'
                # os.makedirs(sample_path3, exist_ok=True)
                # for i, image_ in enumerate(x1_in_form):
                #     # Move the image tensor to CUDA and save it with the index as the filename
                #     image_ = image_.to("cuda")
                #     # save_image(image, f"output/visualize/{folder_name}/image_{i}.png")
                #     save_image(image_, f"{sample_path3}/image_{i}.png") 
        
        
                # print(image_new.shape, x2_in_form.shape)
                z1_pre = self.network(x1_in_form)
                z2_pre = self.network(x2_in_form)
                # z_all_pre = self.network(x_all_in_form)
                
                
                
                z1_splits = list(z1_pre.split(z1_pre.size(0) // self.split_num ** 2, dim=0))  # 4b x c x
                z2_splits = list(z2_pre.split(z2_pre.size(0) // self.split_num ** 2, dim=0))
                # z_all_splits = list(z_all_pre.split(z_all_pre.size(0) // self.split_num ** 2, dim=0))  # 4b x c x

                # print(z1_splits.shape)  

                z1_orthmix_ = torch.cat(list(map(lambda x: sum(x) / self.combs, list(combinations(z1_splits, r=self.combs)))), dim=0) # 6 of 2combs / 4 of 3combs
                z2_orthmix_ = torch.cat(list(map(lambda x: sum(x) / self.combs, list(combinations(z2_splits, r=self.combs)))), dim=0) # 6 of 2combs / 4 of 3combs
                # z_all_orthmix_ = torch.cat(list(map(lambda x: sum(x) / self.combs, list(combinations(z_all_splits, r=self.combs)))), dim=0) # 6 of 2combs / 4 of 3combs
                
                # 
                # 
                # (x2_in_form.shape, z1_pre.shape,z1_orthmix.shape)  ## torch.Size([64, 3, 112, 112]) torch.Size([64, 2048]) torch.Size([96, 2048])
                z1_orthmix_ = self.projector(z1_orthmix_)
                z2_orthmix_ = self.projector(z2_orthmix_)
                    
                # z_all_orthmix_ = self.projector(z_all_orthmix_)

                
                
                p1_orthmix_, p2_orthmix_ = self.predictor(z1_orthmix_), self.predictor(z2_orthmix_)
                
                # p_all_orthmix_= self.predictor(z_all_orthmix_)
                
                
                
                z1_orthmix = z1_orthmix_.split(image_ori.size(0), dim=0)
                z2_orthmix = z2_orthmix_.split(image_new.size(0), dim=0)

                p1_orthmix = p1_orthmix_.split(image_ori.size(0), dim=0)
                p2_orthmix = p2_orthmix_.split(image_new.size(0), dim=0)
                

                # z_all_orthmix = z_all_orthmix_.split(image_all.size(0), dim=0)

                # p_all_orthmix = p_all_orthmix_.split(image_all.size(0), dim=0)
                
                
                # print(len(p1_orthmix),p1_orthmix[0].shape, z1_pre.shape)  ## 6 torch.Size([64, 2048]) torch.Size([64, 2048]) batch_size=16
                    
                    
                # z1_orthmix, z2_orthmix = nn.functional.normalize(z1_orthmix, dim=-1), nn.functional.normalize(z2_orthmix, dim=-1)
                # p1_orthmix, p2_orthmix = nn.functional.normalize(p1_orthmix, dim=-1), nn.functional.normalize(p2_orthmix, dim=-1)
                
                # print(self.random_matrix)
            
                # loss, loss_dict_iter = self.criterion([output_new, output_new_masked], [features_ori_z1, features_new_z2, z1_sup, z2_sup, p1, p2, z1_orthmix, z2_orthmix,p1_orthmix, p2_orthmix], label, domain)
                loss, loss_dict_iter = self.criterion([output_new, output_ori, output_new_masked, output_ori_masked_c], [features_ori, features_new, features_ori_z1, features_new_z2, z1_sup, z2_sup, p1, p2, z1_orthmix, z2_orthmix, p1_orthmix, p2_orthmix],  [z_new_masked, z_ori_masked_c, p_new_masked, p_ori_masked_c], label, domain, self.random_matrix)


# features_ori, features_new, z1,z2, z1_sup, z2_sup, p1, p2, z1_orthmix, z2_orthmix, p1_orthmix, p2_orthmix = features


                # loss += 1.0 * self.E_dis(features_new_masked ,features_new.detach())  # features_new_masked, features_new.detach()
    # features_ori_masked_c
                # loss += 1.0 * self.E_dis(features_new_masked[2], features_new[2].detach())  # features_new_masked, features_new.detach()

                # loss += 1.0 * self.E_dis(features_new_masked , features_ori_masked_c)  # features_new_masked, features_new.detach()


                loss_feat = 0.0
                temp = self.cfg.KD
                loss_feat += temp * self.E_dis(features_new_masked[3], features_new_ema[3].detach())  # features_new_masked, features_new.detach()
                loss_feat += temp * self.E_dis(features_new_masked[2], features_new_ema[2].detach())  # features_new_masked, features_new.detach()
                loss_feat += temp * self.E_dis(features_new_masked[1], features_new_ema[1].detach())  # features_new_masked, features_new.detach()
                loss_feat += temp * self.E_dis(features_new_masked[0], features_new_ema[0].detach())  # features_new_masked, features_new.detach()

                # loss_feat += temp * self.E_dis(features_new_masked[3], features_new[3].detach())  # features_new_masked, features_new.detach()
                # loss_feat += temp * self.E_dis(features_new_masked[2], features_new[2].detach())  # features_new_masked, features_new.detach()
                # loss_feat += temp * self.E_dis(features_new_masked[1], features_new[1].detach())  # features_new_masked, features_new.detach()
                # loss_feat += temp * self.E_dis(features_new_masked[0], features_new[0].detach())  # features_new_masked, features_new.detach()


                loss += loss_feat

                if self.cfg.DG_MODE =='DG':
                    loss_dict_iter["loss_feat"] = loss_feat.item()
                # print(self.E_dis(features_new_masked[3], features_new[3].detach()))

                # loss += 1.0 * F.mse_loss(features_ori, features_new_masked_c)

    # features_ori, features_new,features_ori_z1, features_new_z2, z1_sup, z2_sup, p1, p2, z1_orthmix, z2_orthmix, p1_orthmix, p2_orthmix, features_x_mixed, mix_z_, mix_result, targets_b,lam
                # temperature=2.0

                # loss += - 0.5  * (F.softmax(output_new / temperature, 1).detach() * F.log_softmax(output_new1 / temperature, 1)).sum() / output_new1.size()[0]
                # loss += - 0.5  * (F.softmax(output_new / temperature, 1).detach() * F.log_softmax(output_new2 / temperature, 1)).sum() / output_new2.size()[0]
                # loss += - 0.5  * (F.softmax(output_new / temperature, 1).detach() * F.log_softmax(output_new3 / temperature, 1)).sum() / output_new3.size()[0]
                
                # loss += 0.5 * JS_Divergence(output_new,output_new1,output_new2, output_new3)
        
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
        # ema_ratio = min(1 - 1 / (iters + 1), 0.996)
        ema_ratio = 0.996 # 0.996 # min(1 - 1 / (iters + 1), 0.996)
        # if self.cfg.DG_MODE =='DG':
        #     ema_ratio = 0.996
        # else:
        #     ema_ratio =  min(1 - 1 / (iters + 1), 0.996)

        for param, param_ema in zip(self.model.parameters(), self.model_ema.parameters()):
            param_ema.copy_(param_ema * ema_ratio + param.detach() * (1 - ema_ratio))
            
        # return self.criterion.update_alpha(epoch)


    def update_rho(self, iters):
        # self.epoch = epoch
        # iters = epoch * len(trainloader_u) + i
        adjust_rho(self.optimizer, self.epoch)
            
            
            
    def update_epoch(self, epoch):
        self.epoch = epoch
        return self.criterion.update_alpha(epoch)


    def update_epoch_weight(self, epoch):
        self.epoch = epoch
        return self.criterion.get_weights_v2(epoch)
    
    
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
    
    def validate_class(self, val_loader, test_loader, writer):
        val_auc = -1
        test_auc = -1
        if self.epoch <= self.cfg.EPOCHS:
            val_auc, val_loss = algorithm_validate_class(self, val_loader, writer, self.epoch, 'val')
            test_auc, test_loss = algorithm_validate_class(self, test_loader, writer, self.epoch, 'test')
            if self.epoch == self.cfg.EPOCHS:
                self.epoch += 1
        else:
            test_auc, test_loss = algorithm_validate_class(self, test_loader, writer, self.cfg.EPOCHS + self.cfg.VAL_EPOCH, 'test')
            logging.info('Best performance on test domain(s): {}'.format(test_auc))
                
        return val_auc, test_auc
    


    def eval_mu(self, val_loader, test_loader):
        mu, sigma = algorithm_eval_mu(self, test_loader, self.cfg.EPOCHS + self.cfg.VAL_EPOCH, 'test')
        return mu, sigma
    
    def eval_tsne(self, val_loader, test_loader):
        feats, labels = algorithm_eval_tsne(self, test_loader, self.cfg.EPOCHS + self.cfg.VAL_EPOCH, 'test')
        return feats, labels
    
    def eval_heat(self, val_loader, test_loader):
        feats = algorithm_eval_heat(self, test_loader, self.cfg.EPOCHS + self.cfg.VAL_EPOCH, 'test')
        return feats
    

    def save_model(self, log_path):
        logging.info("Saving best model...")
        torch.save(self.network.state_dict(), os.path.join(log_path, 'best_model.pth'))
        torch.save(self.classifier.state_dict(), os.path.join(log_path, 'best_classifier.pth'))
        torch.save(self.network_ema.state_dict(), os.path.join(log_path, 'best_model_ema.pth'))
        torch.save(self.classifier_ema.state_dict(), os.path.join(log_path, 'best_classifier_ema.pth'))

    def renew_model(self, log_path):
        net_path = os.path.join(log_path, 'best_model.pth')
        classifier_path = os.path.join(log_path, 'best_classifier.pth')
        self.network.load_state_dict(torch.load(net_path))
        self.classifier.load_state_dict(torch.load(classifier_path))

        net_path_ema = os.path.join(log_path, 'best_model_ema.pth')
        classifier_path_ema = os.path.join(log_path, 'best_classifier_ema.pth')
        self.network_ema.load_state_dict(torch.load(net_path_ema))
        self.classifier_ema.load_state_dict(torch.load(classifier_path_ema))

    def load_model(self, model_path):
        if self.cfg.DG_MODE =='DG':
            checkpoint = torch.load(os.path.join(model_path, 'best_model_ema.pth'))
            checkpoint_ = torch.load(os.path.join(model_path, 'best_classifier_ema.pth'))
        else:
            checkpoint = torch.load(os.path.join(model_path, 'best_model.pth'))
            checkpoint_ = torch.load(os.path.join(model_path, 'best_classifier.pth'))
        # print(checkpoint, checkpoint_)
        self.network.load_state_dict(checkpoint)
        # checkpoint_ = torch.load(os.path.join(model_path, 'best_classifier.pth'))
        self.classifier.load_state_dict(checkpoint_)

    def predict(self, x):
        return self.classifier(self.network(x))
        # if self.cfg.DG_MODE =='DG':
        #     return self.classifier_ema(self.network_ema(x))
        # else:
        #     return self.classifier(self.network(x))
        # return self.classifier_ema(self.network_ema(x))

def JS_Divergence(output_clean, output_aug1, output_aug2, output_aug3):
    p_clean, p_aug1 = F.softmax(output_clean, dim=1), F.softmax(output_aug1, dim=1)
    p_aug2, p_aug3 = F.softmax(output_aug2, dim=1), F.softmax(output_aug3, dim=1)
    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2 + p_aug3) / 4., 1e-7, 1).log()
    # loss_ctr = (F.kl_div(p_mixture, p_clean, reduction='batchmean') + F.kl_div(p_mixture, p_aug1,
    #                                                                            reduction='batchmean') + F.kl_div(
    #     p_mixture, p_aug2, reduction='batchmean') + F.kl_div(p_mixture, p_aug3, reduction='batchmean')) / 4.
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

class GDRNet_DUAL(Algorithm):
    def __init__(self, num_classes, cfg):
        super(GDRNet_DUAL, self).__init__(num_classes, cfg)
        
        self.network1, self.network2 = models.get_net(cfg)
        self.classifier1 = models.get_classifier(self.network1.out_features(), cfg)

        # self.network2 = models.get_net(cfg)
        self.classifier2 = models.get_classifier(self.network2.out_features(), cfg)
        
        dim_in1 = 2048
        if cfg.BACKBONE2 == 'resnet18' or cfg.BACKBONE2 == 'resnet34':
            dim_in2 = 512
        else:
            dim_in2 = 2048

        feat_dim=5


        self.head1 = nn.Sequential(nn.Linear(dim_in1, dim_in1), nn.BatchNorm1d(dim_in1), nn.ReLU(inplace=True),
                                   nn.Linear(dim_in1, feat_dim))
        
        self.head2 = nn.Sequential(nn.Linear(dim_in2, dim_in2), nn.BatchNorm1d(dim_in2), nn.ReLU(inplace=True),
                                   nn.Linear(dim_in2, feat_dim))
        
        self.head_fc1 = nn.Sequential(nn.Linear(5, dim_in1), nn.BatchNorm1d(dim_in1), nn.ReLU(inplace=True),
                                   nn.Linear(dim_in1, feat_dim))
        
        self.head_fc2 = nn.Sequential(nn.Linear(5, dim_in2), nn.BatchNorm1d(dim_in2), nn.ReLU(inplace=True),
                                   nn.Linear(dim_in2, feat_dim))
        
        
        self.optimizer1 = torch.optim.SGD(
        # self.optimizer = torch.optim.Adam(
            [{"params":self.network1.parameters()},
            {"params":self.classifier1.parameters()},
            {"params":self.head1.parameters()},
            {"params":self.head_fc1.parameters()},],
            lr = cfg.LEARNING_RATE * 5,
            momentum = cfg.MOMENTUM,
            weight_decay = cfg.WEIGHT_DECAY,
            nesterov=True)
        
        self.optimizer2 = torch.optim.SGD(
        # self.optimizer = torch.optim.Adam(
            [{"params":self.network2.parameters()},
            {"params":self.classifier2.parameters()},
            {"params":self.head2.parameters()},
            {"params":self.head_fc2.parameters()},],
            lr = cfg.LEARNING_RATE,
            momentum = cfg.MOMENTUM,
            weight_decay = cfg.WEIGHT_DECAY,
            nesterov=True)
      

        self.fundusAug = get_post_FundusAug(cfg)
        self.criterion = DahLoss_Dual_BalSCL(beta= cfg.GDRNET.BETA, max_iteration = cfg.EPOCHS, \
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
        # expert = torch.zeros(all_y.shape[0], self.num_classes).to('cuda')

        self.optimizer1.zero_grad()
        self.optimizer2.zero_grad()

        image_new, image_ori = self.img_process(image, mask, self.fundusAug)
        features_ori1 = self.network1(image_ori)
        features_new1 = self.network1(image_new)
        output_new1 = self.classifier1(features_new1)

        features_ori1_mlp = F.normalize(self.head1(features_ori1), dim=1)
        features_new1_mlp = F.normalize(self.head1(features_new1), dim=1)
        
        features1_mlp = torch.cat([features_ori1_mlp.unsqueeze(1), features_new1_mlp.unsqueeze(1)], dim=1)
        centers_logits1 = F.normalize(self.head_fc1(self.classifier1.weight.T), dim=1)[:5]  ## Protetype

        features_ori2 = self.network2(image_ori)
        features_new2 = self.network2(image_new)
        output_new2 = self.classifier2(features_new2)
        features_ori2_mlp = F.normalize(self.head2(features_ori2), dim=1)
        features_new2_mlp = F.normalize(self.head2(features_new2), dim=1)
        features2_mlp = torch.cat([features_ori2_mlp.unsqueeze(1), features_new2_mlp.unsqueeze(1)], dim=1)

        centers_logits2 = F.normalize(self.head_fc2(self.classifier2.weight.T), dim=1)[:5]  ## Protetype

        
        # loss, loss_dict_iter = self.criterion([output_new1, output_new2], [features_ori1, features_new1, features_ori2, features_new2], label, domain)
        
        loss, loss_dict_iter = self.criterion([output_new1, output_new2], [features1_mlp, centers_logits1, features2_mlp, centers_logits2], label, domain)

        
        loss.backward()
        self.optimizer1.step()
        self.optimizer2.step()


        return loss_dict_iter



    def update_epoch(self, epoch):
        self.epoch = epoch
        return self.criterion.update_alpha(epoch)

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
        torch.save(self.network1.state_dict(), os.path.join(log_path, 'best_model.pth'))
        torch.save(self.classifier1.state_dict(), os.path.join(log_path, 'best_classifier.pth'))
    
    def renew_model(self, log_path):
        net_path = os.path.join(log_path, 'best_model.pth')
        classifier_path = os.path.join(log_path, 'best_classifier.pth')
        self.network1.load_state_dict(torch.load(net_path))
        self.classifier1.load_state_dict(torch.load(classifier_path))

    def predict(self, x):
        return self.classifier1(self.network1(x)) # ,self.classifier2(self.network2(x))
    
class GDRNet_DUAL_MASK(Algorithm):
    def __init__(self, num_classes, cfg):
        super(GDRNet_DUAL_MASK, self).__init__(num_classes, cfg)
        
        self.cfg = cfg
        self.network1, self.network2 = models.get_net(cfg)
        self.classifier1 = models.get_classifier(self.network1.out_features(), cfg)

        # self.network2 = models.get_net(cfg)
        self.classifier2 = models.get_classifier(self.network2.n_outputs, cfg)

        # self.featurizer_tea = networks.ResNet_tea(cfg)
        # self.network_tea = nn.Sequential(self.featurizer_tea, self.classifier_tea)
        self.id_featurizer = self.network1

        self.id_featurizer_tea = self.network2

        if cfg.BACKBONE1 == 'resnet34' or cfg.BACKBONE1 =='resnet18':
            dim_in1=512 
            feat_dim1=512  ## Resnet 18/34
        else:
            dim_in1=2048 ## Resnet 50/101
            feat_dim1=512
 
        if cfg.BACKBONE2 == 'resnet34' or cfg.BACKBONE2 == 'resnet18':
            dim_in2=512 
            
            feat_dim2=512 ## Resnet 18/34
        else:
            dim_in2=2048 ## Resnet 50/101
            feat_dim2=512

        # dim_in2=2048  ## Resnet 50/101
        # feat_dim2=512
        
        self.head1 = nn.Sequential(nn.Linear(dim_in1, dim_in1), 
                                   nn.BatchNorm1d(dim_in1), 
                                   nn.ReLU(inplace=True),
                                   nn.Linear(dim_in1, feat_dim1))
        
        self.head2 = nn.Sequential(nn.Linear(dim_in2, dim_in2), 
                                   nn.BatchNorm1d(dim_in2),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(dim_in2, feat_dim2))
        
        # #         # build a 3-layer projector
        # self.head1 = nn.Sequential(nn.Linear(dim_in1, dim_in1, bias=False), 
        #                                nn.BatchNorm1d(dim_in1),
        #                             nn.ReLU(inplace=True), 
        #                             nn.Linear(dim_in1, feat_dim1, bias=False), 
        #                             nn.BatchNorm1d(feat_dim1),
        #                             nn.ReLU(inplace=True), 
        #                             nn.Linear(feat_dim1, dim_in1, bias=False), 
        #                             nn.BatchNorm1d(dim_in1, affine=False)
        #                             )
        
        # self.head2 = nn.Sequential(nn.Linear(dim_in1, dim_in1, bias=False), 
        #                                nn.BatchNorm1d(dim_in1),
        #                             nn.ReLU(inplace=True), 
        #                             nn.Linear(dim_in1, feat_dim2, bias=False), 
        #                             nn.BatchNorm1d(feat_dim2),
        #                             nn.ReLU(inplace=True), 
        #                             nn.Linear(feat_dim2, dim_in1, bias=False), 
        #                             nn.BatchNorm1d(dim_in1, affine=False)
        #                             )
        
        
        init_weights_MLPHead(self.head1, init_method='He')
        init_weights_MLPHead(self.head2, init_method='He')



        #         # build a 3-layer projector
        # self.projector = nn.Sequential(nn.Linear(self.in_dim, dim1, bias=False), nn.BatchNorm1d(self.dim),
        #                             nn.ReLU(inplace=True), nn.Linear(self.dim, self.dim, bias=False), nn.BatchNorm1d(self.dim),
        #                             nn.ReLU(inplace=True), nn.Linear(self.dim, self.dim, bias=False), nn.BatchNorm1d(self.dim, affine=False))

        # # build a 2-layer predictor
        # self.predictor = nn.Sequential(nn.Linear(self.dim, self.pred_dim, bias=False),
        #                                 nn.BatchNorm1d(self.pred_dim),
        #                                 nn.ReLU(inplace=True), # hidden layer
        #                                 nn.Linear(self.pred_dim, self.dim)) # output layer
        
        # self.optimizer1 = torch.optim.SGD(
        self.optimizer1 = torch.optim.AdamW(
            [{"params":self.network1.parameters(), 'fix_lr': False},
            {"params":self.classifier1.parameters(), 'fix_lr': False},
            {"params":self.head1.parameters(), 'fix_lr': False},
            ],
            lr = cfg.LEARNING_RATE,
            # momentum = cfg.MOMENTUM,
            weight_decay = 0.0001, #cfg.WEIGHT_DECAY,
            # nesterov=True
            )
        
        # self.optimizer2 = torch.optim.SGD(
        self.optimizer2 = torch.optim.AdamW(
            [{"params":self.network2.parameters(), 'fix_lr': False},
            {"params":self.classifier2.parameters(), 'fix_lr': False},
            {"params":self.head2.parameters(), 'fix_lr': False},
            ],
            lr = cfg.LEARNING_RATE,
            # momentum = cfg.MOMENTUM,
            weight_decay = 0.0001, # cfg.WEIGHT_DECAY,
            # nesterov=True
            )
                                    

        self.optimizer_LD = torch.optim.AdamW([
        # self.optimizer_LD = torch.optim.SGD([
            {'params': self.network2.norm0.parameters(), 'fix_lr': False},
            {'params': self.network2.norm1.parameters(), 'fix_lr': False},
            {'params': self.network2.norm2.parameters(), 'fix_lr': False},
            {'params': self.network2.norm3.parameters(), 'fix_lr': False},
            {'params': self.network2.norm4.parameters(), 'fix_lr': False}],
            lr = cfg.LEARNING_RATE,
            # momentum = cfg.MOMENTUM,
            weight_decay = cfg.WEIGHT_DECAY,
            # nesterov=True
            )
        
        
        self.fundusAug = get_post_FundusAug(cfg)
        self.criterion = DahLoss_Dual(beta= cfg.GDRNET.BETA, max_iteration = cfg.EPOCHS, \
                                training_domains = cfg.DATASET.SOURCE_DOMAINS, temperature = cfg.GDRNET.TEMPERATURE, \
                                scaling_factor = cfg.GDRNET.SCALING_FACTOR)

        self.id_criterion = nn.CrossEntropyLoss()
        # self.domain_0, self.domain_1, self.domain_2, self.domain = {}, {}, {}, {}
        self.E_dis = nn.MSELoss()
                      

        self.masking = Masking(
            block_size=7,
            ratio=0.7,
            color_jitter_s=0.0,
            color_jitter_p=0.0,
            blur=False,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225))

        # _hparam('lambda_sem', 1.0, lambda r: 10**r.uniform(-1, 1))
    def img_process(self, img_tensor, mask_tensor, fundusAug):
        
        img_tensor_new, mask_tensor_new = fundusAug['post_aug1'](img_tensor.clone(), mask_tensor.clone())
        img_tensor_new = img_tensor_new * mask_tensor_new
        img_tensor_new = fundusAug['post_aug2'](img_tensor_new)
        img_tensor_ori = fundusAug['post_aug2'](img_tensor)

        return img_tensor_new, img_tensor_ori
    
    
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

        return x_masked

    # def forward(self, x, augsub_type='none', augsub_ratio=0.0):
    #     if augsub_type == 'masking':
    #         if augsub_ratio > 0.0:
    #             x = self.patchify(x)
    #             x = self.random_masking(x, augsub_ratio)
    #             x = self.unpatchify(x)
    #     elif augsub_type != 'none':
    #         raise NotImplementedError('Only support augsub_type == masking')
    #     x = self.forward_features(x)
    #     x = self.global_pool(x)
    #     if self.drop_rate:
    #         x = F.dropout(x, p=float(self.drop_rate), training=self.training)
    #     x = self.fc(x)
    #     return x
    
    def _define_hparam(hparams, hparam_name, default_val, random_val_fn):
        hparams[hparam_name] = (hparams, hparam_name, default_val, random_val_fn)
        
    def _hparams(algorithm, dataset, random_seed):
        """
        Global registry of hyperparams. Each entry is a (default, random) tuple.
        New algorithms / networks / etc. should add entries here.
        """
        SMALL_IMAGES = ['Debug28', 'RotatedMNIST', 'ColoredMNIST']

        hparams = {}
    
        # def _hparam(name, default_val, random_val_fn):
            
        #     hparams = {}
        #     """Define a hyperparameter. random_val_fn takes a RandomState and
        #     returns a random hyperparameter value."""
        #     assert(name not in hparams)
        #     random_state = np.random.RandomState(
        #         misc.seed_hash(random_seed, name)
        #     )
        #     hparams[name] = (default_val, random_val_fn(random_state))
            
        # # _hparam('lambda', 1.0, lambda r: 10**r.uniform(-1, 1))
        # _hparam('lambda_sem', 1.0, lambda r: 10**r.uniform(-1, 1))
        # _hparam('lambda_dis', 1.0, lambda r: 10**r.uniform(-1, 1))
        # # _hparam('margin', 0.75, lambda r: r.choice([0.1, 0.25, 0.5, 0.75]))
        
        
    def gram(self, y):
        # """ Returns the gram matrix of y (used to compute style loss) """
        (b, c, h, w) = y.size()

        features = y.view(b, c, w * h)
        features_t = features.transpose(1, 2) 
        gram_y = features.bmm(features_t) / (c * h * w)
        # print(gram_y.shape)  ## torch.Size([16, 64, 64])
        return gram_y

    def F_distance(self, x, y):
        return (torch.norm(x - y)).mean()
        # return torch.norm(x - y) + torch.norm(x - z)
        
        
    def update(self, minibatch):
        
        amp_grad_scaler = GradScaler()

        image, mask, label, domain = minibatch
        # expert = torch.zeros(all_y.shape[0], self.num_classes).to('cuda')

        self.optimizer1.zero_grad()
        self.optimizer2.zero_grad()

        image_new, image_ori = self.img_process(image, mask, self.fundusAug)
        
        # print(image_ori.shape,self.network1(image_ori).shape )  ## torch.Size([16, 3, 224, 224]) torch.Size([16, 2048])
        
        perturb = True
        # with autocast():
        features_ori1 = self.head1(self.network1(image_ori))
        features_new1 = self.network1(image_new)
        features_new1_ = self.head1(features_new1)

        output_new1 = self.classifier1(features_new1)

        features_ori2 = self.head2(self.network2(image_ori,perturb))
        features_new2 = self.network2(image_new,perturb)
        features_new2_ = self.head2(features_new2)

        output_new2 = self.classifier2(features_new2)
        
        
        
        self.consis_stu = self.E_dis(output_new1, output_new2.detach())
        self.consis_tea = self.E_dis(output_new1.detach(), output_new2)
    
        # image_new_masked = self.masking(image_new)    ################  masking
        # features_new_masked1 = self.network1(image_new_masked)

        # features_new_masked2 = self.network2(image_new_masked)

        image_new_masked = self.patchify(image_new, block=self.cfg.BLOCK)
        image_new_masked = self.random_masking(image_new_masked, self.cfg.MASK_RATIO)
        image_new_masked = self.unpatchify(image_new_masked)
        
        
        features_new_masked1 = self.network1(image_new_masked,drop_rate=0.3)
        features_new_masked2 = self.network2(image_new_masked,perturb)
        
        output_new_masked1 = self.classifier1(features_new_masked1)

        output_new_masked2 = self.classifier2(features_new_masked2)

        loss, loss_dict_iter = self.criterion([output_new1, output_new2, output_new_masked1, output_new_masked2], [features_ori1, features_new1_, features_ori2, features_new2_], label, domain)
        loss += 1.0 * self.consis_stu + 1.0 * self.consis_tea
        # print(self.consis_stu)
        # self.loss_stu = self.loss_id + self.hparams['lambda_sem'] * self.consis_stu
        # self.loss_tea = self.loss_id_tea + self.hparams['lambda_sem'] * self.consis_tea

        loss.backward()
        self.optimizer1.step()
        self.optimizer2.step()
        # amp_grad_scaler.update()
        
        
        # amp_grad_scaler.scale(loss).backward()
        # amp_grad_scaler.step(self.optimizer1)
        # amp_grad_scaler.step(self.optimizer2)
        # amp_grad_scaler.update()

        # with autocast():

        # student
        _ = self.id_featurizer(image_new)
        fea_0, fea_1, fea_2, fea_3, fea_4 = self.id_featurizer.x0, self.id_featurizer.x1, self.id_featurizer.x2, self.id_featurizer.x3, self.id_featurizer.x4
        # fea_0, fea_1 = self.id_featurizer.x0, self.id_featurizer.x1

        # teacher
        _ = self.id_featurizer_tea(image_new, perturb)
        fea_tea_0, fea_tea_1, fea_tea_2, fea_tea_3, fea_tea_4 = self.id_featurizer_tea.x0, self.id_featurizer_tea.x1, self.id_featurizer_tea.x2, self.id_featurizer_tea.x3, self.id_featurizer_tea.x4
        # fea_tea_0, fea_tea_1 = self.id_featurizer_tea.x0, self.id_featurizer_tea.x1

        self.simi_tea0 = -self.F_distance(self.gram(fea_0.detach()), self.gram(fea_tea_0))
        self.simi_tea1 = -self.F_distance(self.gram(fea_1.detach()), self.gram(fea_tea_1))
        self.simi_tea2 = -self.F_distance(self.gram(fea_2.detach()), self.gram(fea_tea_2))
        self.simi_tea3 = -self.F_distance(self.gram(fea_3.detach()), self.gram(fea_tea_3))
        self.simi_tea4 = -self.F_distance(self.gram(fea_4.detach()), self.gram(fea_tea_4))

        self.loss_norm = 0.05 * (self.simi_tea0 + self.simi_tea1 + self.simi_tea2 + self.simi_tea3 + self.simi_tea4)
        # print(self.loss_norm)
        # self.loss_norm = self.hparams['beta'] * (self.simi_tea0 + self.simi_tea1)

        self.optimizer_LD.zero_grad()
        self.loss_norm.backward()
        self.optimizer_LD.step()

        # amp_grad_scaler.scale(self.loss_norm).backward()
        # amp_grad_scaler.step(self.optimizer_LD)
        # amp_grad_scaler.update()
        
        
        # return loss_dict_iter
        
        # loss.backward()
        # self.optimizer1.step()
        # self.optimizer2.step()


        return loss_dict_iter
    


    def update_epoch(self, epoch):
        self.epoch = epoch
        return self.criterion.update_alpha(epoch)


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
        torch.save(self.network1.state_dict(), os.path.join(log_path, 'best_model1.pth'))
        torch.save(self.classifier1.state_dict(), os.path.join(log_path, 'best_classifier1.pth'))
        torch.save(self.network2.state_dict(), os.path.join(log_path, 'best_model.pth'))
        torch.save(self.classifier2.state_dict(), os.path.join(log_path, 'best_classifier.pth'))
    
    def renew_model(self, log_path):
        net_path = os.path.join(log_path, 'best_model.pth')
        classifier_path = os.path.join(log_path, 'best_classifier.pth')
        self.network1.load_state_dict(torch.load(net_path))
        self.classifier1.load_state_dict(torch.load(classifier_path))

    def predict(self, x):
        # return self.classifier1(self.network1(x))
        x0 = self.network2(x, True)
        x0 = self.classifier2(x0)
        return x0 # self.classifier2(self.network2(x))

    



@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class GDRNet_DUAL_MASK_SIAM(Algorithm):
    def __init__(self, num_classes, cfg):
        super(GDRNet_DUAL_MASK_SIAM, self).__init__(num_classes, cfg)
        
        self.cfg = cfg
        self.network1, self.network2 = models.get_net(cfg)
        self.classifier1 = models.get_classifier(self.network1.out_features(), cfg)

        # self.network2 = models.get_net(cfg)
        self.classifier2 = models.get_classifier(self.network2.out_features(), cfg)

        if cfg.BACKBONE1 == 'resnet34' or cfg.BACKBONE1 =='resnet18':
            dim_in1=512 
            feat_dim1=512  ## Resnet 18/34
        else:
            dim_in1=2048 ## Resnet 50/101
            feat_dim1=512
 
        if cfg.BACKBONE2 == 'resnet34' or cfg.BACKBONE2 == 'resnet18':
            dim_in2=512 
            
            feat_dim2=512  ## Resnet 18/34
        else:
            dim_in2=2048 ## Resnet 50/101
            feat_dim2=512
            
        # dim_in2=2048  ## Resnet 50/101
        # feat_dim2=512
        
        # self.head1 = nn.Sequential(nn.Linear(dim_in1, dim_in1), 
        #                            nn.BatchNorm1d(dim_in1), 
        #                            nn.ReLU(inplace=True),
        #                            nn.Linear(dim_in1, feat_dim1))
        
        # self.head2 = nn.Sequential(nn.Linear(dim_in2, dim_in2), 
        #                            nn.BatchNorm1d(dim_in2),
        #                            nn.ReLU(inplace=True),
        #                            nn.Linear(dim_in2, feat_dim2))
        
        #         # build a 3-layer projector
        self.projector1 = nn.Sequential(nn.Linear(dim_in1, dim_in1, bias=False), nn.BatchNorm1d(dim_in1),
                                    nn.ReLU(inplace=True), nn.Linear(dim_in1, dim_in1, bias=False), nn.BatchNorm1d(dim_in1),
                                    nn.ReLU(inplace=True), nn.Linear(dim_in1, dim_in1, bias=False), nn.BatchNorm1d(dim_in1, affine=False))

        # build a 2-layer predictor
        self.predictor1 = nn.Sequential(nn.Linear(dim_in1, dim_in1, bias=False),
                                        nn.BatchNorm1d(dim_in1),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(dim_in1, dim_in1)) # output layer
        
        
        #         # build a 3-layer projector
        self.projector2 = nn.Sequential(nn.Linear(dim_in2, dim_in2, bias=False), nn.BatchNorm1d(dim_in2),
                                    nn.ReLU(inplace=True), nn.Linear(dim_in2, dim_in2, bias=False), nn.BatchNorm1d(dim_in2),
                                    nn.ReLU(inplace=True), nn.Linear(dim_in2, dim_in2, bias=False), nn.BatchNorm1d(dim_in2, affine=False))

        # build a 2-layer predictor
        self.predictor2 = nn.Sequential(nn.Linear(dim_in2, dim_in2, bias=False),
                                        nn.BatchNorm1d(dim_in2),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(dim_in2, dim_in2)) # output layer
        
        
        self.optimizer1 = torch.optim.SGD(
        # self.optimizer = torch.optim.Adam(
            [{"params":self.network1.parameters(), 'fix_lr': False},
            {"params":self.classifier1.parameters(), 'fix_lr': False},
            {"params":self.projector1.parameters(), 'fix_lr': False},
            {"params":self.predictor1.parameters(), 'fix_lr': True},
            ],
            lr = cfg.LEARNING_RATE * 5,
            momentum = cfg.MOMENTUM,
            weight_decay = cfg.WEIGHT_DECAY,
            nesterov=True)
        
        self.optimizer2 = torch.optim.SGD(
        # self.optimizer = torch.optim.Adam(
            [{"params":self.network2.parameters(), 'fix_lr': False},
            {"params":self.classifier2.parameters(), 'fix_lr': False},
            {"params":self.projector2.parameters(), 'fix_lr': False},
            {"params":self.predictor2.parameters(), 'fix_lr': True},
            ],
            lr = cfg.LEARNING_RATE,
            momentum = cfg.MOMENTUM,
            weight_decay = cfg.WEIGHT_DECAY,
            nesterov=True)
                                    
        K=1024
        dim=2048
        self.K=K
        self.num_positive = 0

        # create queue for keeping neighbor
        self.register_buffer("queue", torch.randn(K, dim))
        self.queue = nn.functional.normalize(self.queue, dim=-1)
        self.register_buffer("queue_labels", -torch.ones(K, dtype=torch.long))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        self.fundusAug = get_post_FundusAug(cfg)
        self.criterion =  DahLoss_Dual_Siam(beta= cfg.GDRNET.BETA, max_iteration = cfg.EPOCHS, \
                                training_domains = cfg.DATASET.SOURCE_DOMAINS, temperature = cfg.GDRNET.TEMPERATURE, \
                                scaling_factor = cfg.GDRNET.SCALING_FACTOR)
                                    

        self.masking = Masking(
            block_size=7,
            ratio=0.7,
            color_jitter_s=0.0,
            color_jitter_p=0.0,
            blur=False,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225))


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
        img_tensor_new = img_tensor_new * mask_tensor_new
        img_tensor_new = fundusAug['post_aug2'](img_tensor_new)
        img_tensor_ori = fundusAug['post_aug2'](img_tensor)

        return img_tensor_new, img_tensor_ori
    
    
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

        return x_masked
    
    
    def update(self, minibatch):
        
        image, mask, label, domain = minibatch
        # expert = torch.zeros(all_y.shape[0], self.num_classes).to('cuda')

        self.optimizer1.zero_grad()
        self.optimizer2.zero_grad()

        image_new, image_ori = self.img_process(image, mask, self.fundusAug)
        
        # print(image_ori.shape,self.network1(image_ori).shape )  ## torch.Size([16, 3, 224, 224]) torch.Size([16, 2048])
        
        features_ori1_z1 = self.projector1(self.network1(image_ori))  ## z1
        features_new1 = self.network1(image_new)
        features_new1_z2 = self.projector1(features_new1)
        
        p11, p12 = self.predictor1(features_ori1_z1), self.predictor1(features_new1_z2) 


        features_ori1_z1, features_new1_z2 = nn.functional.normalize(features_ori1_z1, dim=-1), nn.functional.normalize(features_new1_z2, dim=-1)
        p11, p12 = nn.functional.normalize(p11, dim=-1), nn.functional.normalize(p12, dim=-1)
        
        
        features_ori2_z1 = self.projector2(self.network2(image_ori))  ## z1
        features_new2 = self.network2(image_new)
        features_new2_z2 = self.projector2(features_new2)
        
        
        p21, p22 = self.predictor2(features_ori2_z1), self.predictor2(features_new2_z2) 

        features_ori2_z1, features_new2_z2 = nn.functional.normalize(features_ori2_z1, dim=-1), nn.functional.normalize(features_new2_z2, dim=-1)
        p21, p22 = nn.functional.normalize(p21, dim=-1), nn.functional.normalize(p22, dim=-1)
        
        
        # sample supervised targets
        z11_sup = self.sample_target(features_ori1_z1.detach(), label)
        z12_sup = self.sample_target(features_new1_z2.detach(), label)
        
        
        z21_sup = self.sample_target(features_ori2_z1.detach(), label)
        z22_sup = self.sample_target(features_new2_z2.detach(), label)

        
        output_new1 = self.classifier1(features_new1)
        output_new2 = self.classifier2(features_new2)

        image_new_masked = self.patchify(image_new, block=self.cfg.BLOCK)
        image_new_masked = self.random_masking(image_new_masked, self.cfg.MASK_RATIO)
        image_new_masked = self.unpatchify(image_new_masked)
        
        
        features_new_masked1 = self.network1(image_new_masked,drop_rate=0.3)
        features_new_masked2 = self.network2(image_new_masked,drop_rate=0.3)
        
        output_new_masked1 = self.classifier1(features_new_masked1)

        output_new_masked2 = self.classifier2(features_new_masked2)


        # dequeue and enqueue
        self.dequeue_and_enqueue(features_new1_z2.detach(), label)
        self.dequeue_and_enqueue(features_new2_z2.detach(), label)

        
        loss, loss_dict_iter = self.criterion([output_new1, output_new2, output_new_masked1, output_new_masked2], [features_ori1_z1,features_new1_z2, z11_sup, z12_sup, p11, p12, features_ori2_z1,features_new2_z2, z21_sup, z22_sup, p21, p22], label, domain)
        
        
        loss.backward()
        self.optimizer1.step()
        self.optimizer2.step()


        return loss_dict_iter
    


    def update_epoch(self, epoch):
        self.epoch = epoch
        return self.criterion.update_alpha(epoch)


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
        torch.save(self.network1.state_dict(), os.path.join(log_path, 'best_model.pth'))
        torch.save(self.classifier1.state_dict(), os.path.join(log_path, 'best_classifier.pth'))
    
    def renew_model(self, log_path):
        net_path = os.path.join(log_path, 'best_model.pth')
        classifier_path = os.path.join(log_path, 'best_classifier.pth')
        self.network1.load_state_dict(torch.load(net_path))
        self.classifier1.load_state_dict(torch.load(classifier_path))

    def predict(self, x):
        return self.classifier1(self.network1(x))
    
    
    
class GDRNet_LFME(Algorithm):
    def __init__(self, num_classes, cfg):
        super(GDRNet, self).__init__(num_classes, cfg)

        self.expert_number = num_domains + 1
        self.num_classes = num_classes
        self.featurizer = [None] * self.expert_number
        self.classifier = [None] * self.expert_number
        self.network = [None] * self.expert_number
        self.optimizer = [None] * self.expert_number
        device = 'cuda' #or 'cpu'
        for i in range(self.expert_number):
            self.featurizer[i] = models.get_net(cfg).to(device)
            self.classifier[i] = models.get_classifier(self.network.out_features(), cfg).to(device)
            self.network[i] = nn.Sequential(self.featurizer[i], self.classifier[i])
            self.optimizer[i] = torch.optim.Adam(
                self.network[i].parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay']
                )

        # self.optimizer = torch.optim.SGD(
        # # self.optimizer = torch.optim.Adam(
        #     [{"params":self.network.parameters()},
        #     {"params":self.classifier.parameters()}],
        #     lr = cfg.LEARNING_RATE,
        #     momentum = cfg.MOMENTUM,
        #     weight_decay = cfg.WEIGHT_DECAY,
        #     nesterov=True)

        self.expert_number = num_domains + 1
        self.fundusAug = get_post_FundusAug(cfg)
        self.criterion = DahLoss(beta= cfg.GDRNET.BETA, max_iteration = cfg.EPOCHS, \
                                training_domains = cfg.DATASET.SOURCE_DOMAINS, temperature = cfg.GDRNET.TEMPERATURE, \
                                scaling_factor = cfg.GDRNET.SCALING_FACTOR)
                                    

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
        # expert = torch.zeros(all_y.shape[0], self.num_classes).to('cuda')

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
    
    

class LFME(Algorithm):
    """
    Learning from Multiple Experts for Domain Generalization
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(LFME, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.MSEloss = nn.MSELoss()
        self.expert_number = num_domains + 1
        self.num_classes = num_classes
        self.featurizer = [None] * self.expert_number
        self.classifier = [None] * self.expert_number
        self.network = [None] * self.expert_number
        self.optimizer = [None] * self.expert_number
        device = 'cuda' #or 'cpu'
        for i in range(self.expert_number):
            self.featurizer[i] = networks.Featurizer(input_shape, self.hparams).to(device)
            self.classifier[i] = networks.Classifier(self.featurizer[i].n_outputs,
                num_classes,self.hparams['nonlinear_classifier']).to(device)
            self.network[i] = nn.Sequential(self.featurizer[i], self.classifier[i])
            self.optimizer[i] = torch.optim.Adam(
                self.network[i].parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay']
                )
			
    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        expert = torch.zeros(all_y.shape[0], self.num_classes).to('cuda')
        for i in range(self.expert_number-1):
            mmbatch = minibatches[i]
            part_x, part_y = mmbatch[0], mmbatch[1]
            result_expert = self.network[i](part_x)
            loss = F.cross_entropy(result_expert, part_y)
            self.optimizer[i].zero_grad()
            loss.backward()
            self.optimizer[i].step()
            index, end = (i) * part_y.shape[0], (i + 1) * part_y.shape[0]
            expert[index:end, :] = F.softmax(result_expert, dim=1)

        result_target = self.network[-1](all_x)
        loss_cla = F.cross_entropy(result_target, all_y)
        loss_guid = self.MSEloss(result_target, expert.detach())
        loss = loss_cla + loss_guid * self.hparams['lfe_reg']
        self.optimizer[-1].zero_grad()
        loss.backward()
        self.optimizer[-1].step()
        return {'loss': loss.item()}


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
       return self.network[-1](x)



class GREEN(Algorithm):
    def __init__(self, num_classes, cfg):
        super(GREEN, self).__init__(num_classes, cfg)
        self.network = models.get_net(cfg)
        self.optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr = cfg.LEARNING_RATE,
            momentum = cfg.MOMENTUM,
            weight_decay = cfg.WEIGHT_DECAY,
            nesterov=True)
    
    def update(self, minibatch):
        image, mask, label, domain = minibatch
        self.optimizer.zero_grad()

        output = self.network(image)
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
    
    def renew_model(self, log_path):
        net_path = os.path.join(log_path, 'best_model.pth')
        self.network.load_state_dict(torch.load(net_path))
    
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

        return {'loss':loss}
    
    def mixup_data(self, x, y, alpha=1.0, use_cuda=True):
        '''Returns mixed inputs, pairs of targets, and lambda'''
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
            models.get_classifier(self.network._out_features, cfg)
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
        #self.network.train()

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
                # Reset Adam as in IRM or V-REx, because it may not like the sharp jump in
                # gradient magnitudes that happens at this step.
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
            #torch.autograd.grad(outputs=loss,inputs=list(self.classifier.parameters()),retain_graph=True, create_graph=True)
            
        # compute individual grads for all samples across all domains simultaneously
        dict_grads = OrderedDict(
            [
                (name, weights.grad_batch.clone().view(weights.grad_batch.size(0), -1))
                for name, weights in self.classifier.named_parameters()
            ]
        )
        return dict_grads

    def _get_grads_var_per_domain(self, dict_grads, len_minibatches):
        # grads var per domain
        grads_var_per_domain = [{} for _ in range(self.num_groups)]
        for name, _grads in dict_grads.items():
            all_idx = 0
            for domain_id, bsize in enumerate(len_minibatches):
                env_grads = _grads[all_idx:all_idx + bsize]
                all_idx += bsize
                env_mean = env_grads.mean(dim=0, keepdim=True)
                env_grads_centered = env_grads - env_mean
                grads_var_per_domain[domain_id][name] = (env_grads_centered).pow(2).mean(dim=0)

        # moving average
        for domain_id in range(self.num_groups):
            grads_var_per_domain[domain_id] = self.ema_per_domain[domain_id].update(
                grads_var_per_domain[domain_id]
            )

        return grads_var_per_domain

    def _compute_distance_grads_var(self, grads_var_per_domain):

        # compute gradient variances averaged across domains
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

# DRGen is built based on Fishr method

class DRGen(Algorithm):
    '''
    Refer to the paper 'DRGen: Domain Generalization in Diabetic Retinopathy Classification' 
    https://link.springer.com/chapter/10.1007/978-3-031-16434-7_61
    
    '''
    def __init__(self, num_classes, cfg):
        super(DRGen, self).__init__(num_classes, cfg)
        algorithm_class = get_algorithm_class('Fishr')
        self.algorithm = algorithm_class(cfg.DATASET.NUM_CLASSES, cfg)
        self.optimizer = self.algorithm.optimizer
        
        self.swad_algorithm = AveragedModel(self.algorithm)
        self.swad_algorithm.cuda()
        #swad_cls = getattr(swad_module, 'LossValley')
        #swad_cls = LossValley()
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
                        #break
                    
                    self.swad_algorithm = AveragedModel(self.algorithm)  # reset
            
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