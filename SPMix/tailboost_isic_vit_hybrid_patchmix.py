import argparse
import builtins
import math
import os

import numpy as np
from torchvision.datasets import ImageFolder
from torch.utils.tensorboard import SummaryWriter

os.environ['OPENBLAS_NUM_THREADS'] = '2'
import random
import shutil
import time
import warnings
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from functools import partial
from models import resnet_imagenet
from randaugment import rand_augment_transform, GaussianBlur
import moco.loader
import moco.builder_patchmix
from dataset.imagenet import ImageNetLT
from dataset.imagenet_moco import ImageNetLT_moco
from dataset.isic_mix_2 import ISICImageFolder
from dataset.inat import INaturalist
from dataset.inat_moco import INaturalist_moco
from losses import PaCoLoss
from utils import shot_acc
import moco.vision_transformer_hybrid as vits
import torchvision.models as torchvision_models
from torch.utils.data.sampler import WeightedRandomSampler
from accelerate import Accelerator


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='pad', choices=['inat', 'imagenet', 'pad','aptos'])
parser.add_argument('--data', metavar='DIR', default='./data')
parser.add_argument('--mix_data', metavar='DIR', default='./data')
parser.add_argument('--root_path', type=str, default='./data')
parser.add_argument('-a', '--arch', metavar='ARCH', default='vit_small',)
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=500, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=5e-6, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=.1, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:9987', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', default=True, type=bool,
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# moco specific configs:
parser.add_argument('--moco-dim', default=256, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=8192, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-mlp-dim', default=4096, type=int,
                    help='hidden dimension in MLPs (default: 4096)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.2, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--stop-grad-conv1', action='store_true',
                    help='stop-grad after first conv, or patch embedding')
# options for moco v2
parser.add_argument('--mlp', default=True, type=bool,
                    help='use mlp head')
parser.add_argument('--aug-plus', default=True, type=bool,
                    help='use moco v2 data augmentation')
parser.add_argument('--cos', default=True, type=bool,
                    help='use cosine lr schedule')
parser.add_argument('--normalize', default=False, type=bool,
                    help='use cosine lr schedule')

# options for paco
parser.add_argument('--mark', default=None, type=str,
                    help='log dir')
parser.add_argument('--reload', default=None, type=str,
                    help='load supervised model')
parser.add_argument('--warmup_epochs', default=10, type=int,
                    help='warmup epochs')
parser.add_argument('--alpha', default=0.05, type=float,
                    help='contrast weight among samples')
parser.add_argument('--beta', default=1.0, type=float,
                    help='contrast weight between centers and samples')
parser.add_argument('--gamma', default=1.0, type=float,
                    help='paco loss')
parser.add_argument('--aug', default="randcls_sim", type=str,
                    help='aug strategy')
parser.add_argument('--randaug_m', default=10, type=int, help='randaug-m')
parser.add_argument('--randaug_n', default=2, type=int, help='randaug-n')
parser.add_argument('--num_classes', default=6, type=int, help='num classes in dataset')
parser.add_argument('--feat_dim', default=768, type=int,
                    help='last feature dim of backbone')
parser.add_argument('--weighted_alpha', default=1, type=float, help='weighted alpha for sampling probability (q(1,k))')

# fp16
parser.add_argument('--fp16', action='store_true', help=' fp16 training')

best_acc = 0
writer = SummaryWriter("./Paco_logs/logs_train_pad_vit_hybrid_pretrain_gpaco_spmix")


def main():
    args = parser.parse_args()
    args.root_model = f'{args.root_path}/{args.dataset}/{args.mark}'
    os.makedirs(args.root_model, exist_ok=True)
    if args.seed is not None:
    #    seed_everything(args.seed)
       print("seed:{}".format(args.seed))
       os.environ['PYTHONHASHSEED'] = str(args.seed)  # 为了禁止hash随机化，使得实验可复现
       np.random.seed(args.seed)
       torch.cuda.manual_seed(args.seed)
       torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
       random.seed(args.seed)
       torch.manual_seed(args.seed)
       cudnn.deterministic = True
       warnings.warn('You have chosen to seed training. '
                     'This will turn on the CUDNN deterministic setting, '
                     'which can slow down your training considerably! '
                     'You may see unexpected behavior when restarting '
                     'from checkpoints.')
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    main_worker(args)


def main_worker(args):
    global best_acc
    accelerator = Accelerator(mixed_precision='fp16')
    # num_classes = 8142 if args.dataset == 'inat' else 1000
    accelerator.print("=> creating model '{}'".format(args.arch))
    if args.arch.startswith('vit'):
        model = moco.builder_patchmix.MoCo_ViT(
            partial(vits.__dict__[args.arch]),
            args.moco_dim, args.moco_mlp_dim, args.moco_k, args.moco_t,args.feat_dim, args.normalize,
        num_classes=args.num_classes)
    if args.distributed:

        filename = f'{args.root_model}/moco_ckpt.best.pth.tar'
        if os.path.exists(filename):
            args.resume = filename
        if args.reload:
            unwrapped_model = accelerator.unwrap_model(model)
            state_dict = model.state_dict()
            state_dict_ssp = torch.load(args.reload)['state_dict']
            accelerator.print(state_dict_ssp.keys())
            for key in state_dict.keys():
                accelerator.print(key)
                if key in state_dict_ssp.keys() and state_dict[key].shape == state_dict_ssp[key].shape:
                    state_dict[key] = state_dict_ssp[key]
                    accelerator.print(key + " ****loaded******* ")
                else:
                    accelerator.print(key + " ****unloaded******* ")
            unwrapped_model.load_state_dict(state_dict)
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    criterion_ce = nn.CrossEntropyLoss()
    criterion = PaCoLoss(alpha=args.alpha, beta=args.beta, gamma=args.gamma, temperature=args.moco_t, K=args.moco_k,
                         num_classes=args.num_classes)

    optimizer =  torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            accelerator.print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            unwrapped_model = accelerator.unwrap_model(model)
            args.start_epoch = checkpoint['epoch']
            unwrapped_model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            accelerator.print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            accelerator.print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    txt_train = f'./dataset/data_txt/APTOS_train.txt' if args.dataset == 'aptos' \
        else f"/raid5/weityx/Dataset/PAD/PAD_train.txt"

    txt_test = f'./dataset/data_txt/APTOS_val.txt' if args.dataset == 'aptos' \
        else f"/raid5/weityx/Dataset/PAD/PAD_val.txt"

    normalize = transforms.Normalize(mean=[0.466, 0.471, 0.380], std=[0.195, 0.194, 0.192]) if args.dataset == 'inat' \
        else transforms.Normalize(mean=[0.765, 0.545, 0.569],
                                  std=[0.140, 0.151, 0.169])
    if args.aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    else:
        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]

    augmentation_regular = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
        transforms.ToTensor(),
        normalize,
    ]

    augmentation_sim = [
        transforms.RandomResizedCrop(224),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)  # not strengthened
        ], p=1.0),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    augmentation_sim02 = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)  # not strengthened
        ], p=1.0),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    rgb_mean = (0.485, 0.456, 0.406)
    ra_params = dict(translate_const=int(224 * 0.45), img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]), )
    augmentation_randnclsstack = [
        transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
        ], p=1.0),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
        rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(args.randaug_n, args.randaug_m), ra_params),
        transforms.ToTensor(),
        normalize,
    ]

    augmentation_randncls = [
        transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
        ], p=1.0),
        rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(args.randaug_n, args.randaug_m), ra_params),
        transforms.ToTensor(),
        normalize,
    ]

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize])

    val_dataset = ImageNetLT(
        root=args.data,
        txt=txt_test,
        transform=val_transform
    ) if args.dataset == 'aptos' else ImageNetLT(
        root=args.data,
        txt=txt_test,
        transform=val_transform)

    if args.aug == 'regular_regular':
        transform_train = [transforms.Compose(augmentation_regular), transforms.Compose(augmentation)]
    elif args.aug == 'mocov2_mocov2':
        transform_train = [transforms.Compose(augmentation), transforms.Compose(augmentation)]
    elif args.aug == 'sim_sim':
        transform_train = [transforms.Compose(augmentation_sim), transforms.Compose(augmentation_sim)]
    elif args.aug == 'randcls_sim':
        transform_train = [transforms.Compose(augmentation_randncls), transforms.Compose(augmentation_sim)]
    elif args.aug == 'randclsstack_sim':
        transform_train = [transforms.Compose(augmentation_randnclsstack), transforms.Compose(augmentation_sim)]
    elif args.aug == 'randclsstack_sim02':
        transform_train = [transforms.Compose(augmentation_randnclsstack), transforms.Compose(augmentation_sim02)]

    train_dataset = INaturalist_moco(
        root=args.data,
        txt=txt_train,
        transform=transform_train
    ) if args.dataset == 'inat' else ISICImageFolder(
        root=args.data,
        mix_root=args.mix_data,
        txt=txt_train,
        num_classes=args.num_classes,
        transform=transform_train)
    accelerator.print(f'===> Training data length {len(train_dataset)}')

    criterion.cal_weight_for_classes(train_dataset.cls_num_list)
    cls_weight = 1.0 / (np.array(train_dataset.cls_num_list) ** args.weighted_alpha)
    cls_weight = cls_weight / np.sum(cls_weight) * len(train_dataset.cls_num_list)
    samples_weight = np.array([cls_weight[t] for t in train_dataset.labels])
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    accelerator.print(cls_weight)
    weighted_sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight),
                                                              replacement=True)
    weighted_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                            num_workers=args.workers, pin_memory=True,
                                                            sampler=weighted_sampler)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)


    model, optimizer, train_loader, val_loader,weighted_train_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader,weighted_train_loader
    )
    if args.evaluate:
        accelerator.print(" start evaualteion **** ")
        test_matrix(val_loader, model, args,train_loader,accelerator)
        return

    # mixed precision 
    scaler = GradScaler()

    for epoch in range(args.start_epoch, args.epochs):
        # if args.distributed:
        #     train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader,weighted_train_loader, model, criterion, optimizer, epoch, scaler, args,accelerator)
        acc = validate(val_loader, train_loader, model, criterion_ce, args,accelerator)
        if acc > best_acc:
            best_acc = acc
            is_best = True
        else:
            is_best = False

        accelerator.print("训练次数:{},{}".format(epoch, best_acc))
        writer.add_scalar("test_accuracy", acc, epoch)
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'acc': acc,
                'state_dict': unwrapped_model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best=is_best, accelerator=accelerator, filename=f'{args.root_model}/moco_ckpt.pth.tar')

def test_matrix(test_loader, model, args,train_loader,accelerator):
    y_pre = []
    y_target = []
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(test_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)
            # compute output
            output = model(images)
            _, y = output.max(1)
            y, target = accelerator.gather_for_metrics((y, target))
            y_pre.extend(list(y.cpu().numpy()))
            y_target.extend(list(target.cpu().numpy()))

    y_pre = np.array(y_pre).reshape(1, -1)
    y_pre = np.squeeze(y_pre, 0)
    y_target = np.array(y_target).reshape(1, -1)
    y_target = np.squeeze(y_target, 0)
    many_acc_top1, median_acc_top1, low_acc_top1, cls_accs = shot_acc(y_pre, y_target,train_loader ,many_shot_thr=224,low_shot_thr=172,
                                                                          acc_per_cls=True)
    accelerator.print('Many_acc: %.5f, Medium_acc: %.5f Low_acc: %.5f\n' % (many_acc_top1, median_acc_top1, low_acc_top1))
    accelerator.print(classification_report(y_target,y_pre, digits=4))

#找到saliencymap里最小score对应patch index
def saliency_sortpatch(img_saliency,reimg_saliency,mix_threshold=98):
    row_step=16
    B=img_saliency.shape[0]
    patch_score=F.avg_pool2d(img_saliency,16,16).squeeze(1).flatten(1)
    repatch_score=F.avg_pool2d(reimg_saliency,16,16).squeeze(1).flatten(1)
    patch_score=torch.maximum(patch_score,repatch_score)
    max_idx=torch.topk(patch_score,mix_threshold)[1].flatten()
    min_idx=torch.topk(patch_score,mix_threshold,largest=False)[1].flatten()
    bi=torch.repeat_interleave(torch.arange(0,B),mix_threshold)
    patch_idx_max=[bi,max_idx]
    patch_idx_min=[bi,min_idx]
    return patch_idx_max,patch_idx_min

#找到saliencymap里最大score对应patch index
def saliency_maxpatch(img_saliency,mix_threshold):
    row_step=16
    B=img_saliency.shape[0]
    patch_score=F.avg_pool2d(img_saliency,16,16).squeeze(1).flatten(1)
    patch_score=torch.sort(patch_score.argsort())[0]
    patch_idx_max=torch.where(patch_score>=196-mix_threshold)
    return patch_idx_max

def saliency_score(img_saliency):
    B=img_saliency.shape[0]
    noise=torch.randint(-5000,5000,img_saliency.shape).cuda()
    img_saliency+=noise*0.0001
    patch_score=F.avg_pool2d(img_saliency,16,16).reshape(B,196)
    min_i, _ = torch.min(patch_score, dim=1, keepdim=True)
    max_i, _ = torch.max(patch_score, dim=1, keepdim=True)
    patch_score = (patch_score - min_i) / (max_i-min_i)
    patch_score =patch_score/2+0.5
    return patch_score.half()

def random_mixpatch_ratio(bs,mix_threshold=98):   #0.5-1的小数
    sz=torch.zeros((bs,mix_threshold))
    ratio=torch.randint_like(sz,2000,5000)*0.0001
    ratio=torch.sort(ratio)[0].half()
    # ratio=ratio.half()
    return 1-ratio,ratio

def train(train_loader,weighted_train_loader, model, criterion, optimizer, epoch, scaler, args,accelerator):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    mix_threshold=98
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    iters_per_epoch = len(train_loader)
    end = time.time()
    moco_m = args.moco_m
    weighted_train_loader = iter(weighted_train_loader)

    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        try:
            input2,target2 = next(weighted_train_loader)
        except:
            weighted_train_loader = iter(weighted_train_loader)
            input2, target2 = next(weighted_train_loader)

        #saliency score做ratio
        # min_ratio = torch.ones_like(ratio_q)*0.2
        ratio_q=torch.maximum(saliency_score(images[2]),saliency_score(input2[2]))
        # ratio_q=saliency_score(images[2])
        # ratio_q = torch.where(ratio_q <0.2, min_ratio, ratio_q)
        max_ratio = torch.ones_like(ratio_q)*0.8
        ratio_q = torch.where(ratio_q >0.8, max_ratio, ratio_q)
        ratio_k=torch.maximum(saliency_score(images[3]),saliency_score(input2[3]))
        # ratio_k=saliency_score(images[3])
        # ratio_k = torch.where(ratio_k <0.2, min_ratio, ratio_k)
        ratio_k = torch.where(ratio_k >0.8, max_ratio, ratio_k)
        ratio=[ratio_q,ratio_k]
        # compute output
        if not args.fp16:
            features, labels, logits = model(im_q=images[0], im_k=images[1],im_q_2=input2[0],im_k_2=input2[1],labels=target,ratio=ratio,accelerator=accelerator)
            loss = criterion(features, labels, logits)
        else:
            with autocast():
                features, labels, logits = model(im_q=images[0], im_k=images[1],im_q_2=input2[0],im_k_2=input2[1],labels=target,ratio=ratio,accelerator=accelerator)
                loss = criterion(features, labels, logits)
        
        logits=logits[:target.size(0)]
        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        losses.update(loss.item(), logits.size(0))
        top1.update(acc1[0], logits.size(0))
        top5.update(acc5[0], logits.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if not args.fp16:
            accelerator.backward(loss)
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        writer.add_scalar("train_loss", losses.avg, epoch)
        writer.add_scalar("train_acc", top1.avg, epoch)
        if i % args.print_freq == 0:
            progress.display(i, args)


def validate(val_loader, train_loader, model, criterion, args,accelerator):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    total_logits = torch.empty((0, args.num_classes)).cuda()
    total_labels = torch.empty(0, dtype=torch.long).cuda()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            # compute output
            output = model(images)
            loss = criterion(output, target)
            output, target = accelerator.gather_for_metrics((output, target))
            total_logits = torch.cat((total_logits, output))
            total_labels = torch.cat((total_labels, target))

            # measure accuracy and record loss
            acc1, acc5 = accuracy(total_logits, total_labels, topk=(1, 5))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i, args)

        # TODO: this should also be done with the ProgressMeter
        open(args.root_model + "/" + args.mark + ".log", "a+").write(' * Acc@1 {top1:.3f} Acc@5 {top5:.3f}\n'
                                                                     .format(top1=acc1[0], top5=acc5[0]))

        probs, preds = F.softmax(total_logits.detach(), dim=1).max(dim=1)
        many_acc_top1, median_acc_top1, low_acc_top1, cls_accs = shot_acc(preds, total_labels, train_loader,
                                                                          acc_per_cls=True)

        open(args.root_model + "/" + args.mark + ".log", "a+").write(
            'Many_acc: %.5f, Medium_acc: %.5f Low_acc: %.5f\n' % (many_acc_top1, median_acc_top1, low_acc_top1))
    return acc1[0]


def save_checkpoint(state, is_best, accelerator,filename='checkpoint.pth.tar'):
    accelerator.save(state,filename)
    if is_best:
        accelerator.save(state, filename.replace('pth.tar', 'best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, args):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        open(args.root_model + "/" + args.mark + ".log", "a+").write('\t'.join(entries) + "\n")

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if epoch < args.warmup_epochs:
        lr = lr / args.warmup_epochs * (epoch + 1)
    elif args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs + 1) / (args.epochs - args.warmup_epochs )))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_moco_momentum(epoch, args):
    """Adjust moco momentum based on current epoch"""
    m = 1. - 0.5 * (1. + math.cos(math.pi * epoch / args.epochs)) * (1. - args.moco_m)
    return m

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred)).contiguous()

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
