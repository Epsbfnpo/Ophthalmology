import sys, os, logging, shutil
from torch.utils.tensorboard import SummaryWriter
import torch, random
import numpy as np
from collections import Counter
import math

ALL_DATASETS = ['APTOS', 'DEEPDR', 'FGADR', 'IDRID', 'MESSIDOR', 'RLDR']
ESDG_DATASETS = ['APTOS', 'DEEPDR', 'FGADR', 'IDRID', 'MESSIDOR', 'RLDR', 'DDR', 'EYEPACS']

# [Merged] 更新为包含 MASK_SIAM 系列的完整列表
ALL_METHODS = [
    'GDRNet', 'GDRNet_DUAL', 'GDRNet_Mask', 'GDRNet_MASK_SIAM', 'GDRNet_DUAL_MASK',
    'GDRNet_DUAL_MASK_SIAM', 'ERM', 'GREEN', 'CABNet', 'MixupNet', 'MixStyleNet',
    'Fishr', 'DRGen'
]


def count_samples_per_class(targets, num_classes):
    counts = Counter()
    for y in targets:
        counts[int(y)] += 1
    return [counts[i] if counts[i] else np.inf for i in range(num_classes)]


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def init_log(args, cfg, log_path, train_loader_length, dataset_size):
    assert cfg.ALGORITHM in ALL_METHODS
    if not cfg.RANDOM:
        setup_seed(cfg.SEED)

    init_output_foler(cfg, log_path)
    writer = SummaryWriter(os.path.join(log_path, 'tensorboard'))
    writer.add_text('config', str(args))
    logging.basicConfig(filename=log_path + '/log.txt', level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info("{} iterations per epoch".format(train_loader_length))
    logging.info(
        "We have {} images in train set, {} images in val set, and {} images in test set.".format(dataset_size[0],
                                                                                                  dataset_size[1],
                                                                                                  dataset_size[2]))
    logging.info(str(args))
    logging.info(str(cfg))
    return writer


def init_output_foler(cfg, log_path):
    # [Merged] 使用 Original 版本更稳健的文件夹检查逻辑
    if os.path.isdir(log_path):
        if os.path.exists(os.path.join(log_path, 'latest_model.pth')):
            return

        if cfg.OVERRIDE:
            shutil.rmtree(log_path)
            os.makedirs(log_path)
        else:
            if os.path.exists(os.path.join(log_path, 'done')):
                print('Already trained, exit')
                exit()
            else:
                shutil.rmtree(log_path)
                os.makedirs(log_path)
    else:
        os.makedirs(log_path)


# [New from MaskSiam] 测试阶段的日志初始化
def init_log_test(args, cfg, log_path, train_loader_length, dataset_size):
    assert cfg.ALGORITHM in ALL_METHODS
    if not cfg.RANDOM:
        setup_seed(cfg.SEED)

    init_output_foler_test(cfg, log_path)
    writer = SummaryWriter(os.path.join(log_path, 'tensorboard'))
    writer.add_text('config', str(args))
    logging.basicConfig(filename=log_path + '_test_{}.txt'.format(cfg.DATASET.TARGET_DOMAINS), level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info("{} iterations per epoch".format(train_loader_length))
    logging.info(
        "We have {} images in train set, {} images in val set, and {} images in test set.".format(dataset_size[0],
                                                                                                  dataset_size[1],
                                                                                                  dataset_size[2]))
    logging.info(str(args))
    logging.info(str(cfg))
    return writer


def init_output_foler_test(cfg, log_path):
    if os.path.isdir(log_path):
        if cfg.OVERRIDE:
            shutil.rmtree(log_path)
    else:
        os.makedirs(log_path)


# [New from MaskSiam] 手动调整学习率辅助函数
def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    if epoch <= args.warmup_epoch:
        cur_lr = init_lr * epoch / args.warmup_epoch
    else:
        cur_lr = init_lr * 0.5 * (
                    1. + math.cos(math.pi * (epoch - args.warmup_epoch - 1) / (args.epochs - args.warmup_epoch)))

    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


# [Original] 保留原版调度器，用于兼容旧算法
def get_scheduler(optimizer, max_epoch):
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch * 0.5], gamma=0.1)
    return scheduler


# [New from MaskSiam] MASK_SIAM 专用的调度器
def get_scheduler_siam(optimizer, init_lr, epoch, max_epoch):
    warmup_epoch = 30
    if epoch <= warmup_epoch:
        cur_lr = init_lr * epoch / warmup_epoch
    else:
        cur_lr = init_lr  # MASK_SIAM 似乎在这个版本中禁用了后续的 cosine decay，保持恒定，严格按照提供的代码复刻

    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


# [New from MaskSiam] MASK_SIAM 专用的 Writer 更新，记录详细 Loss
def update_writer_siam(writer, epoch, loss_dict_iter):
    logging.info('epoch: {}, total loss: {}'.format(epoch, loss_dict_iter['loss']))
    logging.info('epoch: {}, sup loss: {}'.format(epoch, loss_dict_iter['loss_sup']))
    # logging.info('epoch: {}, sup1 loss: {}'.format(epoch, loss_dict_iter['loss1_sup']))
    logging.info('epoch: {}, sup2 loss: {}'.format(epoch, loss_dict_iter['loss2_sup']))
    logging.info('epoch: {}, sup3 loss: {}'.format(epoch, loss_dict_iter['loss3_sup']))
    logging.info('epoch: {}, loss_siam_fastmoco: {}'.format(epoch, loss_dict_iter['loss_siam_fastmoco']))


# [New from MaskSiam] 另一个版本的 Writer 更新
def update_writer_siam_v2(writer, epoch, scheduler, loss_dict_iter):
    logging.info('epoch: {}, total loss: {}'.format(epoch, loss_dict_iter['loss']))

    logging.info('epoch: {}, sup loss: {}'.format(epoch, loss_dict_iter['loss_sup']))
    logging.info('epoch: {}, mask loss: {}'.format(epoch, loss_dict_iter['loss_sup_mask']))
    logging.info('epoch: {}, feat loss: {}'.format(epoch, loss_dict_iter['loss_feat']))
    logging.info('epoch: {}, loss_siam_fastmoco1: {}'.format(epoch, loss_dict_iter['loss_siam_fastmoco1']))
    logging.info('epoch: {}, loss_siam_fastmoco2: {}'.format(epoch, loss_dict_iter['loss_siam_fastmoco2']))

    logging.info('lr: {}, learning rate: {}'.format(epoch, scheduler.get_last_lr()[0]))

    writer.add_scalar('info/lr', scheduler.get_last_lr()[0], epoch)
    writer.add_scalar('info/loss', loss_dict_iter['loss'], epoch)


# [Merged] 通用算法的 Writer 更新，增加了 logging 输出
def update_writer(writer, epoch, scheduler, loss_avg):
    logging.info('epoch: {}, total loss: {}'.format(epoch, loss_avg.mean()))
    logging.info('lr: {}, learning rate: {}'.format(epoch, scheduler.get_last_lr()[0]))
    writer.add_scalar('info/lr', scheduler.get_last_lr()[0], epoch)
    writer.add_scalar('info/loss', loss_avg.mean(), epoch)


class MovingAverage:
    def __init__(self, ema, oneminusema_correction=True):
        self.ema = ema
        self.named_parameters = {}
        self._updates = 0
        self._oneminusema_correction = oneminusema_correction

    def update(self, dict_data):
        ema_dict_data = {}
        for name, data in dict_data.items():
            data = data.view(1, -1)
            if self._updates == 0:
                previous_data = torch.zeros_like(data)
            else:
                previous_data = self.named_parameters[name]

            ema_data = self.ema * previous_data + (1 - self.ema) * data
            if self._oneminusema_correction:
                ema_dict_data[name] = ema_data / (1 - self.ema)
            else:
                ema_dict_data[name] = ema_data
            self.named_parameters[name] = ema_data.clone().detach()

        self._updates += 1
        return ema_dict_data


class LossCounter:
    def __init__(self, start=0):
        self.sum = start
        self.iteration = 0

    def update(self, num):
        self.sum += num
        self.iteration += 1

    def mean(self):
        return self.sum * 1.0 / self.iteration