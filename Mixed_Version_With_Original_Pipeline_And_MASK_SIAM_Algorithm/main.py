import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import random
import time
import logging
import datetime
import sys
from tqdm import tqdm

import algorithms
from utils.args import get_args, setup_cfg
# [Modified] Added get_scheduler_siam and update_writer_siam for MASK_SIAM
from utils.misc import init_log, LossCounter, get_scheduler, update_writer, get_scheduler_siam, update_writer_siam
from utils.validate import algorithm_validate
from dataset.data_manager import get_dataset


def debug_log(msg, rank):
    """强制打印带时间戳和Rank的调试信息"""
    timestamp = datetime.datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}][Rank {rank}] {msg}", flush=True)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def save_checkpoint(path, algorithm, optimizer, scheduler, epoch, best_performance):
    """保存完整训练状态"""
    if hasattr(algorithm.network, 'module'):
        model_state = algorithm.network.module.state_dict()
    else:
        model_state = algorithm.network.state_dict()

    state = {
        'epoch': epoch,
        'model_state': model_state,
        'optimizer_state': optimizer.state_dict(),
        'best_performance': best_performance
    }
    # Scheduler might be None for MASK_SIAM (handled manually)
    if scheduler is not None:
        state['scheduler_state'] = scheduler.state_dict()

    torch.save(state, path)


def main():
    args = get_args()
    cfg = setup_cfg(args)

    # --- 1. 初始化设置 ---
    log_path = os.path.join(cfg.OUT_DIR, cfg.ALGORITHM, cfg.DATASET.ROOT.split('/')[-1], cfg.DATASET.TARGET_DOMAINS[0])

    # DDP 初始化
    if args.local_rank != -1:
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(args.local_rank)
        is_distributed = True
    else:
        is_distributed = False

    # 仅主进程初始化日志和 Tensorboard
    writer = None
    if args.local_rank in [-1, 0]:
        train_loader_dummy, _, _ = get_dataset(args, cfg)  # Just to get length if needed, or rely on later
        # Re-get dataset properly later
        pass

    # --- 2. 数据集准备 ---
    debug_log(f"Loading dataset... {cfg.DATASET.ROOT}", args.local_rank)
    train_loader, val_loader, test_loader = get_dataset(args, cfg)

    if args.local_rank in [-1, 0]:
        writer = init_log(args, cfg, log_path, len(train_loader),
                          [len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset)])

    # --- 3. 模型与算法构建 ---
    debug_log(f"Building algorithm: {cfg.ALGORITHM}", args.local_rank)

    # 注册算法：通过 algorithms.get_algorithm_class 获取类
    # 确保 algorithms.py 中 ALGORITHMS 列表包含了 GDRNet_MASK_SIAM
    algorithm_class = algorithms.get_algorithm_class(cfg.ALGORITHM)
    algorithm = algorithm_class(cfg.DATASET.NUM_CLASSES, cfg)

    if is_distributed:
        algorithm.cuda()
        algorithm.network = DDP(algorithm.network, device_ids=[args.local_rank], output_device=args.local_rank,
                                find_unused_parameters=True)
        # 如果有 classifier 等其他模块，通常也需要处理，或者整个 algorithm 作为一个 Module 放入 DDP
        # 但这里通常 GDRNet 结构比较松散。如果 Algorithm 继承自 nn.Module，最好直接 wrap algorithm
        # 但原始代码似乎只 wrap 了 network。这里保持原逻辑，注意 synchronize。
    else:
        algorithm.cuda()

    # 优化器
    optimizer = algorithm.optimizer

    # [Modified] Scheduler Selection
    # MASK_SIAM uses a custom function inside the loop, not a standard scheduler object
    if cfg.ALGORITHM == 'GDRNet_MASK_SIAM':
        scheduler = None
    else:
        scheduler = get_scheduler(optimizer, cfg.EPOCHS)

    # --- 4. 恢复训练 (Resume) ---
    latest_ckpt_path = os.path.join(log_path, 'latest_model.pth')
    final_ckpt_path = os.path.join(log_path, 'final_model.pth')
    best_performance = -1
    start_epoch = 1

    if os.path.exists(latest_ckpt_path):
        debug_log(f"Resuming from {latest_ckpt_path}", args.local_rank)
        checkpoint = torch.load(latest_ckpt_path, map_location='cpu')

        # 加载模型参数
        if hasattr(algorithm.network, 'module'):
            algorithm.network.module.load_state_dict(checkpoint['model_state'])
        else:
            algorithm.network.load_state_dict(checkpoint['model_state'])

        optimizer.load_state_dict(checkpoint['optimizer_state'])

        if scheduler is not None and 'scheduler_state' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state'])

        start_epoch = checkpoint['epoch'] + 1
        best_performance = checkpoint.get('best_performance', -1)

    # --- 5. 训练主循环 ---
    algorithm.train()
    start_time = time.time()

    debug_log(f"Start training from epoch {start_epoch}", args.local_rank)

    for epoch in range(start_epoch, cfg.EPOCHS + 1):
        if is_distributed:
            train_loader.sampler.set_epoch(epoch)

        # [Modified] MASK_SIAM Specific Scheduler Update at Start of Epoch
        if cfg.ALGORITHM == 'GDRNet_MASK_SIAM':
            get_scheduler_siam(optimizer, cfg.LEARNING_RATE, epoch, cfg.EPOCHS)
            # Log LR for debug
            if args.local_rank in [-1, 0]:
                current_lr = optimizer.param_groups[0]['lr']
                # logging.info(f"Epoch {epoch} LR set to {current_lr}")

        # Update Alpha/Hyperparams
        algorithm.update_epoch(epoch)

        loss_avg = LossCounter()

        # 使用 tqdm 显示进度
        if args.local_rank in [-1, 0]:
            loader_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.EPOCHS}", leave=True, dynamic_ncols=True)
        else:
            loader_bar = train_loader

        for i, batch in enumerate(loader_bar):
            # [Modified] Data Unpacking Logic
            # Handle 5-tuple (Frequency Augmentation) vs 5-tuple (Standard Index)
            if cfg.ALGORITHM == 'GDRNet_MASK_SIAM' and cfg.TRANSFORM.FREQ:
                # 明确的 5 元组解包：(Image, FreqImage, Mask, Label, Domain)
                image, image_freq, mask, label, domain = batch

                # 构建 Minibatch，必须包含 image_freq
                minibatch = [image.cuda(), image_freq.cuda(), mask.cuda(), label.cuda(), domain.cuda()]

            else:
                # 原始逻辑：(Image, Mask, Label, Domain, Index)
                image, mask, label, domain, index = batch
                minibatch = [image.cuda(), mask.cuda(), label.cuda(),
                             domain.cuda()]  # Update usually expects 4 items for standard algos

            # Algorithm Update
            loss_dict_iter = algorithm.update(minibatch)
            loss_avg.update(loss_dict_iter['loss'])

            # TQDM Postfix
            if args.local_rank in [-1, 0]:
                loader_bar.set_postfix({'loss': f"{loss_dict_iter['loss']:.4f}"})

        # --- End of Epoch Operations ---

        # Logging & Tensorboard
        if args.local_rank in [-1, 0]:
            if writer:
                # [Modified] Use specialized writer for MASK_SIAM to log detailed losses
                if cfg.ALGORITHM == 'GDRNet_MASK_SIAM':
                    # Pseudo-scheduler object for logging purposes or just pass optimizer
                    # update_writer_siam expects (writer, epoch, loss_dict)
                    update_writer_siam(writer, epoch, loss_dict_iter)
                    # Log LR manually
                    writer.add_scalar('info/lr', optimizer.param_groups[0]['lr'], epoch)
                else:
                    update_writer(writer, epoch, scheduler, loss_avg)

        # Validation (per interval)
        if epoch % cfg.VAL_EPOCH == 0 or epoch == cfg.EPOCHS:
            debug_log(f"Validating at epoch {epoch}...", args.local_rank)

            # 临时切换为 eval 模式
            network_backup = algorithm.network
            if is_distributed and hasattr(algorithm.network, 'module'):
                # Validate usually expects single model, not DDP wrapper directly if generic
                # But algorithm_validate calls algorithm.predict which calls self.network
                pass

            auc, _ = algorithm.validate(val_loader, test_loader, writer)

            if args.local_rank in [-1, 0]:
                if auc > best_performance:
                    best_performance = auc
                    save_checkpoint(os.path.join(log_path, 'best_model.pth'), algorithm, optimizer, scheduler, epoch,
                                    best_performance)
                    logging.info(f"New Best AUC: {best_performance:.4f}")

                # Save Latest
                save_checkpoint(latest_ckpt_path, algorithm, optimizer, scheduler, epoch, best_performance)

        # Standard Scheduler Step (Only for non-MASK_SIAM)
        if scheduler is not None:
            scheduler.step()

        # Time Limit Check (for clusters)
        if args.time_limit > 0:
            elapsed_time = time.time() - start_time
            if elapsed_time > (args.time_limit - 300):
                debug_log(f"⏰ Time limit reached. Saving latest and exiting cleanly (Code 0).", args.local_rank)

                if args.local_rank in [-1, 0]:
                    save_checkpoint(latest_ckpt_path, algorithm, optimizer, scheduler, epoch, best_performance)
                    if writer: writer.close()

                if is_distributed:
                    dist.barrier()
                    dist.destroy_process_group()

                sys.exit(0)

    # --- 6. 正常结束 ---
    debug_log("All epochs finished.", args.local_rank)

    if args.local_rank in [-1, 0]:
        logging.info("Saving Final Checkpoint...")
        save_checkpoint(final_ckpt_path, algorithm, optimizer, scheduler, cfg.EPOCHS, best_performance)

        # 最终评估
        algorithm.renew_model(log_path)
        _, test_auc = algorithm.validate(val_loader, test_loader, writer)
        logging.info(f"Final Test AUC: {test_auc:.4f}")

        with open(os.path.join(log_path, 'done'), 'w') as f:
            f.write('done')
        if writer: writer.close()

    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()