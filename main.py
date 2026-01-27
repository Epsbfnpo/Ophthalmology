import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import random
import time
import logging
import datetime  # 新增：用于打印精确时间
import sys  # 新增：用于强制刷新缓冲区
from tqdm import tqdm

import algorithms
from utils.args import get_args, setup_cfg
from utils.misc import init_log, LossCounter, get_scheduler, update_writer
from utils.validate import algorithm_validate
from dataset.data_manager import get_dataset


# --- Debug 辅助函数 ---
def debug_log(msg, rank):
    """强制打印带时间戳和Rank的调试信息，确保不被缓存"""
    timestamp = datetime.datetime.now().strftime('%H:%M:%S')
    # 使用 print 并强制 flush，防止日志卡在缓冲区看不到
    print(f"[{timestamp}][Rank {rank}] {msg}", flush=True)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    args = get_args()

    # --- 1. DDP 初始化 ---
    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])

    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl')
        args.device = torch.device('cuda', args.local_rank)
        is_distributed = True
    else:
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        is_distributed = False
        print("Running in Single GPU Mode.")

    debug_log("DDP initialized. Configuration setup starting...", args.local_rank)

    # --- 2. 配置加载 ---
    cfg = setup_cfg(args)
    log_path = os.path.join(args.output, cfg.OUTPUT_PATH)

    # --- 3. 日志初始化 (仅主进程) ---
    writer = None
    if args.local_rank in [-1, 0]:
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        writer = init_log(args, cfg, log_path, 0, [0, 0, 0])
        logging.info(f"Distributed: {is_distributed}, Rank: {args.local_rank}")

    set_seed(args.seed)

    # --- 4. 数据加载 ---
    debug_log("Loading datasets...", args.local_rank)
    train_loader, val_loader, test_loader, dataset_size, train_sampler = get_dataset(args, cfg)
    debug_log(f"Datasets loaded. Sizes: {dataset_size}", args.local_rank)

    # --- 5. 模型初始化 ---
    debug_log("Initializing algorithm/model...", args.local_rank)
    algorithm_class = algorithms.get_algorithm_class(cfg.ALGORITHM)
    algorithm = algorithm_class(cfg.DATASET.NUM_CLASSES, cfg)
    algorithm.network = algorithm.network.to(args.device)

    if hasattr(algorithm, 'classifier'):
        algorithm.classifier = algorithm.classifier.to(args.device)
    if hasattr(algorithm, 'swad_algorithm'):
        algorithm.swad_algorithm = algorithm.swad_algorithm.to(args.device)

    # DDP 模型包装
    if is_distributed:
        debug_log("Converting to SyncBatchNorm and wrapping with DDP...", args.local_rank)
        algorithm.network = nn.SyncBatchNorm.convert_sync_batchnorm(algorithm.network)
        algorithm.network = DDP(algorithm.network, device_ids=[args.local_rank], output_device=args.local_rank)

    # --- 6. 训练循环 ---
    iterator = tqdm(range(cfg.EPOCHS), disable=(args.local_rank not in [-1, 0]))
    scheduler = get_scheduler(algorithm.optimizer, cfg.EPOCHS)
    best_performance = 0.0

    debug_log("Starting training loop...", args.local_rank)

    for i in iterator:
        epoch = i + 1
        debug_log(f"--- Start Epoch {epoch} ---", args.local_rank)

        # DDP Sampler Set Epoch
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(i)

        loss_avg = LossCounter()
        algorithm.train()

        # 训练 Step
        # debug_log(f"Epoch {epoch}: Training Loop Start", args.local_rank)
        for image, mask, label, domain, img_index in train_loader:
            image = image.to(args.device)
            mask = mask.to(args.device)
            label = label.to(args.device).long()
            domain = domain.to(args.device).long()

            minibatch = [image, mask, label, domain]
            loss_dict_iter = algorithm.update(minibatch)
            loss_avg.update(loss_dict_iter['loss'])

        # debug_log(f"Epoch {epoch}: Training Loop End", args.local_rank)

        if hasattr(algorithm, 'update_epoch'):
            algorithm.update_epoch(epoch)

        if args.local_rank in [-1, 0]:
            update_writer(writer, epoch, scheduler, loss_avg)

        scheduler.step()
        debug_log(f"Epoch {epoch}: Training & Scheduler Step Done", args.local_rank)

        # --- 验证逻辑 ---
        if epoch % cfg.VAL_EPOCH == 0:
            debug_log(f"Epoch {epoch}: Reached Validation Block Check", args.local_rank)

            # Rank 0 进入验证逻辑
            if args.local_rank in [-1, 0]:
                debug_log(f"Epoch {epoch}: Rank 0 entering validation...", args.local_rank)
                logging.info(f"Epoch {epoch} Validation...")

                # 1. 解包 DDP
                debug_log(f"Epoch {epoch}: Unwrapping DDP model...", args.local_rank)
                network_backup = algorithm.network
                if is_distributed and hasattr(algorithm.network, 'module'):
                    algorithm.network = algorithm.network.module
                    debug_log(f"Epoch {epoch}: Unwrapping SUCCESS", args.local_rank)
                else:
                    debug_log(f"Epoch {epoch}: No DDP module to unwrap or already unwrapped", args.local_rank)

                # 2. 验证 Val Set
                debug_log(f"Epoch {epoch}: Running validation on VAL set...", args.local_rank)
                val_auc, test_auc = algorithm_validate(algorithm, val_loader, writer, epoch, 'val')
                debug_log(f"Epoch {epoch}: VAL set done. Result: {val_auc}", args.local_rank)

                # 3. 验证 Test Set (这是最耗时的地方)
                debug_log(f"Epoch {epoch}: Running validation on TEST set (this may take a while)...", args.local_rank)
                _, _ = algorithm_validate(algorithm, test_loader, writer, epoch, 'test')
                debug_log(f"Epoch {epoch}: TEST set done.", args.local_rank)

                if val_auc > best_performance and epoch > cfg.EPOCHS * 0.1:
                    best_performance = val_auc
                    logging.info(f"New Best Model! Val AUC: {val_auc:.4f}")
                    debug_log(f"Epoch {epoch}: Saving best model...", args.local_rank)
                    algorithm.save_model(log_path)
                    debug_log(f"Epoch {epoch}: Model saved.", args.local_rank)

                # 4. 恢复 DDP
                if is_distributed:
                    debug_log(f"Epoch {epoch}: Rewrapping DDP model...", args.local_rank)
                    algorithm.network = network_backup
                    debug_log(f"Epoch {epoch}: Rewrapping SUCCESS", args.local_rank)

            else:
                # Rank 1-3
                debug_log(f"Epoch {epoch}: Waiting at Barrier (I am not Rank 0)...", args.local_rank)

            # --- 全局同步 Barrier ---
            if is_distributed:
                debug_log(f"Epoch {epoch}: Entering dist.barrier() - Waiting for all ranks...", args.local_rank)
                dist.barrier()
                debug_log(f"Epoch {epoch}: Passed dist.barrier() - All ranks synced!", args.local_rank)

    # --- 7. 结束与最终测试 ---
    debug_log("Training Loop Finished.", args.local_rank)

    if args.local_rank in [-1, 0]:
        logging.info("Training Finished. Evaluating Best Model on Targets...")
        debug_log("Starting Final Evaluation...", args.local_rank)

        network_backup = algorithm.network
        if is_distributed and hasattr(algorithm.network, 'module'):
            algorithm.network = algorithm.network.module
            debug_log("Final Eval: Unwrapped DDP model.", args.local_rank)

        algorithm.renew_model(log_path)
        debug_log("Final Eval: Best model reloaded.", args.local_rank)

        _, test_auc = algorithm.validate(val_loader, test_loader, writer)
        logging.info(f"Final Test AUC (Avg across targets): {test_auc:.4f}")
        debug_log(f"Final Eval: Done. AUC: {test_auc}", args.local_rank)

        with open(os.path.join(log_path, 'done'), 'w') as f:
            f.write('done')
        writer.close()

        # 恢复 (逻辑完整性)
        if is_distributed:
            algorithm.network = network_backup

    if is_distributed:
        debug_log("Final Barrier before destroy_process_group...", args.local_rank)
        dist.barrier()
        dist.destroy_process_group()
        debug_log("Process group destroyed. Bye!", args.local_rank)


if __name__ == "__main__":
    main()