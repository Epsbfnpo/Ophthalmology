import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import random
import time
import logging
from tqdm import tqdm

import algorithms
from utils.args import get_args, setup_cfg
from utils.misc import init_log, LossCounter, get_scheduler, update_writer
from utils.validate import algorithm_validate
from dataset.data_manager import get_dataset


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

    # --- 2. 配置加载 ---
    # setup_cfg 内部已经处理了 args.batch_size 的覆盖，并冻结了 cfg
    cfg = setup_cfg(args)

    log_path = os.path.join(args.output, cfg.OUTPUT_PATH)

    # --- 3. 日志初始化 (仅主进程) ---
    writer = None
    if args.local_rank in [-1, 0]:
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        # 初始化 logging 和 Tensorboard
        # 注意：dataset_size 暂时填占位符 [0,0,0]，稍后不影响
        writer = init_log(args, cfg, log_path, 0, [0, 0, 0])
        logging.info(f"Distributed: {is_distributed}, Rank: {args.local_rank}")
        logging.info(f"Source Domain: {args.source_domains}")
        logging.info(f"Target Domains: {args.target_domains}")

    set_seed(args.seed)

    # --- 4. 数据加载 ---
    # get_dataset 已经适配了 DDP Sampler
    train_loader, val_loader, test_loader, dataset_size, train_sampler = get_dataset(args, cfg)

    if args.local_rank in [-1, 0]:
        logging.info(f"Dataset Size: {dataset_size}")

    # --- 5. 模型初始化 ---
    algorithm_class = algorithms.get_algorithm_class(cfg.ALGORITHM)
    algorithm = algorithm_class(cfg.DATASET.NUM_CLASSES, cfg)
    algorithm.network = algorithm.network.to(args.device)

    # 将子模块也移动到 device (针对不同算法结构)
    if hasattr(algorithm, 'classifier'):
        algorithm.classifier = algorithm.classifier.to(args.device)
    if hasattr(algorithm, 'swad_algorithm'):
        algorithm.swad_algorithm = algorithm.swad_algorithm.to(args.device)

    # DDP 模型包装
    if is_distributed:
        algorithm.network = nn.SyncBatchNorm.convert_sync_batchnorm(algorithm.network)
        algorithm.network = DDP(algorithm.network, device_ids=[args.local_rank], output_device=args.local_rank)

    # --- 6. 训练循环 ---
    # 仅主进程显示 tqdm 进度条
    iterator = tqdm(range(cfg.EPOCHS), disable=(args.local_rank not in [-1, 0]))
    scheduler = get_scheduler(algorithm.optimizer, cfg.EPOCHS)
    best_performance = 0.0

    for i in iterator:
        epoch = i + 1

        # DDP 必须设置 sampler epoch
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(i)

        loss_avg = LossCounter()
        algorithm.train()

        # 训练 Step
        for image, mask, label, domain, img_index in train_loader:
            image = image.to(args.device)
            mask = mask.to(args.device)
            label = label.to(args.device).long()
            domain = domain.to(args.device).long()

            minibatch = [image, mask, label, domain]
            loss_dict_iter = algorithm.update(minibatch)
            loss_avg.update(loss_dict_iter['loss'])

        # 更新 alpha (GDRNet 特有)
        # 处理 DDP 包装后的 .module 访问
        algo_ptr = algorithm.module if is_distributed and hasattr(algorithm, 'module') else algorithm
        if hasattr(algo_ptr, 'update_epoch'):
            algo_ptr.update_epoch(epoch)

        # 记录日志 (仅主进程)
        if args.local_rank in [-1, 0]:
            # update_writer 内部会调用 logging.info 打印 epoch loss
            update_writer(writer, epoch, scheduler, loss_avg)

        scheduler.step()

        # --- 验证与保存 (建议仅在 Rank 0 进行) ---
        if epoch % cfg.VAL_EPOCH == 0:
            if args.local_rank in [-1, 0]:
                logging.info(f"Epoch {epoch} Validation...")
                # 验证函数通常不支持 DDP 模型，传入解包后的 algo_ptr
                val_auc, test_auc = algorithm_validate(algo_ptr, val_loader, writer, epoch, 'val')
                # 顺便跑一下 Test 集看看进度
                _, _ = algorithm_validate(algo_ptr, test_loader, writer, epoch, 'test')

                if val_auc > best_performance and epoch > cfg.EPOCHS * 0.1:
                    best_performance = val_auc
                    logging.info(f"New Best Model! Val AUC: {val_auc:.4f}")
                    algo_ptr.save_model(log_path)

            # 进程同步
            # if is_distributed:
            #    dist.barrier()

    # --- 7. 结束与最终测试 ---
    if args.local_rank in [-1, 0]:
        logging.info("Training Finished. Evaluating Best Model on Targets...")
        algo_ptr = algorithm.module if is_distributed and hasattr(algorithm, 'module') else algorithm
        algo_ptr.renew_model(log_path)
        _, test_auc = algo_ptr.validate(val_loader, test_loader, writer)
        logging.info(f"Final Test AUC (Avg across targets): {test_auc:.4f}")

        with open(os.path.join(log_path, 'done'), 'w') as f:
            f.write('done')
        writer.close()

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()