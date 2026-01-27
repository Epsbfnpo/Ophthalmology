#!/bin/bash

# ================= 配置区域 =================
# 数据集根目录
ROOT_DATA="/datasets/work/hb-nhmrc-dhcp/work/liu275/DGDR/GDR_Formatted_Data"

# 输出目录
OUTPUT_DIR="./output_esdg_h100"

# 使用的算法
ALGORITHM="GDRNet"

# ESDG (Extreme Single-Domain Generalization) 核心设置
# 必须只指定 1 个源域
SOURCE="MESSIDOR"

# 目标域 (仅用于测试)
TARGETS="APTOS DDR DEEPDR FGADR IDRID RLDR"

# ===========================================

# 自动获取当前节点 GPU 数量
NUM_GPUS=${SLURM_GPUS_ON_NODE:-4}

echo "========================================================"
echo "🚀 启动 ESDG 训练任务"
echo "GPU 数量: $NUM_GPUS"
echo "算法: $ALGORITHM"
echo "源域 (Train): $SOURCE"
echo "目标域 (Test): $TARGETS"
echo "========================================================"

# 使用 torchrun 启动 DDP
# --dg_mode ESDG : 确保加载 GDRBench_ESDG.yaml (针对单域优化的参数)
# --batch-size 64 : 适配 H100 大显存
torchrun --nproc_per_node=$NUM_GPUS \
    --master_port=29505 \
    main.py \
    --root $ROOT_DATA \
    --algorithm $ALGORITHM \
    --dg_mode ESDG \
    --source-domains $SOURCE \
    --target-domains $TARGETS \
    --output $OUTPUT_DIR \
    --batch-size 64 \
    --epochs 100