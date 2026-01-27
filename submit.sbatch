#!/bin/bash -l

#SBATCH --job-name=gdr-esdg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --account=OD-210257
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=4
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=6:00:00

set -euo pipefail

echo "Job $SLURM_JOB_ID started at $(date)"

# 环境变量优化
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
# 解决 PyTorch 显存碎片问题
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 确保 NCCL 通信正常 (根据集群网络情况可能需要调整)
export NCCL_SOCKET_IFNAME=^docker0,lo
# 启用 TF32 加速 (H100 默认支持，显式开启更保险)
export TORCH_ALLOW_TF32=1
export CUBLAS_ALLOW_TF32=1

# 激活环境 (请根据实际路径修改)
source /datasets/work/hb-nhmrc-dhcp/work/liu275/miniconda3/bin/activate
conda activate brainseg

# 设置脚本权限并运行
chmod +x ./run_esdg.sh
./run_esdg.sh

echo "Job finished at $(date)"