#!/bin/bash -l

#SBATCH --job-name=hc-gdrnet     # 修改了任务名称
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32       # 保持原配置
#SBATCH --mem=128G               # 保持原配置
#SBATCH --account=OD-210257      # 保持原配置 (计费账号)
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=4        # 申请 4 张 H100
#SBATCH --output=%x-%j.out       # 日志文件命名格式
#SBATCH --error=%x-%j.err
#SBATCH --time=24:00:00          # 最大运行时间

set -euo pipefail

echo "=============================================================="
echo "HC-GDRNet TRAINING JOB"
echo "Job $SLURM_JOB_ID started at $(date)"
echo "=============================================================="

# Environment Setup (1:1 复刻原配置)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_SOCKET_IFNAME=^docker0,lo
export TORCH_ALLOW_TF32=1
export CUBLAS_ALLOW_TF32=1

# Activate Conda (1:1 复刻原路径)
# 注意：这假设您的用户名和环境路径与示例一致
source /datasets/work/hb-nhmrc-dhcp/work/liu275/miniconda3/bin/activate
conda activate brainseg

# 设置项目根目录为当前提交目录
REPO_ROOT=$(pwd)
echo "Working Directory: $REPO_ROOT"

# 执行训练脚本
# 这里直接调用您的 main_hc.sh
# 确保 main_hc.sh 有执行权限 (chmod +x main_hc.sh)
set +e
bash ./main_hc.sh
STATUS=$?
set -e

echo "Job finished with status $STATUS"

# 错误处理与自动重试逻辑 (参考原脚本)
if [ "$STATUS" -eq 3 ]; then
    echo "⏳ Time limit hit or specific error 3. Resubmitting job."
    # 如果您的代码支持断点续训，可以保留这行；否则建议注释掉以免无限重启
    # sbatch --export=ALL "$0"
    exit 0
fi

if [ "$STATUS" -ne 0 ]; then
    echo "❌ Job failed!"
    exit $STATUS
fi