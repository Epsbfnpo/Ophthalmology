#!/bin/bash -l

#SBATCH --job-name=hc-gdrnet
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --account=OD-210257
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=4
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=2:00:00

set -euo pipefail

echo "=============================================================="
echo "HC-GDRNet TRAINING JOB"
echo "Job $SLURM_JOB_ID started at $(date)"
echo "=============================================================="

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_SOCKET_IFNAME=^docker0,lo
export TORCH_ALLOW_TF32=1
export CUBLAS_ALLOW_TF32=1

source /datasets/work/hb-nhmrc-dhcp/work/liu275/miniconda3/bin/activate
conda activate brainseg

REPO_ROOT=$(pwd)
echo "Working Directory: $REPO_ROOT"

set +e
bash ./main_hc.sh
STATUS=$?
set -e

echo "Job finished with status $STATUS"

if [ "$STATUS" -eq 3 ]; then
    echo "⏳ Time limit hit or specific error 3. Resubmitting job."
    exit 0
fi

if [ "$STATUS" -ne 0 ]; then
    echo "❌ Job failed!"
    exit $STATUS
fi