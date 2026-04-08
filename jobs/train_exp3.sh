#!/bin/bash
#SBATCH --job-name=exp3-v2-deep-sets
#SBATCH --account=kumar-lab
#SBATCH --partition=dgx-b200
#SBATCH --gpus-per-node=8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=112
#SBATCH --mem=1900G
#SBATCH --time=1-00:00:00
#SBATCH --output=/vast/projects/kumar/lab/yishao/b1k_2/logs/exp3-%j.out
#SBATCH --error=/vast/projects/kumar/lab/yishao/b1k_2/logs/exp3-%j.err

PROJECT_DIR="/vast/projects/kumar/lab/yishao/b1k_2"
CONDA_ENV="/vast/projects/kumar/lab/yishao/envs/b1k_solution"

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

cd "$PROJECT_DIR"
mkdir -p logs

export PYTHONUNBUFFERED=1
export SSL_CERT_DIR=/etc/ssl/certs
export TMPDIR=/tmp
export TORCHINDUCTOR_COMPILE_THREADS=1
export UV_LINK_MODE=copy
export UV_NO_CACHE=1
export PATH="/vast/home/y/yishao/.local/bin:$PATH"

hostname
nvidia-smi -L
echo "=== EXP3: V2 data + V2 deep sets encoder, 10K steps, bs=1024, 8x B200 ==="
echo "Start: $(date)"

uv run scripts/train.py pi_behavior_b1k_exp3 \
  --batch_size 1024 \
  --fsdp_devices 8 \
  --log_interval 100 \
  --exp_name "exp3_v2_deep_sets"

echo "=== EXP3 done at $(date) ==="
