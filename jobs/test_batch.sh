#!/bin/bash
#SBATCH --job-name=test_batch
#SBATCH --account=kumar-lab
#SBATCH --partition=dgx-b200
#SBATCH --gpus=8
#SBATCH --cpus-per-task=112
#SBATCH --mem=1900G
#SBATCH --time=02:00:00
#SBATCH --output=/vast/projects/kumar/lab/yishao/b1k_2/logs/test_batch-%j.out
#SBATCH --error=/vast/projects/kumar/lab/yishao/b1k_2/logs/test_batch-%j.err

PROJECT_DIR="/vast/projects/kumar/lab/yishao/b1k_2"
CONDA_ENV="/vast/projects/kumar/lab/yishao/envs/b1k_solution"
CONFIG="pi_behavior_b1k_fast"
BATCH_SIZE="${1:-1024}"

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
echo "=== Batch size test: bs=${BATCH_SIZE}, 8x B200 ==="
echo "Start: $(date)"

uv run scripts/train.py "$CONFIG" \
  --batch_size "$BATCH_SIZE" \
  --num_train_steps 20 \
  --fsdp_devices 8 \
  --save_interval 9999 \
  --keep_period 9999 \
  --log_interval 5 \
  --exp_name "test_batch_${BATCH_SIZE}" \
  --overwrite

echo "=== Done at $(date) ==="
nvidia-smi
