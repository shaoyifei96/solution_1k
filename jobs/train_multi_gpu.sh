#!/bin/bash
#SBATCH --job-name=train_b1k_multi
#SBATCH --requeue
#SBATCH --time=4-00:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=8
#SBATCH --cpus-per-task=112
#SBATCH --mem=1900G
#SBATCH --output=./logs/%x-%j.out
#SBATCH --error=./logs/%x-%j.err

# --- Configuration (edit these) ---
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-$PROJECT_DIR/../envs/b1k_solution}"
CONFIG="${CONFIG:-pi_behavior_b1k_fast}"
FSDP_DEVICES="${FSDP_DEVICES:-8}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
NUM_STEPS="${NUM_STEPS:-20000000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-2000}"
KEEP_PERIOD="${KEEP_PERIOD:-20000}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
# ----------------------------------

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

cd "$PROJECT_DIR"
mkdir -p logs

hostname
export SSL_CERT_DIR=/etc/ssl/certs
export TMPDIR=/tmp
export TORCHINDUCTOR_COMPILE_THREADS=1
export UV_LINK_MODE=copy
export UV_NO_CACHE=1
# Set WANDB_API_KEY in your environment before submitting

uv run scripts/train.py "$CONFIG" \
  --batch_size="$BATCH_SIZE" \
  --num_train_steps="$NUM_STEPS" \
  --fsdp_devices="$FSDP_DEVICES" \
  --save_interval="$SAVE_INTERVAL" \
  --keep_period="$KEEP_PERIOD" \
  --log_interval=150 \
  $EXTRA_ARGS
