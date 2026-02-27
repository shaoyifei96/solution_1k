#!/bin/bash
#SBATCH --job-name=preprocess_b1k
#SBATCH --requeue
#SBATCH --time=10:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=0
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=./logs/%x-%j.out
#SBATCH --error=./logs/%x-%j.err

# --- Configuration (edit these) ---
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-$PROJECT_DIR/../envs/b1k_solution}"
CONFIG="${CONFIG:-pi_behavior_b1k_fast}"
NUM_WORKERS="${NUM_WORKERS:-64}"
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
# Set WANDB_API_KEY in your environment before submitting

uv run scripts/compute_norm_stats.py \
  --config-name "$CONFIG" \
  --correlation \
  --num-workers "$NUM_WORKERS" \
  $EXTRA_ARGS
