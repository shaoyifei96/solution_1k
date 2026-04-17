#!/bin/bash
#SBATCH --job-name=exp1-expanded
#SBATCH --account=kumar-lab
#SBATCH --partition=dgx-b200
#SBATCH --gpus-per-node=8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=112
#SBATCH --mem=1900G
#SBATCH --time=15:00:00
#SBATCH --output=/vast/projects/kumar/lab/yishao/b1k_2/logs/exp1-expanded-%j.out
#SBATCH --error=/vast/projects/kumar/lab/yishao/b1k_2/logs/exp1-expanded-%j.err

# EXP1 EXPANDED: V2 data expanded to per-object binary + V1 encoder
# Runs from WORKTREE (b1k_2_exp1_expanded) with PYTHONPATH override so Python loads
# the worktree's b1k module (TASK_NUM_PREDICATES=248) instead of the main repo's (218).
# Verified: without PYTHONPATH, Python uses main repo's editable install.

PROJECT_DIR="/vast/projects/kumar/lab/yishao/b1k_2"
WORKTREE_DIR="/vast/projects/kumar/lab/yishao/b1k_2/b1k_2_exp1_expanded"
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

# CRITICAL: override editable install with worktree src so TASK_NUM_PREDICATES=248
export PYTHONPATH="${WORKTREE_DIR}/src:${PYTHONPATH}"

hostname
nvidia-smi -L
echo "=== EXP1 EXPANDED: V2 data (per-object binary) + V1 encoder, 6K steps ==="
echo "Start: $(date)"
echo "PYTHONPATH=${PYTHONPATH}"

# Verify worktree config is loaded BEFORE starting (fail loud if not)
# Use uv run since flax is in the uv-managed venv, not plain conda
uv run python -c "
import b1k.models.pi_behavior_config as cfg
expected_total = 248
if cfg.TOTAL_TASK_PREDICATE_EMBEDDINGS != expected_total:
    raise RuntimeError(
        f'CONFIG MISMATCH: loaded {cfg.__file__} with total={cfg.TOTAL_TASK_PREDICATE_EMBEDDINGS}, expected {expected_total}. '
        f'PYTHONPATH override failed — will train with wrong TASK_NUM_PREDICATES.'
    )
print(f'Config OK: {cfg.__file__} total={cfg.TOTAL_TASK_PREDICATE_EMBEDDINGS}')
"
if [ $? -ne 0 ]; then
    echo "FATAL: Config preflight check failed"
    exit 1
fi

uv run scripts/train.py pi_behavior_b1k_exp1_expanded \
  --batch_size 1024 \
  --fsdp_devices 8 \
  --log_interval 100 \
  --exp_name "exp1_expanded_v2data_v1arch_perobject"

echo "=== EXP1 EXPANDED done at $(date) ==="
