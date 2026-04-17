#!/bin/bash
#SBATCH --job-name=eval_ph
#SBATCH --account=kumar-lab
#SBATCH --requeue
#SBATCH --time=4:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=250G
#SBATCH --output=/vast/projects/kumar/lab/yishao/jobs/logs/eval_ph-%j.out
#SBATCH --error=/vast/projects/kumar/lab/yishao/jobs/logs/eval_ph-%j.err
#SBATCH --exclude=dgx021

# Usage: sbatch eval_exp3_progresshead_single.sh <TASK_NAME> <INSTANCE_ID>
# Tests: using progress_pred_from_vlm (MSE head) instead of sigmoid(binary logits) for progress feedback
TASK_NAME=$1
INSTANCE_ID=$2

if [ -z "$TASK_NAME" ] || [ -z "$INSTANCE_ID" ]; then
    echo "Usage: sbatch $0 <TASK_NAME> <INSTANCE_ID>"
    exit 1
fi

export TORCHINDUCTOR_COMPILE_THREADS=1 UV_LINK_MODE=copy UV_NO_CACHE=1 PYTHONUNBUFFERED=1 SSL_CERT_DIR=/etc/ssl/certs TMPDIR=/tmp

PORT=$((8500 + RANDOM % 100))

HOME_DIR=/vast/projects/kumar/lab/yishao
B1K_DIR=$HOME_DIR/b1k_2
BEHAVIOR_DIR=$HOME_DIR/BEHAVIOR-1K
CHECKPOINT_DIR=$HOME_DIR/checkpoints_50/pi_behavior_b1k_exp3/exp3_v2_deep_sets/6000
LOG_PATH=$HOME_DIR/eval_logs/${TASK_NAME}_exp3_progresshead_${SLURM_JOB_ID}

mkdir -p $HOME_DIR/eval_logs $HOME_DIR/jobs/logs
module load anaconda3; source "$(conda info --base)/etc/profile.d/conda.sh"

hostname
echo "=== Eval exp3 PROGRESS-HEAD: $TASK_NAME instance $INSTANCE_ID ==="
echo "Start: $(date)"

cleanup() { [ -n "$SERVER_PID" ] && kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null; exit; }
trap cleanup EXIT SIGINT SIGTERM

conda activate "$HOME_DIR/envs/b1k_solution"; cd $B1K_DIR
uv run scripts/serve_b1k.py --port $PORT \
    --predicate_metadata_path $HOME_DIR/b1k_2/data/predicate_data_v2_smoothed \
    policy:checkpoint --policy.config pi_behavior_b1k_exp3 --policy.dir $CHECKPOINT_DIR &
SERVER_PID=$!; sleep 45
if ! kill -0 $SERVER_PID 2>/dev/null; then echo "ERROR: Server failed"; exit 1; fi

conda activate "$HOME_DIR/envs/behavior"; cd $BEHAVIOR_DIR

python OmniGibson/omnigibson/learning/eval_selfpred.py \
    policy=websocket task.name=$TASK_NAME model.port=$PORT \
    eval_instance_ids=[$INSTANCE_ID] log_path=$LOG_PATH

echo "=== Done at $(date) ==="
