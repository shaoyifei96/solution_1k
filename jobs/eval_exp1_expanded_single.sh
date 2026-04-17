#!/bin/bash
#SBATCH --job-name=eval_e1x
#SBATCH --account=kumar-lab
#SBATCH --requeue
#SBATCH --time=4:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=250G
#SBATCH --output=/vast/projects/kumar/lab/yishao/jobs/logs/eval_e1x-%j.out
#SBATCH --error=/vast/projects/kumar/lab/yishao/jobs/logs/eval_e1x-%j.err
#SBATCH --exclude=dgx021

TASK_NAME=$1
INSTANCE_ID=$2

if [ -z "$TASK_NAME" ] || [ -z "$INSTANCE_ID" ]; then
    echo "Usage: sbatch $0 <TASK_NAME> <INSTANCE_ID>"
    exit 1
fi

export TORCHINDUCTOR_COMPILE_THREADS=1 UV_LINK_MODE=copy UV_NO_CACHE=1 PYTHONUNBUFFERED=1 SSL_CERT_DIR=/etc/ssl/certs TMPDIR=/tmp

# CRITICAL: use worktree src for TASK_NUM_PREDICATES=248
export PYTHONPATH="/vast/projects/kumar/lab/yishao/b1k_2/b1k_2_exp1_expanded/src:${PYTHONPATH}"

PORT=$((8800 + RANDOM % 100))

HOME_DIR=/vast/projects/kumar/lab/yishao
B1K_DIR=$HOME_DIR/b1k_2
BEHAVIOR_DIR=$HOME_DIR/BEHAVIOR-1K
CHECKPOINT_DIR=$HOME_DIR/checkpoints_50/pi_behavior_b1k_exp1_expanded/exp1_expanded_v2data_v1arch_perobject/5999
LOG_PATH=$HOME_DIR/eval_logs/${TASK_NAME}_exp1_expanded_${SLURM_JOB_ID}

mkdir -p $HOME_DIR/eval_logs $HOME_DIR/jobs/logs
module load anaconda3; source "$(conda info --base)/etc/profile.d/conda.sh"

hostname
echo "=== Eval exp1-expanded: $TASK_NAME instance $INSTANCE_ID ==="

cleanup() { [ -n "$SERVER_PID" ] && kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null; exit; }
trap cleanup EXIT SIGINT SIGTERM

conda activate "$HOME_DIR/envs/b1k_solution"; cd $B1K_DIR
uv run scripts/serve_b1k.py --port $PORT \
    --predicate_metadata_path $HOME_DIR/b1k_2/data/predicate_data_v2_expanded \
    policy:checkpoint --policy.config pi_behavior_b1k_exp1_expanded --policy.dir $CHECKPOINT_DIR &
SERVER_PID=$!; sleep 45
if ! kill -0 $SERVER_PID 2>/dev/null; then echo "ERROR: Server failed"; exit 1; fi

conda activate "$HOME_DIR/envs/behavior"; cd $BEHAVIOR_DIR

python OmniGibson/omnigibson/learning/eval_selfpred.py \
    policy=websocket task.name=$TASK_NAME model.port=$PORT \
    eval_instance_ids=[$INSTANCE_ID] log_path=$LOG_PATH

echo "=== Done at $(date) ==="
