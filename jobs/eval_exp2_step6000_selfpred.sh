#!/bin/bash
#SBATCH --job-name=eval_exp2_sp
#SBATCH --account=kumar-lab
#SBATCH --requeue
#SBATCH --time=02:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=250G
#SBATCH --output=/vast/projects/kumar/lab/yishao/jobs/logs/eval_exp2_sp-%j.out
#SBATCH --error=/vast/projects/kumar/lab/yishao/jobs/logs/eval_exp2_sp-%j.err
#SBATCH --exclude=dgx021

export TORCHINDUCTOR_COMPILE_THREADS=1
export UV_LINK_MODE=copy
export UV_NO_CACHE=1
export PYTHONUNBUFFERED=1
export SSL_CERT_DIR=/etc/ssl/certs
export TMPDIR=/tmp

TASK_NAME=chop_an_onion
PORT=8297

HOME_DIR=/vast/projects/kumar/lab/yishao
BEHAVIOR_DIR=$HOME_DIR/BEHAVIOR-1K
B1K_DIR=$HOME_DIR/b1k_2
CHECKPOINT_DIR=$HOME_DIR/checkpoints_50/pi_behavior_b1k_exp2/exp2_v2_progress/6000
EVAL_LOG_DIR=$HOME_DIR/eval_logs
LOG_PATH=${EVAL_LOG_DIR}/${TASK_NAME}_exp2_step6000_selfpred_${SLURM_JOB_ID}

mkdir -p $EVAL_LOG_DIR /vast/projects/kumar/lab/yishao/jobs/logs

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"

hostname
echo "=== Eval exp2 step 6000 (self-predict, no oracle): $TASK_NAME, 1 instance ==="
echo "Start: $(date)"

cleanup() {
    [ -n "$SERVER_PID" ] && kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null
    exit
}
trap cleanup EXIT SIGINT SIGTERM

conda activate "$HOME_DIR/envs/b1k_solution"
cd $B1K_DIR
uv run scripts/serve_b1k.py \
    --port $PORT \
    policy:checkpoint \
    --policy.config pi_behavior_b1k_exp2 \
    --policy.dir $CHECKPOINT_DIR &
SERVER_PID=$!

sleep 45
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "ERROR: Server failed to start"
    exit 1
fi

conda activate "$HOME_DIR/envs/behavior"
cd $BEHAVIOR_DIR

python OmniGibson/omnigibson/learning/eval.py \
    policy=websocket \
    task.name=$TASK_NAME \
    model.port=$PORT \
    eval_instance_ids=[0] \
    log_path=$LOG_PATH

echo "=== Done at $(date) ==="
