#!/bin/bash
#SBATCH --job-name=smoke_online_pred
#SBATCH --account=kumar-lab
#SBATCH --time=01:30:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --output=/vast/projects/kumar/lab/yishao/jobs/logs/smoke_online-%j.out
#SBATCH --error=/vast/projects/kumar/lab/yishao/jobs/logs/smoke_online-%j.err

# Smoke test for online predicate manager.
# Runs exp1 step 6000 on 1 fast task (setting_mousetraps), 1 instance.
# Goal: verify the new code path (eval.py online predicate state ->
# wrapper -> B1kInputs -> Observation) doesn't crash and produces a
# sensible q_score. Compares to the same task/instance from job 5187953.

export TORCHINDUCTOR_COMPILE_THREADS=1
export UV_LINK_MODE=copy
export UV_NO_CACHE=1
export PYTHONUNBUFFERED=1
export SSL_CERT_DIR=/etc/ssl/certs
export TMPDIR=/tmp

HOME_DIR=/vast/projects/kumar/lab/yishao
BEHAVIOR_DIR=$HOME_DIR/BEHAVIOR-1K
B1K_DIR=$HOME_DIR/b1k_2
CHECKPOINT_DIR=$HOME_DIR/checkpoints_50/pi_behavior_b1k_fast/exp1_v2data_v1arch/6000
TASK_NAME=setting_mousetraps
PORT=8290
LOG_PATH=$HOME_DIR/eval_logs/smoke_online_${SLURM_JOB_ID}

mkdir -p $HOME_DIR/jobs/logs

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"

hostname
echo "=== Smoke test: online predicate manager ==="
echo "Task: $TASK_NAME, instance 0, exp1 step 6000"
echo "Start: $(date)"

cleanup() {
    [ -n "$SERVER_PID" ] && kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null
    exit
}
trap cleanup EXIT SIGINT SIGTERM

# Start policy server (no metadata path needed for exp1 V1 encoder)
conda activate "$HOME_DIR/envs/b1k_solution"
cd $B1K_DIR
uv run scripts/serve_b1k.py \
    --port $PORT \
    policy:checkpoint \
    --policy.config pi_behavior_b1k_fast \
    --policy.dir $CHECKPOINT_DIR &
SERVER_PID=$!

sleep 60
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "ERROR: Server failed to start"
    exit 1
fi

# Run eval (1 instance only)
conda activate "$HOME_DIR/envs/behavior"
cd $BEHAVIOR_DIR

python OmniGibson/omnigibson/learning/eval.py \
    policy=websocket \
    task.name=$TASK_NAME \
    model.port=$PORT \
    eval_instance_ids=[0] \
    log_path=$LOG_PATH

echo "=== Done at $(date) ==="
echo "--- Metric file ---"
ls $LOG_PATH/metrics/ 2>/dev/null
cat $LOG_PATH/metrics/*.json 2>/dev/null
