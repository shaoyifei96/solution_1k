#!/bin/bash
#SBATCH --job-name=eval_exp2_5k
#SBATCH --account=kumar-lab
#SBATCH --requeue
#SBATCH --time=12:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=250G
#SBATCH --output=/vast/projects/kumar/lab/yishao/jobs/logs/eval_exp2_5k-%A-%a.out
#SBATCH --error=/vast/projects/kumar/lab/yishao/jobs/logs/eval_exp2_5k-%A-%a.err
#SBATCH --array=0-19

export TORCHINDUCTOR_COMPILE_THREADS=1
export UV_LINK_MODE=copy
export UV_NO_CACHE=1
export PYTHONUNBUFFERED=1
export SSL_CERT_DIR=/etc/ssl/certs
export TMPDIR=/tmp

# Checkpoint 1's 20 tasks (indices 0-19 map to ckpt1 tasks)
TASKS=(
    "putting_away_Halloween_decorations"
    "cleaning_up_plates_and_food"
    "setting_mousetraps"
    "hiding_Easter_eggs"
    "set_up_a_coffee_station_in_your_kitchen"
    "putting_dishes_away_after_cleaning"
    "loading_the_car"
    "carrying_in_groceries"
    "bringing_in_wood"
    "outfit_a_basic_toolbox"
    "boxing_books_up_for_storage"
    "storing_food"
    "clearing_food_from_table_into_fridge"
    "getting_organized_for_work"
    "clean_up_your_desk"
    "hanging_pictures"
    "chop_an_onion"
    "chopping_wood"
    "freeze_pies"
    "canning_food"
)

TASK_IDX=$SLURM_ARRAY_TASK_ID
TASK_NAME=${TASKS[$TASK_IDX]}
PORT=$((8222 + TASK_IDX))

HOME_DIR=/vast/projects/kumar/lab/yishao
BEHAVIOR_DIR=$HOME_DIR/BEHAVIOR-1K
B1K_DIR=$HOME_DIR/b1k_2
CHECKPOINT_DIR=$HOME_DIR/checkpoints_50/pi_behavior_b1k_exp2/exp2_v2_progress/5000
EVAL_LOG_DIR=$HOME_DIR/eval_logs
LOG_PATH=${EVAL_LOG_DIR}/${TASK_NAME}_exp2_step5000_${SLURM_ARRAY_JOB_ID}

mkdir -p $EVAL_LOG_DIR
mkdir -p /vast/projects/kumar/lab/yishao/jobs/logs

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"

hostname
echo "=== Eval exp1 step 6000: $TASK_NAME (4 instances) ==="
echo "Start: $(date)"

cleanup() {
    [ -n "$SERVER_PID" ] && kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null
    exit
}
trap cleanup EXIT SIGINT SIGTERM

# Start policy server
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

# Run eval with 4 instances
conda activate "$HOME_DIR/envs/behavior"
cd $BEHAVIOR_DIR

python OmniGibson/omnigibson/learning/eval.py \
    policy=websocket \
    task.name=$TASK_NAME \
    model.port=$PORT \
    eval_instance_ids=[0,1,2,3] \
    log_path=$LOG_PATH

echo "=== Done at $(date) ==="
