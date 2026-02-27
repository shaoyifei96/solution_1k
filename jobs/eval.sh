#!/bin/bash
#SBATCH --job-name=eval_b1k
#SBATCH --requeue
#SBATCH --time=30:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=250G
#SBATCH --output=./logs/%x-%A-%a.out
#SBATCH --error=./logs/%x-%A-%a.err
#SBATCH --array=0-49

# --- Configuration (edit these) ---
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_DIR="${WORKSPACE_DIR:-$(dirname "$PROJECT_DIR")}"
POLICY_ENV="${POLICY_ENV:-$WORKSPACE_DIR/envs/b1k_solution}"
BEHAVIOR_ENV="${BEHAVIOR_ENV:-$WORKSPACE_DIR/envs/behavior}"
BEHAVIOR_DIR="${BEHAVIOR_DIR:-$WORKSPACE_DIR/BEHAVIOR-1K}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:?Set CHECKPOINT_DIR to the model checkpoint path}"
CONFIG="${CONFIG:-pi_behavior_b1k_fast}"
EVAL_LOG_DIR="${EVAL_LOG_DIR:-$WORKSPACE_DIR/eval_logs}"
BASE_PORT="${BASE_PORT:-8222}"
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:-}"
# ----------------------------------

# All 50 BEHAVIOR tasks (index must match task IDs in config)
TASKS=(
    "turning_on_radio"
    "picking_up_trash"
    "putting_away_Halloween_decorations"
    "cleaning_up_plates_and_food"
    "can_meat"
    "setting_mousetraps"
    "hiding_Easter_eggs"
    "picking_up_toys"
    "rearranging_kitchen_furniture"
    "putting_up_Christmas_decorations_inside"
    "set_up_a_coffee_station_in_your_kitchen"
    "putting_dishes_away_after_cleaning"
    "preparing_lunch_box"
    "loading_the_car"
    "carrying_in_groceries"
    "bringing_in_wood"
    "moving_boxes_to_storage"
    "bringing_water"
    "tidying_bedroom"
    "outfit_a_basic_toolbox"
    "sorting_vegetables"
    "collecting_childrens_toys"
    "putting_shoes_on_rack"
    "boxing_books_up_for_storage"
    "storing_food"
    "clearing_food_from_table_into_fridge"
    "assembling_gift_baskets"
    "sorting_household_items"
    "getting_organized_for_work"
    "clean_up_your_desk"
    "setting_the_fire"
    "clean_boxing_gloves"
    "wash_a_baseball_cap"
    "wash_dog_toys"
    "hanging_pictures"
    "attach_a_camera_to_a_tripod"
    "clean_a_patio"
    "clean_a_trumpet"
    "spraying_for_bugs"
    "spraying_fruit_trees"
    "make_microwave_popcorn"
    "cook_cabbage"
    "chop_an_onion"
    "slicing_vegetables"
    "chopping_wood"
    "cook_hot_dogs"
    "cook_bacon"
    "freeze_pies"
    "canning_food"
    "make_pizza"
)

# Get task for this array job
TASK_IDX=$SLURM_ARRAY_TASK_ID
TASK_NAME=${TASKS[$TASK_IDX]}
PORT=$((BASE_PORT + TASK_IDX))
LOG_PATH="${EVAL_LOG_DIR}/${TASK_NAME}_${SLURM_ARRAY_JOB_ID}"

mkdir -p "$EVAL_LOG_DIR"
mkdir -p "$(dirname "$0")/logs"

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID, Array Task: $TASK_IDX"
echo "Task: $TASK_NAME"
echo "Port: $PORT"
echo "Checkpoint: $CHECKPOINT_DIR"
echo "Hostname: $(hostname)"
echo "=========================================="

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"

export SSL_CERT_DIR=/etc/ssl/certs
export TMPDIR=/tmp
export TORCHINDUCTOR_COMPILE_THREADS=1
export UV_LINK_MODE=copy
export UV_NO_CACHE=1

# Cleanup background processes on exit
cleanup() {
    echo "Cleaning up..."
    if [ -n "$SERVER_PID" ]; then
        kill "$SERVER_PID" 2>/dev/null
        wait "$SERVER_PID" 2>/dev/null
    fi
    exit
}
trap cleanup EXIT SIGINT SIGTERM

# Start the policy server (b1k_solution env)
echo "Starting policy server on port $PORT..."
conda activate "$POLICY_ENV"
cd "$PROJECT_DIR"

uv run scripts/serve_b1k.py \
    --port "$PORT" \
    $EXTRA_SERVE_ARGS \
    policy:checkpoint \
    --policy.config "$CONFIG" \
    --policy.dir "$CHECKPOINT_DIR" &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

echo "Waiting for server to initialize..."
sleep 10

if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "ERROR: Server failed to start"
    exit 1
fi

# Run the evaluation client (behavior env)
echo "Starting evaluation for task: $TASK_NAME"
conda activate "$BEHAVIOR_ENV"
cd "$BEHAVIOR_DIR"

python OmniGibson/omnigibson/learning/eval.py \
    policy=websocket \
    task.name="$TASK_NAME" \
    model.port="$PORT" \
    log_path="$LOG_PATH"

EVAL_EXIT_CODE=$?

echo "Evaluation completed with exit code: $EVAL_EXIT_CODE"
echo "Logs saved to: $LOG_PATH"

exit $EVAL_EXIT_CODE
