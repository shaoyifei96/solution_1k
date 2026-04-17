#!/bin/bash
#SBATCH --job-name=eval_e3_sp
#SBATCH --account=kumar-lab
#SBATCH --requeue
#SBATCH --time=12:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=250G
#SBATCH --output=/vast/projects/kumar/lab/yishao/jobs/logs/eval_e3_sp-%A-%a.out
#SBATCH --error=/vast/projects/kumar/lab/yishao/jobs/logs/eval_e3_sp-%A-%a.err
#SBATCH --array=0-19
#SBATCH --exclude=dgx021

TASKS=(
    "putting_away_Halloween_decorations" "cleaning_up_plates_and_food" "setting_mousetraps"
    "hiding_Easter_eggs" "set_up_a_coffee_station_in_your_kitchen" "putting_dishes_away_after_cleaning"
    "loading_the_car" "carrying_in_groceries" "bringing_in_wood" "outfit_a_basic_toolbox"
    "boxing_books_up_for_storage" "storing_food" "clearing_food_from_table_into_fridge"
    "getting_organized_for_work" "clean_up_your_desk" "hanging_pictures"
    "chop_an_onion" "chopping_wood" "freeze_pies" "canning_food"
)

export TORCHINDUCTOR_COMPILE_THREADS=1 UV_LINK_MODE=copy UV_NO_CACHE=1 PYTHONUNBUFFERED=1 SSL_CERT_DIR=/etc/ssl/certs TMPDIR=/tmp

TASK_IDX=$SLURM_ARRAY_TASK_ID
TASK_NAME=${TASKS[$TASK_IDX]}
PORT=$((8320 + TASK_IDX))

HOME_DIR=/vast/projects/kumar/lab/yishao
B1K_DIR=$HOME_DIR/b1k_2
BEHAVIOR_DIR=$HOME_DIR/BEHAVIOR-1K
CHECKPOINT_DIR=$HOME_DIR/checkpoints_50/pi_behavior_b1k_exp3/exp3_v2_deep_sets/6000
LOG_PATH=$HOME_DIR/eval_logs/${TASK_NAME}_exp3_selfpred_${SLURM_ARRAY_JOB_ID}

mkdir -p $HOME_DIR/eval_logs $HOME_DIR/jobs/logs
module load anaconda3; source "$(conda info --base)/etc/profile.d/conda.sh"

hostname
echo "=== Eval exp3 step 6000 SELF-PREDICT (no oracle): $TASK_NAME ==="

cleanup() { [ -n "$SERVER_PID" ] && kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null; exit; }
trap cleanup EXIT SIGINT SIGTERM

conda activate "$HOME_DIR/envs/b1k_solution"; cd $B1K_DIR
uv run scripts/serve_b1k.py --port $PORT \
    --predicate_metadata_path $HOME_DIR/b1k_2/data/predicate_data_v2_smoothed \
    policy:checkpoint --policy.config pi_behavior_b1k_exp3 --policy.dir $CHECKPOINT_DIR &
SERVER_PID=$!; sleep 45
if ! kill -0 $SERVER_PID 2>/dev/null; then echo "ERROR: Server failed"; exit 1; fi

conda activate "$HOME_DIR/envs/behavior"; cd $BEHAVIOR_DIR

# Use eval_selfpred.py — eval.py WITHOUT online predicate extraction
python OmniGibson/omnigibson/learning/eval_selfpred.py \
    policy=websocket task.name=$TASK_NAME model.port=$PORT \
    eval_instance_ids=[0,1,2,3] log_path=$LOG_PATH

echo "=== Done at $(date) ==="
