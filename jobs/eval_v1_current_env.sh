#!/bin/bash
#SBATCH --job-name=eval_v1_env
#SBATCH --account=kumar-lab
#SBATCH --time=02:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=250G
#SBATCH --output=/vast/projects/kumar/lab/yishao/jobs/logs/eval_v1_env-%j.out
#SBATCH --error=/vast/projects/kumar/lab/yishao/jobs/logs/eval_v1_env-%j.err
#SBATCH --exclude=dgx021

export TORCHINDUCTOR_COMPILE_THREADS=1 UV_LINK_MODE=copy UV_NO_CACHE=1 PYTHONUNBUFFERED=1 SSL_CERT_DIR=/etc/ssl/certs TMPDIR=/tmp
HOME_DIR=/vast/projects/kumar/lab/yishao
B1K_DIR=$HOME_DIR/b1k_2
BEHAVIOR_DIR=$HOME_DIR/BEHAVIOR-1K
# V1 config + V1 checkpoint (233 embeddings)
CHECKPOINT_DIR=$HOME_DIR/checkpoints_50/pi_behavior_b1k_fast/b1k_predicate_ckpt_1_rand/6000
PORT=8294
TASK_NAME=set_up_a_coffee_station_in_your_kitchen
LOG_PATH=$HOME_DIR/eval_logs/${TASK_NAME}_v1_current_env_${SLURM_JOB_ID}
mkdir -p $HOME_DIR/eval_logs $HOME_DIR/jobs/logs
module load anaconda3; source "$(conda info --base)/etc/profile.d/conda.sh"
cleanup() { [ -n "$SERVER_PID" ] && kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null; exit; }
trap cleanup EXIT SIGINT SIGTERM

hostname
echo "=== V1 checkpoint in current eval env (PassthroughWrapper + oracle) ==="
echo "Task: $TASK_NAME, 4 instances"
echo "Start: $(date)"

conda activate "$HOME_DIR/envs/b1k_solution"; cd $B1K_DIR
uv run scripts/serve_b1k.py --port $PORT policy:checkpoint --policy.config pi_behavior_b1k_fast --policy.dir $CHECKPOINT_DIR &
SERVER_PID=$!; sleep 45
if ! kill -0 $SERVER_PID 2>/dev/null; then echo "ERROR: Server failed"; exit 1; fi
conda activate "$HOME_DIR/envs/behavior"; cd $BEHAVIOR_DIR
python OmniGibson/omnigibson/learning/eval.py policy=websocket task.name=$TASK_NAME model.port=$PORT eval_instance_ids=[0,1,2,3] log_path=$LOG_PATH
echo "=== Done at $(date) ==="
