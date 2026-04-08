#!/bin/bash
#SBATCH --job-name=eval_2sims_test
#SBATCH --account=kumar-lab
#SBATCH --time=02:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --output=/vast/projects/kumar/lab/yishao/jobs/logs/eval_2sims-%j.out
#SBATCH --error=/vast/projects/kumar/lab/yishao/jobs/logs/eval_2sims-%j.err

# Test: run 2 OmniGibson eval processes on 1 GPU, sharing a single policy server.
# Uses 2 fast tasks with max_steps=1500 to finish quickly.

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
PORT=8222

mkdir -p /vast/projects/kumar/lab/yishao/jobs/logs

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"

hostname
nvidia-smi -L
echo "=== 2-sims-on-1-GPU test ==="
echo "Start: $(date)"

# Start nvidia-smi monitor
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,power.draw --format=csv -l 15 > /vast/projects/kumar/lab/yishao/jobs/logs/gpu_2sims_${SLURM_JOB_ID}.csv &
MONITOR_PID=$!

cleanup() {
    echo "Cleaning up..."
    kill $MONITOR_PID 2>/dev/null
    kill $SERVER_PID 2>/dev/null
    kill $EVAL1_PID 2>/dev/null
    kill $EVAL2_PID 2>/dev/null
    exit
}
trap cleanup EXIT SIGINT SIGTERM

# Start single policy server
echo "[$(date +%T)] Starting policy server on port $PORT..."
conda activate "$HOME_DIR/envs/b1k_solution"
cd $B1K_DIR

uv run scripts/serve_b1k.py \
    --port $PORT \
    policy:checkpoint \
    --policy.config pi_behavior_b1k_fast \
    --policy.dir $CHECKPOINT_DIR &

SERVER_PID=$!
sleep 45

if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "ERROR: Server failed to start"
    exit 1
fi
echo "[$(date +%T)] Server ready (PID=$SERVER_PID)"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv
echo ""

# Launch 2 eval processes in parallel, both talking to same policy server
conda activate "$HOME_DIR/envs/behavior"
cd $BEHAVIOR_DIR

echo "[$(date +%T)] Launching eval 1: setting_mousetraps (instance 0)"
python OmniGibson/omnigibson/learning/eval.py \
    policy=websocket \
    task.name=setting_mousetraps \
    model.port=$PORT \
    eval_on_train_instances=true \
    eval_instance_ids=[0] \
    max_steps=1500 \
    log_path=$HOME_DIR/eval_logs/2sims_test_mousetraps > /vast/projects/kumar/lab/yishao/jobs/logs/2sims_eval1_${SLURM_JOB_ID}.log 2>&1 &
EVAL1_PID=$!

sleep 20

echo "[$(date +%T)] Launching eval 2: hanging_pictures (instance 0)"
python OmniGibson/omnigibson/learning/eval.py \
    policy=websocket \
    task.name=hanging_pictures \
    model.port=$PORT \
    eval_on_train_instances=true \
    eval_instance_ids=[0] \
    max_steps=1500 \
    log_path=$HOME_DIR/eval_logs/2sims_test_hanging > /vast/projects/kumar/lab/yishao/jobs/logs/2sims_eval2_${SLURM_JOB_ID}.log 2>&1 &
EVAL2_PID=$!

echo "[$(date +%T)] Both evals launched. PIDs: $EVAL1_PID, $EVAL2_PID"
echo "Waiting for both to finish..."

# Wait for both
wait $EVAL1_PID
EXIT1=$?
echo "[$(date +%T)] Eval 1 exited with code $EXIT1"

wait $EVAL2_PID
EXIT2=$?
echo "[$(date +%T)] Eval 2 exited with code $EXIT2"

echo ""
echo "=== Final GPU state ==="
nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw --format=csv
echo ""
echo "=== Done at $(date) ==="

# Summary
if [ $EXIT1 -eq 0 ] && [ $EXIT2 -eq 0 ]; then
    echo ">>> SUCCESS: Both evals completed!"
    python3 -c "
import csv
with open('/vast/projects/kumar/lab/yishao/jobs/logs/gpu_2sims_${SLURM_JOB_ID}.csv') as f:
    reader = csv.reader(f)
    next(reader)
    mem = []; util = []; pw = []
    for row in reader:
        try:
            mem.append(int(row[1].strip().replace(' MiB','')))
            util.append(int(row[2].strip().replace(' %','')))
            pw.append(float(row[3].strip().replace(' W','')))
        except: pass
    if mem:
        print(f'GPU Memory: max={max(mem)} MiB ({max(mem)/1024:.1f} GB)')
        print(f'GPU Util (active phase): mean={sum(u for u in util if u > 10)/max(1,sum(1 for u in util if u > 10)):.0f}%, max={max(util)}%')
        print(f'Power: max={max(pw):.0f}W, mean={sum(pw)/len(pw):.0f}W')
"
else
    echo ">>> FAILED: eval1=$EXIT1, eval2=$EXIT2"
fi
