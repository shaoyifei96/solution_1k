#!/bin/bash
#SBATCH --job-name=extract_predicates
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-49
#SBATCH --output=./logs/extract_predicates_%A_%a.log
#SBATCH --error=./logs/extract_predicates_%A_%a.log

# --- Configuration (edit these) ---
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_DIR="${WORKSPACE_DIR:-$(dirname "$PROJECT_DIR")}"
BEHAVIOR_ENV="${BEHAVIOR_ENV:-$WORKSPACE_DIR/envs/behavior}"
BEHAVIOR_DIR="${BEHAVIOR_DIR:-$WORKSPACE_DIR/BEHAVIOR-1K}"
RAW_DATA_DIR="${RAW_DATA_DIR:-$WORKSPACE_DIR/data/behavior_rawdata}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-$WORKSPACE_DIR/data/predicate_data}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-5}"
SIM_STEPS="${SIM_STEPS:-1}"
# ----------------------------------

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$BEHAVIOR_ENV"
cd "$BEHAVIOR_DIR"

export HF_HOME="${HF_HOME:-$WORKSPACE_DIR/cache/huggingface}"
export SSL_CERT_DIR=/etc/ssl/certs
export TMPDIR=/tmp

TASK_ID=$SLURM_ARRAY_TASK_ID
TASK_DIR=$(printf "task-%04d" "$TASK_ID")
INPUT_DIR="$RAW_DATA_DIR/$TASK_DIR"
OUTPUT_DIR="$OUTPUT_BASE_DIR/$TASK_DIR"

echo "Job array task ID: $TASK_ID"
echo "Processing: $TASK_DIR"

if [ ! -d "$INPUT_DIR" ]; then
    echo "Skipping $TASK_DIR (not found: $INPUT_DIR)"
    exit 0
fi

processed=0
skipped=0

for HDF5_FILE in "$INPUT_DIR"/*.hdf5; do
    [ -f "$HDF5_FILE" ] || continue

    EPISODE_NAME=$(basename "$HDF5_FILE" .hdf5)
    OUTPUT_FILE="$OUTPUT_DIR/${EPISODE_NAME}_predicates.jsonl"

    if [ -f "$OUTPUT_FILE" ]; then
        skipped=$((skipped + 1))
        continue
    fi

    echo "Processing: $HDF5_FILE"
    python "$PROJECT_DIR/predicate_scripts/replay_extract_predicates.py" \
        --file "$HDF5_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --sim-steps "$SIM_STEPS" \
        --sample-interval "$SAMPLE_INTERVAL"
    echo "Exit code: $?"
    processed=$((processed + 1))
done

echo "Processed: $processed, Skipped: $skipped"
echo "Done with task $TASK_ID ($TASK_DIR)"
