#!/bin/bash
# Submit 12 jobs: 3 tasks × 4 instances
# Tests whether using progress head output improves nested task performance

TASKS=(
    "bringing_in_wood"
    "putting_away_Halloween_decorations"
    "putting_dishes_away_after_cleaning"
)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

for task in "${TASKS[@]}"; do
    for inst in 0 1 2 3; do
        echo "Submitting: $task instance $inst"
        sbatch "$SCRIPT_DIR/eval_exp3_progress_head_single.sh" "$task" "$inst"
    done
done

echo "Submitted 12 jobs."
