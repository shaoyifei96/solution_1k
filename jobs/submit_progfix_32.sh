#!/bin/bash
# Submit 32 independent jobs: 8 tasks × 4 instances
# Each job runs 1 instance in isolation — no cross-instance contamination

TASKS=(
    "putting_away_Halloween_decorations"
    "cleaning_up_plates_and_food"
    "setting_mousetraps"
    "hiding_Easter_eggs"
    "putting_dishes_away_after_cleaning"
    "bringing_in_wood"
    "clearing_food_from_table_into_fridge"
    "freeze_pies"
)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

for task in "${TASKS[@]}"; do
    for inst in 0 1 2 3; do
        echo "Submitting: $task instance $inst"
        sbatch "$SCRIPT_DIR/eval_exp3_progfix_single.sh" "$task" "$inst"
    done
done

echo "Submitted 32 jobs."
