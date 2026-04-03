# Exp 1: V1 Architecture + V2 Data

## Overview

V1 模型架构完全不变，只换 V2 extraction 数据。作为 baseline，验证新数据本身的价值。

## What Changed

| Component | V1 (original) | Exp 1 |
|-----------|---------------|-------|
| Model architecture | V1 (233 embeddings, gates, mean pool) | **Same** |
| Predicate data | V1 .pkl (binary only) | **V2 .pkl** (from simulator extraction) |
| Checkpoint | Pi0.5 base | **checkpoint_1** (20-task fine-tuned) |
| Loss | BCE only | **Same** |

## Files Changed (2)

- `src/b1k/training/config.py` — data path, weight loader, exp_name
- `src/b1k/models/pi_behavior_config.py` — TASK_NUM_PREDICATES (only if V2 counts differ)

## How to Run

```bash
# L40S validation (Pi0.5 base, batch_size=2)
uv run scripts/train.py pi_behavior_b1k_fast \
  --batch_size=2 --num_train_steps=100 --save_interval=50 --overwrite

# B200 full training (checkpoint_1, FSDP)
# First: change weight_loader in config.py to checkpoint_1 path
uv run scripts/train.py pi_behavior_b1k_fast \
  --batch_size=2048 --fsdp_devices=8 --num_train_steps=200000 \
  --save_interval=500 --keep_period=2000
```

## Expected Outcome

If V2 data > V1 data, pred accuracy should improve even without architecture changes.
This isolates the data contribution from the architecture contribution.
