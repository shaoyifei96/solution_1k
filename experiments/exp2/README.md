# Exp 2: V1 + Progress Ratio + Per-Type Modules

## Overview

在 V1 架构基础上加两个改进：
1. **Progress ratio**: forall/exists predicates 使用连续值 (count/threshold) 替代 binary
2. **Per-type modules**: forall 和 exists 各有独立的 MLP 处理 progress 信息

## What Changed

| Component | V1 (original) | Exp 2 |
|-----------|---------------|-------|
| Predicate encoding | 233 independent embeddings | **Same** (+ per-type module on top) |
| Progress info | Binary only (0/1) | **Continuous ratio** (0.0 - 1.0) |
| Type handling | None | **ForallModule + ExistsModule** (MLP + residual) |
| Aggregation | Mean pool + gates | **Same** |
| Loss | BCE only | **BCE + MSE** (progress regression) |
| Checkpoint | checkpoint_1 | **Same** |

## Files Changed (6)

- `src/b1k/models/observation.py` — add `predicate_progress` field
- `src/b1k/shared/predicate_data.py` — return 3-tuple (binary, mask, progress)
- `src/b1k/transforms.py` — pass `predicate_progress` through pipeline
- `src/b1k/models/pi_behavior.py` — ForallModule, ExistsModule, progress MSE loss
- `src/b1k/training/weight_loaders.py` — missing_regex for new modules
- `src/b1k/training/config.py` + `pi_behavior_config.py` — progress_loss_weight, paths

## Key Architecture Change

```
V1:  embedding[233] → mean pool → gates → 4 fused repr
                                                        
Exp2: embedding[233] → per-type module → mean pool → gates → 4 fused repr
                          ↑
                    ForallModule: MLP([emb; ratio]) + residual
                    ExistsModule: MLP([emb; ratio]) + residual
                    Atomic: pass through (no change)
```

## How to Run

```bash
# L40S validation
uv run scripts/train.py pi_behavior_b1k_fast \
  --batch_size=2 --num_train_steps=100 --save_interval=50 --overwrite

# B200 full training
uv run scripts/train.py pi_behavior_b1k_fast \
  --batch_size=2048 --fsdp_devices=8 --num_train_steps=200000 \
  --save_interval=500 --keep_period=2000
```

## Expected Outcome

- forall/exists pred accuracy should improve (progress info preserved)
- atomic pred accuracy should stay the same (no change for binary predicates)
- action loss should improve if progress info helps policy planning
