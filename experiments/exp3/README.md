# Exp 3: Full V2 — Compositional Predicate Encoder (CPE)

## Overview

完整的 V2 架构，三层 Compositional Predicate Encoder：
1. **Layer 1: Shared MLP φ** — 同名 predicate 共享权重 (schema-level sharing)
2. **Layer 2: Per-type modules** — forall/exists 各自处理 progress ratio
3. **Layer 3: Deep Sets** — permutation-invariant aggregation ρ(Σφ(xᵢ))

## What Changed

| Component | V1 (original) | Exp 3 (Full V2) |
|-----------|---------------|-----------------|
| Predicate encoding | 233 independent embeddings | **Shared MLP φ** (type + name + arg embeddings) |
| Progress info | Binary only | **Continuous ratio** |
| Type handling | None | **ForallModule + ExistsModule** |
| Aggregation | Mean pool + 3 gates + 4 fused repr | **Deep Sets: sum → MLP ρ → project to 4 tokens** |
| Parameters | ~239K (embedding only) | **~45K** (stronger bias, fewer params) |
| Loss | BCE only | **BCE + MSE** |
| Task tokens | 5 (base + 4 fused) | **5** (base + 4 projected from Deep Sets) |
| Checkpoint | checkpoint_1 | **Same** (V1 predicate params filtered out) |

## Files Changed (7+)

- `src/b1k/models/observation.py` — add progress, name_ids, arg_ids, type_ids
- `src/b1k/shared/predicate_data.py` — add `get_predicate_metadata()`, structured vocab
- `src/b1k/transforms.py` — pass all structured fields through pipeline
- `src/b1k/models/pi_behavior.py` — **core change**: Deep Sets encoder replaces V1 fusion
- `src/b1k/models/pi_behavior_config.py` — Deep Sets config params
- `src/b1k/training/weight_loaders.py` — filter removed V1 params + init new params
- `src/b1k/training/config.py` — paths, exp_name
- `predicate_scripts/build_predicate_vocab.py` — build name/arg vocab from JSONL

## Architecture

```
Raw Input (per predicate):
  [type_id, name_id, arg_id, progress, satisfied]
         ↓
  Layer 1: Shared MLP φ (type_emb + name_emb + arg_emb → 322-dim → 512 → 1024)
         ↓
  Layer 2: Per-type modules
    atomic:  pass through
    forall:  MLP([emb; ratio]) + residual
    exists:  MLP([emb; ratio]) + residual
         ↓
  Layer 3: Deep Sets
    masked sum over predicates → MLP ρ (1024 → 2048)
         ↓
  Project to 4 tokens → [base_task, tok0, tok1, tok2, tok3] → PaliGemma 2B
```

## Weight Loading from checkpoint_1

checkpoint_1 contains V1 params that Exp 3 **removed**. The weight loader:
1. Filters out: `task_predicate_embeddings`, `gate_*`, `fusion_layer*`, `predicate_projection`
2. Loads remaining params (task_embeddings, PaliGemma, action_expert, etc.)
3. Random-inits new params: `pred_*_emb`, `phi_fc*`, `rho_fc*`, `forall_fc*`, `exists_fc*`, `pred_token_proj`

## How to Run

```bash
# Build vocab first
python predicate_scripts/build_predicate_vocab.py \
  --input-dir /pool/yishao/v2_extract/predicates \
  --output-dir /pool/yishao/v2_extract/predicate_vocab

# L40S validation
uv run scripts/train.py pi_behavior_b1k_fast \
  --batch_size=2 --num_train_steps=100 --save_interval=50 --overwrite

# B200 full training
uv run scripts/train.py pi_behavior_b1k_fast \
  --batch_size=2048 --fsdp_devices=8 --num_train_steps=200000 \
  --save_interval=500 --keep_period=2000
```

## Why This Design

1. **Shared MLP φ** — inside(A,B) and inside(C,D) share weights → cross-task generalization
2. **Deep Sets** — ρ(Σφ(xᵢ)) is the **unique** permutation-invariant form (Zaheer et al., 2017)
3. **Per-type modules** — forall 3/5 ≠ forall 0/5 (V1 treated both as False)
4. **V1 is a special case** — remove L2+L3, replace L1 with independent embeddings → V1

## Expected Outcome

- Cross-task pred accuracy >> V1 (shared encoding generalizes)
- Progress predicates much more accurate (MSE on continuous values)
- Action loss should improve from better predicate conditioning
- If not better than Exp 2, the bottleneck is aggregation not encoding
