# Predicate-Based Conditioning Migration Summary

## Overview

This document summarizes the migration from **stage-based conditioning** (single-label categorical classification of 1-of-15 stages) to **predicate-based conditioning** (multi-label binary classification where each predicate indicates if an object is at its final position).

## Key Concepts

### Stage-Based (Old)
- Single categorical label per frame: "what stage (1-15) is the task at?"
- Cross-entropy loss
- One stage can be true at a time

### Predicate-Based (New)
- Multiple binary labels per frame: "which objects are done moving?"
- BCE (Binary Cross-Entropy) loss with masking
- Multiple predicates can be true simultaneously (multi-label)
- Per-task predicate counts (1-20 predicates depending on task complexity)

## Data Format

Predicate data is stored in `data/predicate_data/task_XXXX_state_action_vectors.pkl` files:

```python
{
    'demo_vectors': {
        'episode_id': {
            'states': np.ndarray,  # [T, num_items] - True if object is done
            'actions': np.ndarray  # [T, num_items] - action vectors
        }
    },
    'num_items': int,           # Number of predicates for this task
    'item_to_index': dict,      # Object name → index mapping
    'index_to_item': dict       # Index → object name mapping
}
```

## Constants

```python
# Per-task predicate counts (50 tasks)
TASK_NUM_PREDICATES = (
    1, 4, 6, 4, 6, 4, 4, 6, 3, 7,   # Tasks 0-9
    5, 8, 6, 3, 3, 3, 2, 2, 3, 5,   # Tasks 10-19
    15, 7, 4, 7, 8, 4, 20, 6, 6, 10,  # Tasks 20-29
    4, 2, 2, 4, 1, 1, 1, 1, 1, 1,   # Tasks 30-39
    1, 5, 3, 6, 5, 2, 2, 4, 7, 8,   # Tasks 40-49
)

MAX_NUM_PREDICATES = 20  # Maximum predicates per task (task 26 has 20)
TOTAL_TASK_PREDICATE_EMBEDDINGS = 233  # Sum of all task predicates
```

## Files Modified

### 1. `src/b1k/models/pi_behavior_config.py`

**Changes:**
- Added predicate constants: `TASK_NUM_PREDICATES`, `MAX_NUM_PREDICATES`, `TOTAL_TASK_PREDICATE_EMBEDDINGS`, `TASK_PREDICATE_OFFSETS`
- Added `predicate_loss_weight: float = 0.1` for BCE loss weighting
- Added `predicate_data_path: str = "data/predicate_data"` for data location
- Removed all stage-related constants and `subtask_loss_weight`

### 2. `src/b1k/models/observation.py`

**Changes:**
- Added `predicate_states: at.Bool[ArrayT, "*b p"] | None = None` field
- Added `predicate_mask: at.Bool[ArrayT, "*b p"] | None = None` field
- Updated `from_dict()` to parse these fields from batches

### 3. `src/b1k/shared/predicate_data.py` (New File)

**Purpose:** Load and serve predicate state vectors from pkl files.

**Key Components:**
- `PredicateDataStore` class - loads all task pkl files into memory at initialization
- `get_predicate_state(task_id, episode_id, frame_idx)` - returns `(predicate_states, predicate_mask)`
- `get_predicate_store()` - singleton accessor for global instance

### 4. `src/b1k/transforms.py`

**Changes:**
- Added `ComputePredicateStateFromData` transform class
- Uses `PredicateDataStore` to lookup predicate states from pkl files
- Converts timestamp (seconds) to frame index (30 FPS)

### 5. `src/b1k/models/pi_behavior.py`

**Removed:**
- `encode_subtask_state()` method
- `fuse_task_and_subtask()` method
- Stage prediction in `compute_detailed_loss()` and `sample_actions()`
- Subtask loss computation

**Added/Modified:**
- `self.predicate_pred_from_vlm` - Linear layer for BCE prediction (outputs `MAX_NUM_PREDICATES` logits)
- `self.task_predicate_embeddings` - Embed layer (233 total entries, one per predicate per task)
- `self.predicate_encoding_dim` - Half of task embedding dim (1024)
- `self.gate_predicate` - Gate network for predicate signal
- `self.predicate_projection` - Projection for remaining-focus representation

**New Methods:**
- `encode_predicate_progress()` - Sincos encoding of `num_done / num_total`
- `aggregate_predicate_embeddings()` - Mean pooling for done/remaining predicates
- `fuse_task_and_predicates()` - Returns 4 fused representations:
  1. **Task-gated**: Task embedding modulated by predicates
  2. **Balanced fusion**: Task + predicates combined through MLP
  3. **Remaining-focus**: What to manipulate next (remaining predicates)
  4. **Done-focus**: What to avoid (done predicates)

**Updated Methods:**
- `embed_prefix()` - Uses `fuse_task_and_predicates()` when `obs.predicate_states` is available
- `compute_detailed_loss()` - Computes BCE loss for predicates with masking, tracks accuracy metrics
- `sample_actions()` - Returns `(actions, predicate_logits)` instead of `(actions, subtask_logits, predicate_logits)`

### 6. `src/b1k/training/data_loader.py`

**Changes:**
- Updated `transform_dataset()` to accept `model_config` parameter
- Added `ComputePredicateStateFromData` transform when `predicate_data_path` is configured
- Updated both `create_behavior_data_loader()` and `create_behavior_data_loader_grain()`

## Model Architecture

### Token Structure (Prefix)
```
[Image Tokens] [Base Task Token] [Predicate Token 1] [Predicate Token 2] [Predicate Token 3] [Predicate Token 4] [State Tokens] [FAST Tokens (optional)]
```

The 4 predicate tokens represent:
1. Task-gated representation
2. Balanced fusion
3. Remaining-focus (what to manipulate next)
4. Done-focus (what to avoid)

### Predicate Embedding Aggregation

```python
# For each predicate, get its learned embedding
predicate_embeddings = self.task_predicate_embeddings(task_predicate_indices)  # [B, P, D]

# Separate into done and remaining
done_mask = predicate_states & predicate_mask  # Objects that are done
remaining_mask = (~predicate_states) & predicate_mask  # Objects still to manipulate

# Mean pool each group
done_agg = mean(predicate_embeddings * done_mask)  # [B, D]
remaining_agg = mean(predicate_embeddings * remaining_mask)  # [B, D]
```

### Loss Computation

```python
# BCE Loss with masking
pos_loss = -log_sigmoid(predicate_logits)  # Loss when y=1
neg_loss = -log_sigmoid(-predicate_logits)  # Loss when y=0
bce_loss = gt_predicates * pos_loss + (1 - gt_predicates) * neg_loss

# Apply mask for valid predicates
masked_bce = bce_loss * predicate_mask * valid_pred_mask
predicate_loss = mean(masked_bce) / num_valid_predicates

# Total loss
total_loss = action_loss + fast_loss + predicate_loss_weight * predicate_loss
```

### Metrics Tracked

- `predicate_loss` - Mean BCE loss over valid predicates
- `predicate_accuracy` - Fraction of predicates correctly classified (threshold 0.5)
- `predicate_recall_done` - Recall for "done" class (how many done predicates were correctly identified)

## Usage

### Training

The predicate transform is automatically added when the model has a `predicate_data_path` configured:

```python
config = PiBehaviorConfig(
    predicate_data_path="data/predicate_data",
    predicate_loss_weight=0.1,
)
```

### Inference

```python
actions, predicate_logits = model.sample_actions(rng, observation)
# predicate_logits: [B, MAX_NUM_PREDICATES] - apply sigmoid to get probabilities
predicted_done = jax.nn.sigmoid(predicate_logits) > 0.5
```

## Weight Compatibility

The predicate-based model can reuse most weights from a stage-based checkpoint:
- ✅ VLM backbone (PaliGemma)
- ✅ Action expert layers
- ✅ Task embeddings
- ✅ Gate networks (dimensions match)
- ✅ Fusion layers
- ❌ `task_predicate_embeddings` (new, 233 vs 596 stage embeddings)
- ❌ `predicate_pred_from_vlm` (new prediction head)

For finetuning from a stage checkpoint, these new layers will be randomly initialized while others are loaded from the checkpoint.
