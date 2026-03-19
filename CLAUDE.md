# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

1st-place solution for the 2025 BEHAVIOR Challenge (26% success rate). A Vision-Language-Action (VLA) model for robotic manipulation across 50 tasks, built on Pi0.5 from Physical Intelligence. Uses JAX/Flax NNX for the model and PyTorch only for data loading.

## Common Commands

```bash
# Setup
bash setup_remote.sh

# Compute normalization stats (required before first training)
uv run scripts/compute_norm_stats.py --config-name pi_behavior_b1k_fast --correlation

# Train FAST tokenizer
uv run scripts/train_fast_tokenizer.py --config-name pi_behavior_b1k_fast --encoded-dims="0:6,7:23" --vocab-size=1024

# Single-GPU training
uv run scripts/train.py pi_behavior_b1k_fast --batch_size=16 --num_train_steps=200000

# Multi-GPU training (FSDP)
uv run scripts/train.py pi_behavior_b1k_fast --fsdp_devices=8 --batch_size=2048

# Serve policy for evaluation
uv run scripts/serve_b1k.py policy:checkpoint --policy.config pi_behavior_b1k_fast --policy.dir /path/to/checkpoint

# Multi-checkpoint evaluation (routes by task_id)
uv run scripts/serve_b1k.py --task-checkpoint-mapping task_checkpoint_mapping.json \
  policy:checkpoint --policy.config pi_behavior_b1k_fast --policy.dir /path/to/checkpoint

# Run evaluation (separate terminal, requires BEHAVIOR-1K env)
python BEHAVIOR-1K/omnigibson/learning/eval.py log_path=./eval_logs policy=websocket task.name=<task>

# Lint/format
ruff check --fix && ruff format
```

Training flags: `--overwrite` starts fresh, `--resume` continues (same GPU count required), `--fsdp_devices=N` enables FSDP.

## Architecture

### PI_BEHAVIOR Model (`src/b1k/models/pi_behavior.py`)

- **PaliGemma VLM** (2B): Processes 3 RGB cameras (head, left wrist, right wrist) at 224x224
- **50 trainable task embeddings** replace language prompts; `tokenized_prompt` is `[task_id, subtask_state]`
- **KVCacheTransform**: Action expert attends to learned linear combinations of all VLM layers (initialized as identity)
- **Gemma Action Expert** (300M): Decodes 30-step action trajectories via Flow Matching
- **Correlated Flow Matching**: Noise from N(0, 0.5*I + 0.5*Σ) using action correlation matrix
- **Multi-step FM**: 15 action expert predictions per VLM forward pass to reduce variance
- **Auxiliary losses**: FAST discretization (0.05 weight), subtask prediction (0.1), predicate BCE (0.1)

### Data Pipeline

```
BehaviorLeRobotDataset → RepackTransform → B1kInputs → DeltaActions →
ResizeImages → TaskIndexToTaskId → PadStatesAndActions →
TokenizeFASTActions → NormalizeWithPerTimestamp → Model
```

### Inference Optimizations

- **Soft inpainting**: Predict 30 actions, execute 26, keep 4 for next step; correlation-aware
- **Cubic interpolation**: 26 predicted actions executed in 20 steps (1.3x speedup, disabled during gripper changes)
- **Correction rules**: Open gripper after failed grasps
- **4 specialized checkpoints** routed by task_id via `task_checkpoint_mapping.json`

## Key Code Patterns

### Configuration System
All configs are **frozen dataclasses** in `src/b1k/training/config.py`. Named configs live in `_CONFIGS` list at the bottom. CLI overrides via `tyro.cli()`. Use `dataclasses.replace()` to modify frozen configs.

### Adding New Model Parameters
1. Add parameter to `PiBehavior` model class
2. Add its name pattern to `missing_regex` in `PiBehaviorWeightLoader` (`src/b1k/training/weight_loaders.py`)
3. It will be randomly initialized when loading from Pi0.5 checkpoints

### Observation Structure (`src/b1k/models/observation.py`)
- `images`: Dict with keys `base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb`
- `state`: 23-dim proprioception (3 base vel + 4 trunk quat + 7+7 arm joints + 2 grippers)
- `tokenized_prompt`: `[task_id, subtask_state]` int32
- `fast_tokens`: Optional FAST discretized actions

### Predicate System (`src/b1k/models/pi_behavior_config.py`)
50 tasks with variable predicates per task (max 20, 233 total embeddings). `TASK_NUM_PREDICATES` and `TASK_PREDICATE_OFFSETS` tuples define the mapping.

## Repository Structure

- `src/b1k/models/` - PI_BEHAVIOR model, config, observation specs
- `src/b1k/policies/` - Inference policies, checkpoint switching
- `src/b1k/training/` - TrainConfig, data loading, checkpoints, weight loaders
- `src/b1k/shared/` - Normalization, predicate data, correction rules
- `src/b1k/transforms.py` - B1K-specific data transforms
- `scripts/` - Training (`train.py`), serving (`serve_b1k.py`), norm stats, FAST tokenizer
- `predicate_scripts/` - Predicate extraction and processing utilities
- `openpi/` - Git submodule: Pi0.5 base model infrastructure (branch: `behavior`)
- `BEHAVIOR-1K/` - Git submodule: Simulator, dataset, evaluation framework

## Dependencies

Package manager: **uv** (Python >=3.11, <3.12). Pre-commit hooks: ruff (lint/format) + uv-lock. Three editable submodule packages: `openpi`, `bddl` (from BEHAVIOR-1K/bddl3), `omnigibson` (from BEHAVIOR-1K/OmniGibson).

## Predicate Extraction Pipeline

Two approaches for extracting predicate states (e.g., "is cup on table?", "is drawer open?") used as auxiliary training signal.

### Version 1: Label-based (from annotations/BDDL definitions)
Derives predicate states from skill annotations (pick/place labels) + BDDL goal definitions, **without** running the simulator.

Scripts in `/vast/projects/kumar/lab/yishao/data/predicate_data/`:
- `generate_all_action_pairs_from_annotations.py` — Extracts action pairs from B1K skill annotations (pick, place, open, close, etc.) in HDF5 metadata
- `generate_state_action_vectors.py` — Builds per-timestep state/action vectors from annotation-derived action pairs
- `generate_visual_predicates_v3.py` (and v2, `transform_to_visual_predicates.py`) — Parses BDDL goal files to define predicates, converts to visually-grounded forms (counts instead of instance-specific)
- `process_task_samples.py` — Applies visual predicate transformation to sample files
- `compare_bddl_to_visual.py`, `compare_predicate_methods.py`, `compare_predicate_methods_detailed.py`, `generate_comparison_summary.py` — Compare the two approaches
- `bddl_comparison/` — Per-task comparison JSONs (task-0000 to task-0049) + summary CSV

### Version 2: Simulator-based (replay + BDDL evaluation)
Replays demos in OmniGibson and calls `evaluate_goal_conditions()` for ground-truth predicate states.

Scripts in `predicate_scripts/` (this repo):
- `replay_extract_predicates.py` — Core extractor: replays HDF5 in simulator, evaluates BDDL conditions per timestep → JSONL + Parquet
- `generate_predicate_vectors.py` — Consolidates JSONL → .pkl state vectors for training
- `visibility_utils.py` — Object visibility filtering
- `format_predicate_hierarchy.py`, `consolidate_first_last.py`, `delete_failed_predicates.py` — Utilities

SLURM jobs in `/vast/projects/kumar/lab/yishao/jobs/`:
- `extract_predicates*.sh` — Submit simulator replay extraction
- `generate_predicate_vectors.sh` — Post-process into training vectors

### Training integration (uses whichever version's .pkl output)
- `src/b1k/shared/predicate_data.py` — `PredicateDataStore` loads .pkl files
- `src/b1k/transforms.py` — `ComputePredicateStateFromData`
- `src/b1k/models/pi_behavior.py` — Predicate fusion + prediction head
- `src/b1k/models/pi_behavior_config.py` — Predicate dimensions/offsets (50 tasks, max 20 predicates each, 233 total)
- `src/b1k/shared/eval_b1k_wrapper.py` — Inference with consensus voting

## Known Issues

- **Fork bomb on data loading**: `torch.compile` is monkey-patched to no-op in `data_loader.py` — do not remove this
- **JAX OOM**: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` is set in `train.py`
- **FSDP resume**: Only works with the same number of GPUs as the original training run
- **Paths**: Dataset and checkpoint paths are hardcoded in `src/b1k/training/config.py` (lines ~334-378) — update for your environment
