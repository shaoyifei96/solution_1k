# BEHAVIOR Challenge Solution

**1st Place | Public Leaderboard: 26% | Private Leaderboard: 26%**

This repository contains our winning solution for the [2025 BEHAVIOR Challenge](https://behavior.stanford.edu/challenge/), achieving a 26% success rate on both the public and private evaluation sets. Our approach builds upon [Physical Intelligence's Pi0.5](https://www.physicalintelligence.company/blog/pi0) vision-language-action model with several architectural and training innovations.

**📄 [Technical Report](https://arxiv.org/abs/2512.06951)**

<!-- ** 🎥 [Detailed Video Coming Soon](PLACEHOLDER)** -->

---

## 🏆 Solution Summary
We use policy based on Pi0.5 and built on top of [openpi](https://github.com/Physical-Intelligence/openpi) repository.

**Key Architecture Changes:**
- Replaced language model with 50 trainable task embeddings (no text processing)
- Correlated noise for Flow Matching: noise is sampled from N(0, 0.5*I + 0.5*Σ) instead of independent noise, where Σ is the action correlation matrix
- Each layer of the Action Expert attends to linear combination of all VLM layers KV (learnable weights)
- Attention: Images and task embeddings bidirectionally attend to each other, stage (system 2 - see below) and robot state bidirectionally attend to each other and to images and task embeddings, FAST tokens attend to all mentioned above + causally to each other, Flow Matching attends to everything except FAST tokens and attends to each other bidirectionally

**Key Training Changes:**
- Multi-step Flow Matching: 15 action expert predictions (with different time and noise sampled) per each VLM step to reduce training variance
- Delta action space with per-timestamp normalization
- Extra FAST auxiliary loss with 0.05 weight (no knowledge insulation)
- 0.1 weight for subtask prediction loss (system 2 logic, see below)
- Trained on 224x224 RGB images + proprioception states only

**System 2 Implementation:**
- Predicts current task stage (5-15 stages per task solely based on the timestamps) as auxiliary output from VLM (using only images and task embeddings)
- Stage tracking with voting logic for smooth transitions and non-Markovian resolution, passed as input to the model (using mix of sin encoded subtask state and learnable embeddings)

**Predicate-Conditioned VLA:**
- Auxiliary predicate prediction head: predicts per-object goal completion (e.g., "is cup on table?") as binary labels (BCE loss, 0.1 weight)
- 233 task-specific predicate embeddings across 50 tasks (1-20 predicates per task, max 20)
- Gated fusion of done/remaining predicate embeddings with task embeddings (4 fused representations)
- Two predicate extraction approaches: (1) label-based from skill annotations + BDDL goal definitions, (2) simulator-based by replaying demos and evaluating BDDL conditions
- Inference uses consensus voting over a sliding window of predicate predictions for smooth transitions

**Inference Optimizations:**
- Soft inpainting: predict 30 actions, execute 26, keep 4 for next prediction for inpainting. Inpainting is soft (only first 70% of the denoising steps are inpainted) and correlation aware (the rest of the actions are guided towards the linear regression prediction for them from the original 4 actions).
- 1.3x speedup via cubic interpolation (26 predicted actions are executed in 20 steps). Speed up is disabled when gripper state is changing
- General correction rule: open gripper after failed grasp attempts.
- Extra hardcoded correction rule for radio tasks (probably not important but is kept as a legacy).
- We use 4 checkpoints of the same model for different tasks. They were initially trained on 50 tasks simultaneously but later split and finetuned on the separate subsets of tasks.


## 🚀 Quick Start

### Installation

```bash
# Clone with submodules (includes openpi and BEHAVIOR-1K)
git clone --recurse-submodules https://github.com/ilialarchenko/behavior-1k-solution.git
cd behavior-1k-solution

# Run setup script (installs uv, dependencies, and sets up environment)
bash setup_remote.sh
```

### Dataset Preparation

Download the official BEHAVIOR-1K dataset from HuggingFace:

```bash
# Login to HuggingFace (need to avoid request rate limit errors)
uv run huggingface-cli login

# Download the full dataset (~2TB)
uv run python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
  repo_id="behavior-1k/2025-challenge-demos",
  repo_type="dataset",
  local_dir="./data/behavior_dataset",
  local_dir_use_symlinks=False
)
PY
```

**Alternative**: Use the resized RGB-only dataset (224×224, ~260GB) for faster training:
```bash
uv run python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
  repo_id="IliaLarchenko/behavior_224_rgb",
  repo_type="dataset",
  local_dir="./data/behavior_224_rgb",
  local_dir_use_symlinks=False
)
PY
```

### Configuration

Update paths in `src/b1k/training/config.py` (lines 334-378) as needed:

```python
repo_id="behavior-1k/2025-challenge-demos",
behavior_dataset_root="/path/to/data/behavior_dataset",
assets_base_dir="/path/to/outputs/assets",
checkpoint_base_dir="/path/to/outputs/checkpoints",
```

### Pre-training Setup

Compute dataset statistics and train FAST tokenizer:

```bash
# Compute normalization statistics with correlation matrix
uv run scripts/compute_norm_stats.py --config-name pi_behavior_b1k_fast --correlation

# Train FAST tokenizer for action discretization
uv run scripts/train_fast_tokenizer.py \
  --config-name pi_behavior_b1k_fast \
  --encoded-dims="0:6,7:23" \
  --vocab-size=1024
```

**Note**: If using pre-trained checkpoints, you can skip this step and copy assets from the checkpoint directory.

### Training

Login to Weights & Biases for experiment tracking (optional):

```bash
uv run wandb login
```

**Single GPU Training**:
```bash
uv run scripts/train.py pi_behavior_b1k_fast \
  --batch_size=16 \
  --num_train_steps=200000 \
  --save_interval=2000 \
  --keep_period=10000 \
  --log_interval=100
```

**Multi-GPU Training**:
```bash
uv run scripts/train.py pi_behavior_b1k_fast \
  --batch_size=2048 \
  --num_train_steps=200000 \
  --fsdp_devices=8 \
  --save_interval=250 \
  --keep_period=4000 \
  --log_interval=25
```

Adjust batch size and other parameters to your needs.

We used Fully Sharded Data Parallel (FSDP) - for training that allows to use a bigger batch size on multiple GPUs. openpi supports other multi-GPU modes but they were not tested and not used in this project. Please set `fsdp_devices=<number of GPUs>` to avoid any issues.

Use `--overwrite` to start a new training and overwrite existing checkpoints or `--resume` to continue training from the last checkpoint. `--resume` works only with the same number of GPUs as the original training.


### Fine-tuning

You can initialize model weights either from already pretrained checkpoint or from Pi0.5 model (in this case all new parameters will get initial/random weights). Specify it in the training config. Our solution was initialized from Pi0.5 base model. 

```python
weight_loader=weight_loaders.PiBehaviorWeightLoader("/path/to/model/params")
```

### Evaluation

Start the policy server:

```bash
uv run scripts/serve_b1k.py policy:checkpoint \
  --policy.config pi_behavior_b1k_fast \
  --policy.dir /path/to/checkpoint
```

In a separate terminal, [run evaluation](https://behavior.stanford.edu/challenge/baselines.html) (requires BEHAVIOR-1K environment):

```bash
python BEHAVIOR-1K/omnigibson/learning/eval.py \
  log_path=./eval_logs \
  policy=websocket \
  task.name=make_microwave_popcorn \
  model.host=localhost \
  eval_instance_ids="[0,1,2,3]"
```

**Note**: this repo supports only our modification of the model developed for the competition, if you want to use Pi0.5 or Pi0 policies, use original [openpi](https://github.com/Physical-Intelligence/openpi) repository.

---

## 🔍 Predicate Extraction Pipeline

We extract ground-truth predicate states (e.g., "is object at goal location?") from demonstrations to use as an auxiliary training signal. Two extraction approaches exist:

### Version 1: Label-based (from annotations + BDDL definitions)

Derives predicate states from skill annotations (pick/place labels) in the HDF5 metadata combined with BDDL goal definitions, **without** running the simulator. Faster but less accurate.

Key scripts (in the workspace `data/predicate_data/` directory):
- `generate_all_action_pairs_from_annotations.py` — Extracts action pairs from B1K skill annotations
- `generate_state_action_vectors.py` — Builds per-timestep state/action vectors from annotation-derived action pairs
- `generate_visual_predicates_v3.py` — Parses BDDL goal files and converts predicates to visually-grounded forms (counts instead of instance-specific)
- `compare_predicate_methods.py` — Compares label-based vs simulator-based results

### Version 2: Simulator-based (replay + BDDL evaluation)

Replays demos in OmniGibson and calls `evaluate_goal_conditions()` at each timestep for ground-truth predicate states. More accurate but requires GPU + simulator.

Key scripts (in `predicate_scripts/`):
- `replay_extract_predicates.py` — Replays HDF5 demos, evaluates BDDL conditions → JSONL + Parquet
- `generate_predicate_vectors.py` — Consolidates JSONL → `.pkl` state vectors for training
- `visibility_utils.py` — Object visibility filtering for camera-aware predicates

```bash
# Extract predicates via simulator replay (SLURM array job, 1 GPU per task)
sbatch jobs/extract_predicates.sh

# Consolidate into training vectors
sbatch jobs/generate_predicate_vectors.sh
```

### Training Integration

Both versions produce `.pkl` files consumed by the same training pipeline:
- `src/b1k/shared/predicate_data.py` — `PredicateDataStore` loads `.pkl` files
- `src/b1k/transforms.py` — `ComputePredicateStateFromData` transform
- `src/b1k/models/pi_behavior.py` — Predicate fusion + prediction head

Set `predicate_data_path` in the training config to enable predicate conditioning. See `docs/PREDICATE_MIGRATION_SUMMARY.md` for full architecture details.

---

## 📦 Pre-trained Checkpoints

We provide 4 specialized checkpoints that we used for our competition submission. We initially trained them on all 50 tasks but later split and finetuned on the separate subsets of tasks. It was mostly done because of the time and resources constraints, we believe that longer training on 50 tasks can achieve comparable results.

All checkpoints are available on HuggingFace in the same repository: [🤗 Checkpoints](https://huggingface.co/IliaLarchenko/behavior_submission)

You can download them together or one by one.


| Checkpoint | Task Count | Task IDs |
|------------|------------|----------|
| **Checkpoint 1** | 20 tasks | 2, 3, 5, 6, 10, 11, 13, 14, 15, 19, 23, 24, 25, 28, 29, 34, 42, 44, 47, 48 |
| **Checkpoint 2** | 16 tasks | 0, 1, 7, 8, 9, 12, 16, 17, 18, 20, 21, 22, 26, 30, 43, 45 |
| **Checkpoint 3** | 13 tasks | 4, 27, 31, 32, 33, 35, 36, 37, 38, 39, 41, 46, 49 |
| **Checkpoint 4** | 1 task | 40 |

You can run evaluation using all 4 checkpoints at once by configuring it in the `task_checkpoint_mapping.json` file. The policy will use the checkpoint corresponding to the task_id received from the server.

```bash
uv run scripts/serve_b1k.py --task-checkpoint-mapping task_checkpoint_mapping.json \
    policy:checkpoint \
  --policy.config pi_behavior_b1k_fast \
  --policy.dir ~/models/checkpoint_1 #path to any checkpoint
```

We also provide the intermediate checkpoint achieved after the first stage of the model training (simultaneously on 50 tasks). It is not the part of the final submission. 
We didn't properly evaluate it on the full dataset but our guess is that it can achieve around 15-20% q-score. You can find it [here](https://huggingface.co/IliaLarchenko/behavior_50t_checkpoint).

## 👥 Core Team

- **[Ilia Larchenko](https://github.com/IliaLarchenko)**
- **[Zarin Gleb](https://github.com/zaringleb)**
- **[Akash Karnatak](https://github.com/akashkarnatak)**

## 🙏 Acknowledgments

We would like to thank:

- **[Vladimir Ershov](https://github.com/Vladimir-Ershov)**
- **[Justyna Ilczuk](https://github.com/ilonajulczuk)**
- **[Andrey Mulenkov](https://github.com/MulixBF)**


Special thanks to **[Nebius](https://nebius.com/)** for providing cloud GPU compute resources and sponsoring the development of our solution.


Special thanks to **[Physical Intelligence](https://www.physicalintelligence.company/)** for providing open source work on Pi0.5 and openpi which was a great inspiration and fundament of our solution.


## 📚 References

**BEHAVIOR-1K Challenge**
- Website: [BEHAVIOR Challenge 2025](https://behavior.stanford.edu/challenge/)
- Paper: Li et al., "BEHAVIOR-1K: A Human-Centered, Embodied AI Benchmark with 1,000 Everyday Activities and Realistic Simulation" (2024) [[arXiv:2403.09227]](https://arxiv.org/abs/2403.09227)
- Dataset: [HuggingFace Dataset](https://huggingface.co/datasets/behavior-1k/2025-challenge-demos)
- Code: [BEHAVIOR-1K Repository](https://github.com/StanfordVL/BEHAVIOR-1K)

**Pi0.5 (Physical Intelligence)**
- Blog post: [π0.5: a VLA with Open-World Generalization](https://www.physicalintelligence.company/blog/pi05)
- Code: [OpenPI Repository](https://github.com/Physical-Intelligence/openpi)
- Paper: [π0.5](https://www.physicalintelligence.company/download/pi05.pdf)


## Citation

If you find this work useful, please cite:

```bibtex
@misc{larchenko2025behavior,
      title={Task adaptation of Vision-Language-Action model: 1st Place Solution for the 2025 BEHAVIOR Challenge}, 
      author={Ilia Larchenko and Gleb Zarin and Akash Karnatak},
      year={2025},
      eprint={2512.06951},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2512.06951}, 
}
```