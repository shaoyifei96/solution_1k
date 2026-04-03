# V2 Predicate Encoder Experiments

3 experiments, 3 branches, applied from `Vincent_init_fix`.

## Setup
```bash
# Create branches
git branch exp/v2-data-v1-arch Vincent_init_fix
git branch exp/v2-progress Vincent_init_fix
git branch exp/v2-full Vincent_init_fix

# For each experiment, copy the corresponding files from experiments/exp{1,2,3}/
```

## Experiments
- **exp1/**: V1 architecture + V2 data (config change only)
- **exp2/**: V1 + progress ratio + per-type modules
- **exp3/**: Full V2 (shared MLP + Deep Sets + progress)
