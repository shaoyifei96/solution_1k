"""
Exp 2: Changes to config files

1. src/b1k/models/pi_behavior_config.py — add progress_loss_weight field
2. src/b1k/training/config.py — add data path, exp_name, progress weight

--- pi_behavior_config.py ---
Add after line 108 (predicate_loss_weight):
    progress_loss_weight: float = 0.05

--- config.py ---
Line 337: exp_name="exp2_v2_progress_pertype",
Line 353: predicate_data_path="/pool/yishao/v2_extract/predicate_data",
Line 339 (inside PiBehaviorConfig): add progress_loss_weight=0.05,
Line 374-377: weight_loader same as Exp 1
"""
