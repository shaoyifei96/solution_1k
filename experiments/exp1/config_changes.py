"""
Exp 1: V1 Architecture + V2 Data
Changes to src/b1k/training/config.py (lines 334-386)

Only 3 lines change:
1. predicate_data_path → V2 .pkl path
2. weight_loader → checkpoint_1 (B200) or pi05_base (L40S validation)
3. exp_name → "exp1_v2data_v1arch"
"""

# Replace the _CONFIGS list entry with:
# (diff shown below)

# --- DIFF for src/b1k/training/config.py ---
#
# Line 337: exp_name="b1k_predicate_ckpt_50t",
# +         exp_name="exp1_v2data_v1arch",
#
# Line 353: predicate_data_path="/vast/projects/kumar/lab/yishao/data/predicate_data/predicate_data",
# +         predicate_data_path="/pool/yishao/v2_extract/predicate_data",
#           # On B200: "/vast/projects/kumar/lab/yishao/data/predicate_data_v2"
#
# Line 374-377: weight_loader=weight_loaders.PiBehaviorWeightLoader(
#                   "gs://openpi-assets/checkpoints/pi05_base/params"  # L40S validation
#               ),
#           # On B200: "/vast/projects/kumar/lab/yishao/checkpoints/checkpoint_1/params"
