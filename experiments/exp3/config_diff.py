"""
Exp 3: Changes to config files

1. src/b1k/models/pi_behavior_config.py:
   - Add progress_loss_weight: float = 0.05
   - Add num_predicate_names: int = 20
   - Add num_predicate_args: int = 256
   - Add predicate_vocab_dir: str | None = None

2. src/b1k/training/config.py:
   - exp_name="exp3_v2_full"
   - predicate_data_path="/pool/yishao/v2_extract/predicate_data"
   - Add progress_loss_weight=0.05 to PiBehaviorConfig
   - weight_loader: Pi0.5 for L40S, checkpoint_1 for B200

3. src/b1k/transforms.py:
   Same as Exp 2 for predicate_progress, PLUS:
   - Also populate predicate_name_ids, predicate_arg_ids, predicate_type_ids
   - Call store.get_predicate_metadata(task_id) for structured metadata
"""
