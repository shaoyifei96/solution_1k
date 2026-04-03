"""
Exp 2: Changes to src/b1k/training/weight_loaders.py

CHANGE: When loading PI_BEHAVIOR checkpoint, allow new modules to be randomly initialized.

Replace lines 62-66:
    if has_task_embeddings:
        logging.info("Loading PI_BEHAVIOR checkpoint (all weights will be loaded)")
        return _merge_params(loaded_params, params, missing_regex="^$")

With:
    if has_task_embeddings:
        logging.info("Loading PI_BEHAVIOR checkpoint (new V2 modules will use random init)")
        missing_regex = (
            ".*forall_fc.*|"
            ".*exists_fc.*|"
            ".*progress_pred_from_vlm.*"
        )
        return _merge_params(loaded_params, params, missing_regex=missing_regex)
"""
