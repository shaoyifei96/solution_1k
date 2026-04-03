"""
Exp 3: Changes to src/b1k/training/weight_loaders.py

Most complex weight loading: checkpoint_1 has V1 params that Exp 3 REMOVED,
and Exp 3 has NEW params not in checkpoint_1.

Replace lines 59-66 of weight_loaders.py with:
"""

REPLACEMENT = """
        # Detect checkpoint type
        has_task_embeddings = 'task_embeddings' in loaded_params

        if has_task_embeddings:
            # Loading PI_BEHAVIOR checkpoint — V2 Full needs special handling
            logging.info("Loading PI_BEHAVIOR checkpoint (V2 Full: filter removed + init new)")

            # Params REMOVED in V2 Full (exist in checkpoint but not in model)
            removed_param_prefixes = [
                "task_predicate_embeddings",
                "gate_done", "gate_remaining", "gate_task",
                "fusion_layer1", "fusion_layer2",
                "predicate_projection",
            ]

            # Filter out removed params from loaded checkpoint
            filtered_loaded = {}
            for key, value in loaded_params.items():
                skip = False
                for prefix in removed_param_prefixes:
                    if prefix in key:
                        logging.info(f"  Skipping removed V1 param: {key}")
                        skip = True
                        break
                if not skip:
                    filtered_loaded[key] = value

            # Params NEW in V2 Full (exist in model but not in checkpoint)
            missing_regex = (
                ".*pred_type_emb.*|"
                ".*pred_name_emb.*|"
                ".*pred_arg_emb.*|"
                ".*phi_fc.*|"
                ".*rho_fc.*|"
                ".*forall_fc.*|"
                ".*exists_fc.*|"
                ".*pred_token_proj.*|"
                ".*progress_pred_from_vlm.*"
            )

            return _merge_params(filtered_loaded, params, missing_regex=missing_regex)
        else:
            # Loading Pi05 checkpoint — standard handling
            logging.info("Loading Pi05 checkpoint (new PI_BEHAVIOR parameters will use random init)")
            missing_regex = (
                ".*task_embeddings.*|"
                ".*stage_pred_from_vlm.*|"
                ".*task_stage_embeddings.*|"
                ".*gate_sincos.*|"
                ".*gate_task_stage.*|"
                ".*gate_task.*|"
                ".*fusion_layer.*|"
                ".*stage_projection.*|"
                ".*task_subtask_fusion.*|"
                ".*fast_token_embedding.*|"
                ".*fast_token_proj.*|"
                ".*kv_transform.*|"
                ".*pred_type_emb.*|"
                ".*pred_name_emb.*|"
                ".*pred_arg_emb.*|"
                ".*phi_fc.*|"
                ".*rho_fc.*|"
                ".*forall_fc.*|"
                ".*exists_fc.*|"
                ".*pred_token_proj.*|"
                ".*progress_pred_from_vlm.*|"
                ".*predicate_pred_from_vlm.*"
            )
            return _merge_params(loaded_params, params, missing_regex=missing_regex)
"""
