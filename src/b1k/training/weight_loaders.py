"""Weight loaders for PI_BEHAVIOR model initialization from Pi05 checkpoints.

Reference: https://github.com/Physical-Intelligence
"""

import dataclasses
import logging
import re

import flax.traverse_util
import numpy as np
import orbax.checkpoint as ocp

import openpi.shared.array_typing as at
import openpi.shared.download as download

# Re-export base loaders from OpenPI
from openpi.training.weight_loaders import (
    WeightLoader,
    NoOpWeightLoader,
    CheckpointWeightLoader,
    _merge_params,
)

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class PiBehaviorWeightLoader(WeightLoader):
    """Loads checkpoints for PI_BEHAVIOR model.

    Handles 3 encoder types:
    - v1: original task-specific embeddings + gated fusion
    - v2_progress: v1 + forall/exists/progress modules
    - v2_deep_sets: replaces v1 entirely with Deep Sets
    """

    params_path: str
    predicate_encoder_type: str = "v1"

    def load(self, params: at.Params) -> at.Params:
        params_path = download.maybe_download(self.params_path)

        with ocp.PyTreeCheckpointer() as ckptr:
            restored = ckptr.restore(params_path)

        if isinstance(restored, dict) and "params" in restored:
            loaded_params = restored["params"]
        else:
            loaded_params = restored

        flat_params = flax.traverse_util.flatten_dict(loaded_params)
        if all(kp[-1] == "value" for kp in flat_params if len(kp) > 0):
            flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
            loaded_params = flax.traverse_util.unflatten_dict(flat_params)

        has_task_embeddings = 'task_embeddings' in loaded_params

        if has_task_embeddings:
            return self._load_from_pi_behavior(loaded_params, params)
        else:
            return self._load_from_pi05(loaded_params, params)

    def _load_from_pi_behavior(self, loaded_params, params):
        """Load from a PI_BEHAVIOR checkpoint (has task_embeddings)."""
        enc = self.predicate_encoder_type

        if enc == "v1":
            logging.info("Loading PI_BEHAVIOR checkpoint (all weights will be loaded)")
            return _merge_params(loaded_params, params, missing_regex="^$")

        elif enc == "v2_progress":
            logging.info("Loading PI_BEHAVIOR checkpoint (v2_progress: init new progress modules)")
            missing_regex = (
                ".*forall_fc.*|"
                ".*exists_fc.*|"
                ".*progress_pred_from_vlm.*"
            )
            return _merge_params(loaded_params, params, missing_regex=missing_regex)

        elif enc == "v2_deep_sets":
            logging.info("Loading PI_BEHAVIOR checkpoint (v2_deep_sets: filter removed V1 + init new)")
            # Filter out removed V1 params
            removed_prefixes = [
                "task_predicate_embeddings", "gate_done", "gate_remaining",
                "gate_task", "fusion_layer1", "fusion_layer2", "predicate_projection",
            ]
            filtered_loaded = {}
            for key, value in loaded_params.items():
                skip = any(prefix in key for prefix in removed_prefixes)
                if skip:
                    logging.info(f"  Skipping removed V1 param: {key}")
                else:
                    filtered_loaded[key] = value

            missing_regex = (
                ".*pred_type_emb.*|.*pred_name_emb.*|.*pred_arg_emb.*|"
                ".*phi_fc.*|.*rho_fc.*|"
                ".*forall_fc.*|.*exists_fc.*|"
                ".*pred_token_proj.*|"
                ".*progress_pred_from_vlm.*"
            )
            return _merge_params(filtered_loaded, params, missing_regex=missing_regex)

        else:
            raise ValueError(f"Unknown predicate_encoder_type: {enc}")

    def _load_from_pi05(self, loaded_params, params):
        """Load from a Pi05 base checkpoint (no task_embeddings)."""
        logging.info("Loading Pi05 checkpoint (all PI_BEHAVIOR parameters will use random init)")
        missing_regex = (
            ".*task_embeddings.*|"
            ".*stage_pred_from_vlm.*|"
            ".*task_stage_embeddings.*|"
            ".*gate_sincos.*|"
            ".*gate_task_stage.*|"
            ".*gate_task.*|"
            ".*gate_done.*|"
            ".*gate_remaining.*|"
            ".*fusion_layer.*|"
            ".*stage_projection.*|"
            ".*task_subtask_fusion.*|"
            ".*predicate_projection.*|"
            ".*task_predicate_embeddings.*|"
            ".*fast_token_embedding.*|"
            ".*fast_token_proj.*|"
            ".*kv_transform.*|"
            ".*pred_type_emb.*|.*pred_name_emb.*|.*pred_arg_emb.*|"
            ".*phi_fc.*|.*rho_fc.*|"
            ".*forall_fc.*|.*exists_fc.*|"
            ".*pred_token_proj.*|"
            ".*progress_pred_from_vlm.*|"
            ".*predicate_pred_from_vlm.*"
        )
        return _merge_params(loaded_params, params, missing_regex=missing_regex)
