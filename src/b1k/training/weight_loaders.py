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
    
    Automatically detects:
    - Pi05 checkpoint: Loads weights, preserves new PI_BEHAVIOR parameters
    - PI_BEHAVIOR checkpoint: Loads all weights directly
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # Load checkpoint
        params_path = download.maybe_download(self.params_path)
        
        # Load directly with PyTreeCheckpointer (handles both old and new checkpoint formats)
        with ocp.PyTreeCheckpointer() as ckptr:
            restored = ckptr.restore(params_path)
        
        # Handle nested 'params' key (from some checkpoint formats)
        if isinstance(restored, dict) and "params" in restored:
            loaded_params = restored["params"]
        else:
            loaded_params = restored
        
        # Remove 'value' suffixes (from nnx.State format)
        flat_params = flax.traverse_util.flatten_dict(loaded_params)
        if all(kp[-1] == "value" for kp in flat_params if len(kp) > 0):
            flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
            loaded_params = flax.traverse_util.unflatten_dict(flat_params)
        
        # Detect checkpoint type
        has_task_embeddings = 'task_embeddings' in loaded_params
        
        if has_task_embeddings:
            # Loading PI_BEHAVIOR checkpoint — V2 Full: filter removed V1 params + init new
            logging.info("Loading PI_BEHAVIOR checkpoint (V2 Full: filter removed + init new)")

            # Filter out removed V1 params from loaded checkpoint
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

            # New V2 params to random-init
            missing_regex = (
                ".*pred_type_emb.*|.*pred_name_emb.*|.*pred_arg_emb.*|"
                ".*phi_fc.*|.*rho_fc.*|"
                ".*forall_fc.*|.*exists_fc.*|"
                ".*pred_token_proj.*|"
                ".*progress_pred_from_vlm.*"
            )
            return _merge_params(filtered_loaded, params, missing_regex=missing_regex)
        else:
            # Loading Pi05 checkpoint
            logging.info("Loading Pi05 checkpoint (all V2 parameters will use random init)")
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
                ".*pred_type_emb.*|.*pred_name_emb.*|.*pred_arg_emb.*|"
                ".*phi_fc.*|.*rho_fc.*|"
                ".*forall_fc.*|.*exists_fc.*|"
                ".*pred_token_proj.*|"
                ".*progress_pred_from_vlm.*|"
                ".*predicate_pred_from_vlm.*"
            )
            return _merge_params(loaded_params, params, missing_regex=missing_regex)
