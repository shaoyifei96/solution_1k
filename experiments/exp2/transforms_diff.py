"""
Exp 2: Changes to src/b1k/transforms.py ComputePredicateStateFromData

DIFF: Add predicate_progress to data dict.

Replace lines 129-161 of transforms.py with the following __call__ method:
"""

# --- NEW __call__ for ComputePredicateStateFromData ---

def __call__(self, data):
    """DataDict -> DataDict"""
    from b1k.shared.predicate_data import get_predicate_store
    import numpy as np
    import logging

    if "episode_index" not in data or "timestamp" not in data or "task_index" not in data:
        data["predicate_states"] = np.zeros(self.max_predicates, dtype=bool)
        data["predicate_mask"] = np.zeros(self.max_predicates, dtype=bool)
        data["predicate_progress"] = np.zeros(self.max_predicates, dtype=np.float32)
        return data

    task_index = int(data["task_index"])
    episode_index = int(data["episode_index"])
    timestamp = float(data["timestamp"])
    frame_idx = int(timestamp * 30.0)

    try:
        store = get_predicate_store(self.predicate_data_path)
        predicate_states, predicate_mask, predicate_progress = store.get_predicate_state(
            task_id=task_index,
            episode_id=episode_index,
            frame_idx=frame_idx,
            max_predicates=self.max_predicates
        )
    except Exception as e:
        logging.warning(f"Failed to get predicate state: {e}, using zeros")
        predicate_states = np.zeros(self.max_predicates, dtype=bool)
        predicate_mask = np.zeros(self.max_predicates, dtype=bool)
        predicate_progress = np.zeros(self.max_predicates, dtype=np.float32)

    data["predicate_states"] = predicate_states
    data["predicate_mask"] = predicate_mask
    data["predicate_progress"] = predicate_progress

    return data
