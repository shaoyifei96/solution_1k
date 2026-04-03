"""
Exp 3: Changes to src/b1k/transforms.py ComputePredicateStateFromData

Replace __call__ method (lines 129-161) with:
"""

NEW_CALL = """
    def __call__(self, data):
        from b1k.shared.predicate_data import get_predicate_store
        import numpy as np
        import logging

        if "episode_index" not in data or "timestamp" not in data or "task_index" not in data:
            data["predicate_states"] = np.zeros(self.max_predicates, dtype=bool)
            data["predicate_mask"] = np.zeros(self.max_predicates, dtype=bool)
            data["predicate_progress"] = np.zeros(self.max_predicates, dtype=np.float32)
            data["predicate_name_ids"] = np.zeros(self.max_predicates, dtype=np.int32)
            data["predicate_arg_ids"] = np.zeros(self.max_predicates, dtype=np.int32)
            data["predicate_type_ids"] = np.zeros(self.max_predicates, dtype=np.int32)
            return data

        task_index = int(data["task_index"])
        episode_index = int(data["episode_index"])
        timestamp = float(data["timestamp"])
        frame_idx = int(timestamp * 30.0)

        try:
            store = get_predicate_store(self.predicate_data_path)

            # State + progress (3-tuple in V2)
            predicate_states, predicate_mask, predicate_progress = store.get_predicate_state(
                task_id=task_index, episode_id=episode_index,
                frame_idx=frame_idx, max_predicates=self.max_predicates
            )

            # Structured metadata for Deep Sets
            name_ids, arg_ids, type_ids = store.get_predicate_metadata(
                task_id=task_index, max_predicates=self.max_predicates
            )
        except Exception as e:
            logging.warning(f"Failed to get predicate state: {e}, using zeros")
            predicate_states = np.zeros(self.max_predicates, dtype=bool)
            predicate_mask = np.zeros(self.max_predicates, dtype=bool)
            predicate_progress = np.zeros(self.max_predicates, dtype=np.float32)
            name_ids = np.zeros(self.max_predicates, dtype=np.int32)
            arg_ids = np.zeros(self.max_predicates, dtype=np.int32)
            type_ids = np.zeros(self.max_predicates, dtype=np.int32)

        data["predicate_states"] = predicate_states
        data["predicate_mask"] = predicate_mask
        data["predicate_progress"] = predicate_progress
        data["predicate_name_ids"] = name_ids
        data["predicate_arg_ids"] = arg_ids
        data["predicate_type_ids"] = type_ids

        return data
"""
