"""Predicate data loader for BEHAVIOR-1K — V2 with progress support.

CHANGES from V1:
- get_predicate_state() now returns 3-tuple: (states_binary, mask, progress_float)
- Progress values are the raw float32 from .pkl (count/threshold for forall/exists)
"""

import logging
import os
import pickle
from typing import Dict, Tuple, Optional
import numpy as np

logger = logging.getLogger("b1k")


class PredicateDataStore:
    """In-memory store for predicate state vectors with progress support."""

    def __init__(self, predicate_data_path: str):
        self.data_path = predicate_data_path
        self.task_data: Dict[int, dict] = {}
        self.max_num_predicates = 0
        self._load_all_data()
        logger.info(f"PredicateDataStore initialized with {len(self.task_data)} tasks, "
                    f"max_predicates={self.max_num_predicates}")

    def _load_all_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Predicate data path not found: {self.data_path}")

        for task_id in range(50):
            filename = f"task_{task_id:04d}_state_action_vectors.pkl"
            filepath = os.path.join(self.data_path, filename)

            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    task_data = pickle.load(f)

                if 'demo_vectors' not in task_data or 'num_items' not in task_data:
                    logger.warning(f"Invalid data structure in {filename}, skipping")
                    continue

                self.task_data[task_id] = task_data
                self.max_num_predicates = max(self.max_num_predicates, task_data['num_items'])

                num_demos = len(task_data['demo_vectors'])
                num_items = task_data['num_items']
                logger.debug(f"Loaded task {task_id}: {num_demos} demos, {num_items} items")
            else:
                logger.warning(f"Missing predicate data for task {task_id}: {filepath}")

        if len(self.task_data) == 0:
            raise RuntimeError(f"No predicate data files found in {self.data_path}")

    def get_predicate_state(
        self,
        task_id: int,
        episode_id: int,
        frame_idx: int,
        max_predicates: int = 20
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get predicate state for a specific frame.

        Returns:
            predicate_states: [max_predicates] bool — binary satisfied/not
            predicate_mask: [max_predicates] bool — valid predicates
            predicate_progress: [max_predicates] float32 — continuous 0-1 values
        """
        predicate_states = np.zeros(max_predicates, dtype=bool)
        predicate_mask = np.zeros(max_predicates, dtype=bool)
        predicate_progress = np.zeros(max_predicates, dtype=np.float32)

        if task_id not in self.task_data:
            logger.warning(f"Task {task_id} not found in predicate data, returning zeros")
            return predicate_states, predicate_mask, predicate_progress

        task_data = self.task_data[task_id]
        num_items = task_data['num_items']
        demo_vectors = task_data['demo_vectors']

        episode_id_str = None
        candidates = [
            f"{episode_id:08d}",
            f"{task_id:04d}{episode_id % 10000:04d}",
            str(episode_id),
        ]

        for candidate in candidates:
            if candidate in demo_vectors:
                episode_id_str = candidate
                break

        if episode_id_str is None:
            for stored_key in demo_vectors.keys():
                if str(episode_id) in stored_key or stored_key.endswith(f"{episode_id % 10000:04d}"):
                    episode_id_str = stored_key
                    break

        if episode_id_str is None:
            logger.debug(f"Episode {episode_id} not found for task {task_id}, returning zeros")
            return predicate_states, predicate_mask, predicate_progress

        states = demo_vectors[episode_id_str]['states']  # [T, num_items] float32
        frame_idx = max(0, min(frame_idx, states.shape[0] - 1))
        frame_states = states[frame_idx]  # [num_items] float32

        # Binary: threshold at 0.5 (for backwards compat)
        predicate_states[:num_items] = (frame_states >= 0.5).astype(bool)
        predicate_mask[:num_items] = True
        # Progress: raw float32 values
        predicate_progress[:num_items] = frame_states.astype(np.float32)

        return predicate_states, predicate_mask, predicate_progress

    def get_num_predicates(self, task_id: int) -> int:
        if task_id in self.task_data:
            return self.task_data[task_id]['num_items']
        return 0

    def get_item_names(self, task_id: int) -> Dict[int, str]:
        if task_id in self.task_data:
            return self.task_data[task_id].get('index_to_item', {})
        return {}


_predicate_store: Optional[PredicateDataStore] = None


def get_predicate_store(predicate_data_path: str) -> PredicateDataStore:
    global _predicate_store
    if _predicate_store is None:
        logger.info(f"Initializing PredicateDataStore from {predicate_data_path}")
        _predicate_store = PredicateDataStore(predicate_data_path)
    return _predicate_store


def reset_predicate_store():
    global _predicate_store
    _predicate_store = None
