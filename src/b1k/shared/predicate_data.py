"""Predicate data loader for BEHAVIOR-1K.

Loads precomputed state vectors. Supports V1 (binary), V2 (continuous progress),
and Deep Sets metadata (predicate types, names, args).
"""

import json
import logging
import os
import pickle
from typing import Dict, Tuple, Optional
import numpy as np

logger = logging.getLogger("b1k")

DEFAULT_TYPE_VOCAB = {"atomic": 0, "forall": 1, "exists": 2, "not": 3, "forpairs": 4, "or": 5}
DEFAULT_NAME_VOCAB = {
    "inside": 0, "ontop": 1, "nextto": 2, "under": 3, "attached": 4,
    "open": 5, "cooked": 6, "covered": 7, "real": 8, "contains": 9,
    "onfloor": 10, "toggled_on": 11, "saturated": 12, "filled": 13,
    "overlaid": 14, "draped": 15, "folded": 16, "unknown": 17,
}


class PredicateDataStore:
    """In-memory store for predicate state vectors.

    Supports both V1 (binary int32) and V2 (continuous float32) predicate formats.
    """

    def __init__(self, predicate_data_path: str, vocab_dir: str = None):
        self.data_path = predicate_data_path
        self.task_data: Dict[int, dict] = {}
        self.max_num_predicates = 0
        self.type_vocab = DEFAULT_TYPE_VOCAB
        self.name_vocab = dict(DEFAULT_NAME_VOCAB)
        self.arg_vocab: Dict[str, int] = {}

        if vocab_dir and os.path.exists(vocab_dir):
            name_path = os.path.join(vocab_dir, "predicate_name_to_id.json")
            arg_path = os.path.join(vocab_dir, "predicate_arg_to_id.json")
            if os.path.exists(name_path):
                with open(name_path) as f:
                    self.name_vocab = json.load(f)
            if os.path.exists(arg_path):
                with open(arg_path) as f:
                    self.arg_vocab = json.load(f)

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

                # Validate that pkl num_items matches the hardcoded TASK_NUM_PREDICATES.
                # If they drift apart, the model embedding-table offsets become wrong:
                # - pkl > config: predicate slots overflow into the next task's
                #   reserved embedding rows, causing cross-task collision (real
                #   training conflict if both tasks are in the training set, or
                #   silent borrowing of unused V1 rows otherwise)
                # - pkl < config: harmless wasted rows, but still indicates the
                #   config is stale relative to the data
                # Either way, fail loud at load time rather than silently train
                # on a misconfigured model.
                from b1k.models.pi_behavior_config import TASK_NUM_PREDICATES
                expected = TASK_NUM_PREDICATES[task_id]
                actual = task_data['num_items']
                if actual != expected:
                    raise ValueError(
                        f"Predicate count mismatch for task {task_id} ({filename}): "
                        f"pkl has num_items={actual} but TASK_NUM_PREDICATES[{task_id}]={expected}. "
                        f"This causes embedding-row collisions across tasks. Fix by either:\n"
                        f"  (a) updating TASK_NUM_PREDICATES[{task_id}] to {actual} in pi_behavior_config.py, or\n"
                        f"  (b) re-extracting this task's pkl with BDDL goals matching {expected} predicates."
                    )

                self.task_data[task_id] = task_data
                self.max_num_predicates = max(self.max_num_predicates, task_data['num_items'])
            else:
                logger.warning(f"Missing predicate data for task {task_id}: {filepath}")

        if len(self.task_data) == 0:
            raise RuntimeError(f"No predicate data files found in {self.data_path}")

    def _find_episode(self, demo_vectors: dict, task_id: int, episode_id: int) -> Optional[str]:
        """Find episode key in demo_vectors by trying multiple formats."""
        candidates = [
            f"{episode_id:08d}",
            f"{task_id:04d}{episode_id % 10000:04d}",
            str(episode_id),
        ]
        for c in candidates:
            if c in demo_vectors:
                return c
        for key in demo_vectors:
            if str(episode_id) in key or key.endswith(f"{episode_id % 10000:04d}"):
                return key
        return None

    def get_predicate_state(
        self,
        task_id: int,
        episode_id: int,
        frame_idx: int,
        max_predicates: int = 20
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get predicate state for a specific frame.

        Returns:
            predicate_states: [max_predicates] bool — binary satisfied/not (thresholded at 0.5)
            predicate_mask: [max_predicates] bool — valid predicates
            predicate_progress: [max_predicates] float32 — continuous 0-1 values
        """
        predicate_states = np.zeros(max_predicates, dtype=bool)
        predicate_mask = np.zeros(max_predicates, dtype=bool)
        predicate_progress = np.zeros(max_predicates, dtype=np.float32)

        if task_id not in self.task_data:
            return predicate_states, predicate_mask, predicate_progress

        task_data = self.task_data[task_id]
        num_items = task_data['num_items']
        ep_str = self._find_episode(task_data['demo_vectors'], task_id, episode_id)
        if ep_str is None:
            logger.debug(f"Episode {episode_id} not found for task {task_id}, returning zeros")
            return predicate_states, predicate_mask, predicate_progress

        states = task_data['demo_vectors'][ep_str]['states']
        frame_idx = max(0, min(frame_idx, states.shape[0] - 1))
        frame_vals = states[frame_idx]

        predicate_states[:num_items] = (frame_vals > 0.5).astype(bool)
        predicate_mask[:num_items] = True
        predicate_progress[:num_items] = frame_vals.astype(np.float32)

        return predicate_states, predicate_mask, predicate_progress

    def get_predicate_metadata(
        self, task_id: int, max_predicates: int = 20
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get structured metadata for Deep Sets encoder.

        Returns:
            name_ids: [max_predicates] int
            arg_ids: [max_predicates] int
            type_ids: [max_predicates] int — 0=atomic, 1=forall, 2=exists
        """
        name_ids = np.zeros(max_predicates, dtype=np.int32)
        arg_ids = np.zeros(max_predicates, dtype=np.int32)
        type_ids = np.zeros(max_predicates, dtype=np.int32)

        if task_id not in self.task_data:
            return name_ids, arg_ids, type_ids

        task_data = self.task_data[task_id]
        num_items = task_data['num_items']
        pred_types = task_data.get('predicate_types', ['atomic'] * num_items)
        index_to_item = task_data.get('index_to_item', {})

        for i in range(min(num_items, max_predicates)):
            # Type ID
            ptype = pred_types[i] if i < len(pred_types) else 'atomic'
            type_ids[i] = self.type_vocab.get(ptype, 0)

            # Name ID — extract predicate verb from item name
            item_name = index_to_item.get(i, "unknown")
            matched = False
            for name, nid in self.name_vocab.items():
                if name in item_name.lower():
                    name_ids[i] = nid
                    matched = True
                    break
            if not matched:
                name_ids[i] = self.name_vocab.get("unknown", 17)

            # Arg ID
            arg_ids[i] = self.arg_vocab.get(item_name, i)

        return name_ids, arg_ids, type_ids

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
