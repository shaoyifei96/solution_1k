"""Predicate data loader for Exp 3 (Full V2) — with progress + structured metadata.

CHANGES from V1:
- get_predicate_state() returns 3-tuple (binary, mask, progress)
- get_predicate_metadata() returns (name_ids, arg_ids, type_ids) per predicate
- Reads predicate_types, predicate_names from .pkl metadata
"""

import json
import logging
import os
import pickle
from typing import Dict, Tuple, Optional
import numpy as np

logger = logging.getLogger("b1k")

# Default vocab (will be overridden by build_predicate_vocab.py output)
DEFAULT_NAME_VOCAB = {
    "inside": 0, "ontop": 1, "nextto": 2, "under": 3, "attached": 4,
    "open": 5, "cooked": 6, "covered": 7, "real": 8, "contains": 9,
    "onfloor": 10, "toggled_on": 11, "saturated": 12, "filled": 13,
    "overlaid": 14, "draped": 15, "folded": 16, "unknown": 17,
}

DEFAULT_TYPE_VOCAB = {"atomic": 0, "forall": 1, "exists": 2, "not": 3, "forpairs": 4, "or": 5}


class PredicateDataStore:
    """In-memory store with structured predicate metadata for Deep Sets."""

    def __init__(self, predicate_data_path: str, vocab_dir: str | None = None):
        self.data_path = predicate_data_path
        self.task_data: Dict[int, dict] = {}
        self.max_num_predicates = 0

        # Load vocab mappings
        self.name_vocab = DEFAULT_NAME_VOCAB
        self.type_vocab = DEFAULT_TYPE_VOCAB
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
        logger.info(f"PredicateDataStore V2 initialized with {len(self.task_data)} tasks, "
                    f"name_vocab={len(self.name_vocab)}, arg_vocab={len(self.arg_vocab)}")

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
                    continue
                self.task_data[task_id] = task_data
                self.max_num_predicates = max(self.max_num_predicates, task_data['num_items'])

        if len(self.task_data) == 0:
            raise RuntimeError(f"No predicate data files found in {self.data_path}")

    def _find_episode(self, demo_vectors, task_id, episode_id):
        candidates = [
            f"{episode_id:08d}",
            f"{task_id:04d}{episode_id % 10000:04d}",
            str(episode_id),
        ]
        for candidate in candidates:
            if candidate in demo_vectors:
                return candidate
        for stored_key in demo_vectors.keys():
            if str(episode_id) in stored_key or stored_key.endswith(f"{episode_id % 10000:04d}"):
                return stored_key
        return None

    def get_predicate_state(
        self, task_id: int, episode_id: int, frame_idx: int, max_predicates: int = 20
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (binary_states, mask, progress) arrays."""
        states = np.zeros(max_predicates, dtype=bool)
        mask = np.zeros(max_predicates, dtype=bool)
        progress = np.zeros(max_predicates, dtype=np.float32)

        if task_id not in self.task_data:
            return states, mask, progress

        task_data = self.task_data[task_id]
        num_items = task_data['num_items']
        ep_str = self._find_episode(task_data['demo_vectors'], task_id, episode_id)
        if ep_str is None:
            return states, mask, progress

        ep_states = task_data['demo_vectors'][ep_str]['states']
        frame_idx = max(0, min(frame_idx, ep_states.shape[0] - 1))
        frame_vals = ep_states[frame_idx]

        states[:num_items] = (frame_vals >= 0.5).astype(bool)
        mask[:num_items] = True
        progress[:num_items] = frame_vals.astype(np.float32)

        return states, mask, progress

    def get_predicate_metadata(
        self, task_id: int, max_predicates: int = 20
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get structured metadata for Deep Sets encoder.

        Returns:
            name_ids: [max_predicates] int — predicate name index (shared across tasks)
            arg_ids: [max_predicates] int — primary entity argument index
            type_ids: [max_predicates] int — 0=atomic, 1=forall, 2=exists
        """
        name_ids = np.zeros(max_predicates, dtype=np.int32)
        arg_ids = np.zeros(max_predicates, dtype=np.int32)
        type_ids = np.zeros(max_predicates, dtype=np.int32)

        if task_id not in self.task_data:
            return name_ids, arg_ids, type_ids

        task_data = self.task_data[task_id]
        num_items = task_data['num_items']

        # Get predicate types from .pkl metadata
        pred_types = task_data.get('predicate_types', ['atomic'] * num_items)
        # Get predicate names from index_to_item
        index_to_item = task_data.get('index_to_item', {})

        for i in range(min(num_items, max_predicates)):
            # Type ID
            ptype = pred_types[i] if i < len(pred_types) else 'atomic'
            type_ids[i] = self.type_vocab.get(ptype, 0)

            # Name ID — extract predicate verb from item name
            item_name = index_to_item.get(i, "unknown")
            # item_name might be like "inside" or "forall_pumpkin.n.02"
            matched = False
            for name, nid in self.name_vocab.items():
                if name in item_name:
                    name_ids[i] = nid
                    matched = True
                    break
            if not matched:
                name_ids[i] = self.name_vocab.get("unknown", len(self.name_vocab) - 1)

            # Arg ID — extract entity category from item name
            # For now, use a hash-based approach if no vocab
            if self.arg_vocab:
                for arg_name, aid in self.arg_vocab.items():
                    if arg_name in item_name:
                        arg_ids[i] = aid
                        break

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


def get_predicate_store(predicate_data_path: str, vocab_dir: str | None = None) -> PredicateDataStore:
    global _predicate_store
    if _predicate_store is None:
        _predicate_store = PredicateDataStore(predicate_data_path, vocab_dir)
    return _predicate_store


def reset_predicate_store():
    global _predicate_store
    _predicate_store = None
