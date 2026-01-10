"""Predicate data loader for BEHAVIOR-1K.

Loads precomputed state vectors indicating which objects are at their final positions.
All data is loaded into memory at initialization for fast access during training.
"""

import logging
import os
import pickle
from typing import Dict, Tuple, Optional
import numpy as np

logger = logging.getLogger("b1k")


class PredicateDataStore:
    """In-memory store for predicate state vectors.
    
    Loads all predicate data at initialization for fast access during training.
    
    Structure per task:
        - demo_vectors: Dict[episode_id_str, {'states': [T, num_items], 'actions': [T, num_items]}]
        - item_to_index: Dict mapping item names to indices
        - index_to_item: Dict mapping indices to item names
        - num_items: Number of items (predicates) for this task
    
    Usage:
        store = PredicateDataStore("/path/to/predicate_data")
        predicate_states, predicate_mask = store.get_predicate_state(
            task_id=5, episode_id=10, frame_idx=1000
        )
    """
    
    def __init__(self, predicate_data_path: str):
        """Load all predicate data into memory.
        
        Args:
            predicate_data_path: Path to directory containing task_XXXX_state_action_vectors.pkl files
        """
        self.data_path = predicate_data_path
        self.task_data: Dict[int, dict] = {}
        self.max_num_predicates = 0
        
        # Load all task data
        self._load_all_data()
        
        logger.info(f"PredicateDataStore initialized with {len(self.task_data)} tasks, "
                   f"max_predicates={self.max_num_predicates}")
    
    def _load_all_data(self):
        """Load all state_action_vectors.pkl files into memory."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Predicate data path not found: {self.data_path}")
        
        # Find all task files
        for task_id in range(50):
            filename = f"task_{task_id:04d}_state_action_vectors.pkl"
            filepath = os.path.join(self.data_path, filename)
            
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    task_data = pickle.load(f)
                
                # Validate structure
                if 'demo_vectors' not in task_data or 'num_items' not in task_data:
                    logger.warning(f"Invalid data structure in {filename}, skipping")
                    continue
                
                self.task_data[task_id] = task_data
                self.max_num_predicates = max(self.max_num_predicates, task_data['num_items'])
                
                # Log some stats
                num_demos = len(task_data['demo_vectors'])
                num_items = task_data['num_items']
                logger.debug(f"Loaded task {task_id}: {num_demos} demos, {num_items} items")
            else:
                logger.warning(f"Missing predicate data for task {task_id}: {filepath}")
        
        if len(self.task_data) == 0:
            raise RuntimeError(f"No predicate data files found in {self.data_path}")
        
        logger.info(f"Loaded predicate data for {len(self.task_data)} tasks")
    
    def get_predicate_state(
        self, 
        task_id: int, 
        episode_id: int, 
        frame_idx: int,
        max_predicates: int = 20
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get predicate state for a specific frame.
        
        Args:
            task_id: Task index (0-49)
            episode_id: Episode index from dataset
            frame_idx: Frame index within episode
            max_predicates: Maximum number of predicates to pad to
            
        Returns:
            predicate_states: [max_predicates] bool array, True = object is done
            predicate_mask: [max_predicates] bool array, True = valid predicate
        """
        # Initialize output arrays
        predicate_states = np.zeros(max_predicates, dtype=bool)
        predicate_mask = np.zeros(max_predicates, dtype=bool)
        
        # Check if task exists
        if task_id not in self.task_data:
            logger.warning(f"Task {task_id} not found in predicate data, returning zeros")
            return predicate_states, predicate_mask
        
        task_data = self.task_data[task_id]
        num_items = task_data['num_items']
        demo_vectors = task_data['demo_vectors']
        
        # Format episode_id to match stored format (8-digit string like "00010010")
        # First 4 digits = task_id, last 4 digits = demo number within task
        # But the stored format might vary, so we try multiple formats
        episode_id_str = None
        
        # Try different formats
        candidates = [
            f"{episode_id:08d}",  # Full 8-digit
            f"{task_id:04d}{episode_id % 10000:04d}",  # task_id + demo_num
            str(episode_id),  # Plain number
        ]
        
        for candidate in candidates:
            if candidate in demo_vectors:
                episode_id_str = candidate
                break
        
        if episode_id_str is None:
            # Try to find by substring matching
            for stored_key in demo_vectors.keys():
                if str(episode_id) in stored_key or stored_key.endswith(f"{episode_id % 10000:04d}"):
                    episode_id_str = stored_key
                    break
        
        if episode_id_str is None:
            logger.debug(f"Episode {episode_id} not found for task {task_id}, returning zeros")
            return predicate_states, predicate_mask
        
        # Get state vector for this frame
        states = demo_vectors[episode_id_str]['states']  # [T, num_items]
        
        # Clamp frame_idx to valid range
        frame_idx = max(0, min(frame_idx, states.shape[0] - 1))
        
        # Get predicate states for this frame
        frame_states = states[frame_idx]  # [num_items]
        
        # Fill output arrays
        predicate_states[:num_items] = frame_states.astype(bool)
        predicate_mask[:num_items] = True
        
        return predicate_states, predicate_mask
    
    def get_num_predicates(self, task_id: int) -> int:
        """Get number of predicates for a task."""
        if task_id in self.task_data:
            return self.task_data[task_id]['num_items']
        return 0
    
    def get_item_names(self, task_id: int) -> Dict[int, str]:
        """Get item index to name mapping for a task."""
        if task_id in self.task_data:
            return self.task_data[task_id].get('index_to_item', {})
        return {}


# Global singleton instance (lazy initialization)
_predicate_store: Optional[PredicateDataStore] = None


def get_predicate_store(predicate_data_path: str) -> PredicateDataStore:
    """Get or create the global predicate data store.
    
    Uses lazy initialization with singleton pattern for efficiency.
    """
    global _predicate_store
    
    if _predicate_store is None:
        logger.info(f"Initializing PredicateDataStore from {predicate_data_path}")
        _predicate_store = PredicateDataStore(predicate_data_path)
    
    return _predicate_store


def reset_predicate_store():
    """Reset the global predicate store (for testing)."""
    global _predicate_store
    _predicate_store = None
