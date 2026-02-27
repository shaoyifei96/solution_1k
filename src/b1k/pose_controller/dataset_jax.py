"""Dataset for whole-body pose controller training (NumPy version for JAX).

Extracts (current_state, future_eef_pose, action) tuples from BEHAVIOR-1K data.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Iterator
import logging
from tqdm import tqdm
import bisect

logger = logging.getLogger(__name__)


# Default lookahead: 15 steps = 0.5 sec at 30Hz
LOOKAHEAD_STEPS = 15


# R1Pro proprioception indices (from eval_utils.py)
class ProprioIndices:
    """Indices into the 256-dim state vector."""
    # Joint positions
    JOINT_QPOS = slice(0, 28)
    ARM_LEFT_QPOS = slice(158, 165)      # 7 dims
    ARM_LEFT_QPOS_SIN = slice(165, 172)
    ARM_LEFT_QPOS_COS = slice(172, 179)
    ARM_RIGHT_QPOS = slice(197, 204)     # 7 dims
    ARM_RIGHT_QPOS_SIN = slice(204, 211)
    ARM_RIGHT_QPOS_COS = slice(211, 218)
    TRUNK_QPOS = slice(236, 240)         # 4 dims
    BASE_QPOS = slice(244, 247)          # 3 dims (x, y, theta)
    
    # End-effector poses
    EEF_LEFT_POS = slice(186, 189)       # 3 dims
    EEF_LEFT_QUAT = slice(189, 193)      # 4 dims
    EEF_RIGHT_POS = slice(225, 228)      # 3 dims
    EEF_RIGHT_QUAT = slice(228, 232)     # 4 dims
    
    # Gripper
    GRIPPER_LEFT_QPOS = slice(193, 195)  # 2 dims
    GRIPPER_RIGHT_QPOS = slice(232, 234) # 2 dims
    
    # Velocities
    ARM_LEFT_QVEL = slice(179, 186)
    ARM_RIGHT_QVEL = slice(218, 225)
    TRUNK_QVEL = slice(240, 244)
    BASE_QVEL = slice(253, 256)


# Action indices (23-dim action)
class ActionIndices:
    """Indices into the 23-dim action vector."""
    BASE = slice(0, 3)           # velocity: vx, vy, vtheta
    TORSO = slice(3, 7)          # position: 4 joints
    LEFT_ARM = slice(7, 14)      # position: 7 joints
    LEFT_GRIPPER = slice(14, 15) # position: 1 joint
    RIGHT_ARM = slice(15, 22)    # position: 7 joints
    RIGHT_GRIPPER = slice(22, 23)# position: 1 joint


def extract_eef_pose(state: np.ndarray, side: str = "left") -> np.ndarray:
    """Extract EEF pose (pos + quat) from state vector."""
    if side == "left":
        pos = state[..., ProprioIndices.EEF_LEFT_POS]
        quat = state[..., ProprioIndices.EEF_LEFT_QUAT]
    else:
        pos = state[..., ProprioIndices.EEF_RIGHT_POS]
        quat = state[..., ProprioIndices.EEF_RIGHT_QUAT]
    return np.concatenate([pos, quat], axis=-1)


class PoseControlDataLoader:
    """
    Streaming data loader for whole-body pose controller.
    
    Loads data episode by episode and yields batches.
    Memory efficient - only keeps current batch in memory.
    """
    
    def __init__(
        self,
        data_root: str | Path,
        batch_size: int = 256,
        lookahead_steps: int = LOOKAHEAD_STEPS,
        max_tasks: Optional[int] = None,
        max_episodes_per_task: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.data_root = Path(data_root)
        self.batch_size = batch_size
        self.lookahead_steps = lookahead_steps
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        
        # Build index of all parquet files
        self.episode_files = self._find_episodes(max_tasks, max_episodes_per_task)
        logger.info(f"Found {len(self.episode_files)} episodes")
        
        # Compute total samples (for epoch counting)
        self._total_samples = None  # Lazy computed
    
    def _find_episodes(self, max_tasks: Optional[int], max_episodes: Optional[int]) -> list[Path]:
        """Find all episode parquet files."""
        data_path = self.data_root / "data"
        task_dirs = sorted(data_path.glob("task-*"))
        
        if max_tasks:
            task_dirs = task_dirs[:max_tasks]
        
        episode_files = []
        for task_dir in tqdm(task_dirs, desc="Finding episodes"):
            files = sorted(task_dir.glob("episode_*.parquet"))
            if max_episodes:
                files = files[:max_episodes]
            episode_files.extend(files)
        
        return episode_files
    
    def __len__(self):
        """Approximate number of batches per epoch."""
        if self._total_samples is None:
            # Estimate: assume ~300 samples per episode on average
            self._total_samples = len(self.episode_files) * 300
        return self._total_samples // self.batch_size
    
    def __iter__(self) -> Iterator[dict[str, np.ndarray]]:
        """Iterate over batches for one epoch."""
        import pandas as pd
        
        # Shuffle episode order
        episode_order = list(range(len(self.episode_files)))
        if self.shuffle:
            self.rng.shuffle(episode_order)
        
        # Accumulate samples for batching
        buffer = {
            "state": [],
            "target_eef_left": [],
            "target_eef_right": [],
            "action": [],
        }
        
        for ep_idx in episode_order:
            file_path = self.episode_files[ep_idx]
            
            # Load episode
            df = pd.read_parquet(file_path)
            states = np.stack(df["observation.state"].values).astype(np.float32)
            actions = np.stack(df["action"].values).astype(np.float32)
            
            episode_len = len(states)
            valid_samples = episode_len - self.lookahead_steps
            
            if valid_samples <= 0:
                continue
            
            # Extract samples from this episode
            for t in range(valid_samples):
                current_state = states[t]
                future_state = states[t + self.lookahead_steps]
                action = actions[t]
                
                # Extract target EEF poses
                target_left = np.concatenate([
                    future_state[ProprioIndices.EEF_LEFT_POS],
                    future_state[ProprioIndices.EEF_LEFT_QUAT],
                ])
                target_right = np.concatenate([
                    future_state[ProprioIndices.EEF_RIGHT_POS],
                    future_state[ProprioIndices.EEF_RIGHT_QUAT],
                ])
                
                buffer["state"].append(current_state)
                buffer["target_eef_left"].append(target_left)
                buffer["target_eef_right"].append(target_right)
                buffer["action"].append(action)
                
                # Yield batch when full
                if len(buffer["state"]) >= self.batch_size:
                    batch = {
                        k: np.stack(v[:self.batch_size])
                        for k, v in buffer.items()
                    }
                    
                    # Keep remainder
                    buffer = {
                        k: v[self.batch_size:]
                        for k, v in buffer.items()
                    }
                    
                    yield batch
        
        # Yield final partial batch if any
        if len(buffer["state"]) > 0:
            yield {k: np.stack(v) for k, v in buffer.items()}


class PoseControlDataset:
    """
    In-memory dataset for smaller data sizes.
    
    Loads all data into memory at init time.
    """
    
    def __init__(
        self,
        data_root: str | Path,
        lookahead_steps: int = LOOKAHEAD_STEPS,
        max_tasks: Optional[int] = None,
        max_episodes_per_task: Optional[int] = None,
    ):
        self.data_root = Path(data_root)
        self.lookahead_steps = lookahead_steps
        
        # Load all data
        self.states = []
        self.target_eef_left = []
        self.target_eef_right = []
        self.actions = []
        
        self._load_data(max_tasks, max_episodes_per_task)
        
        # Stack into arrays
        self.states = np.stack(self.states).astype(np.float32)
        self.target_eef_left = np.stack(self.target_eef_left).astype(np.float32)
        self.target_eef_right = np.stack(self.target_eef_right).astype(np.float32)
        self.actions = np.stack(self.actions).astype(np.float32)
        
        logger.info(f"Loaded {len(self.states)} samples")
    
    def _load_data(self, max_tasks: Optional[int], max_episodes: Optional[int]):
        """Load data from parquet files."""
        import pandas as pd
        
        data_path = self.data_root / "data"
        task_dirs = sorted(data_path.glob("task-*"))
        
        if max_tasks:
            task_dirs = task_dirs[:max_tasks]
        
        for task_dir in tqdm(task_dirs, desc="Loading tasks"):
            episode_files = sorted(task_dir.glob("episode_*.parquet"))
            
            if max_episodes:
                episode_files = episode_files[:max_episodes]
            
            for ep_file in episode_files:
                df = pd.read_parquet(ep_file)
                states = np.stack(df["observation.state"].values).astype(np.float32)
                actions = np.stack(df["action"].values).astype(np.float32)
                
                episode_len = len(states)
                valid_samples = episode_len - self.lookahead_steps
                
                for t in range(valid_samples):
                    current_state = states[t]
                    future_state = states[t + self.lookahead_steps]
                    action = actions[t]
                    
                    target_left = np.concatenate([
                        future_state[ProprioIndices.EEF_LEFT_POS],
                        future_state[ProprioIndices.EEF_LEFT_QUAT],
                    ])
                    target_right = np.concatenate([
                        future_state[ProprioIndices.EEF_RIGHT_POS],
                        future_state[ProprioIndices.EEF_RIGHT_QUAT],
                    ])
                    
                    self.states.append(current_state)
                    self.target_eef_left.append(target_left)
                    self.target_eef_right.append(target_right)
                    self.actions.append(action)
    
    def __len__(self):
        return len(self.states)
    
    def get_batches(
        self, 
        batch_size: int, 
        shuffle: bool = True, 
        rng: Optional[np.random.Generator] = None,
    ) -> Iterator[dict[str, np.ndarray]]:
        """Yield batches for one epoch."""
        n = len(self.states)
        indices = np.arange(n)
        
        if shuffle:
            if rng is None:
                rng = np.random.default_rng()
            rng.shuffle(indices)
        
        for i in range(0, n, batch_size):
            batch_idx = indices[i:i + batch_size]
            yield {
                "state": self.states[batch_idx],
                "target_eef_left": self.target_eef_left[batch_idx],
                "target_eef_right": self.target_eef_right[batch_idx],
                "action": self.actions[batch_idx],
            }
    
    def split(self, val_ratio: float = 0.1, seed: int = 42):
        """Split into train/val datasets."""
        n = len(self.states)
        indices = np.arange(n)
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
        
        val_size = int(n * val_ratio)
        val_idx = indices[:val_size]
        train_idx = indices[val_size:]
        
        train_data = PoseControlDatasetSplit(
            self.states[train_idx],
            self.target_eef_left[train_idx],
            self.target_eef_right[train_idx],
            self.actions[train_idx],
        )
        val_data = PoseControlDatasetSplit(
            self.states[val_idx],
            self.target_eef_left[val_idx],
            self.target_eef_right[val_idx],
            self.actions[val_idx],
        )
        
        return train_data, val_data


class PoseControlDatasetSplit:
    """A split of PoseControlDataset."""
    
    def __init__(
        self,
        states: np.ndarray,
        target_eef_left: np.ndarray,
        target_eef_right: np.ndarray,
        actions: np.ndarray,
    ):
        self.states = states
        self.target_eef_left = target_eef_left
        self.target_eef_right = target_eef_right
        self.actions = actions
    
    def __len__(self):
        return len(self.states)
    
    def get_batches(
        self, 
        batch_size: int, 
        shuffle: bool = True, 
        rng: Optional[np.random.Generator] = None,
    ) -> Iterator[dict[str, np.ndarray]]:
        """Yield batches for one epoch."""
        n = len(self.states)
        indices = np.arange(n)
        
        if shuffle:
            if rng is None:
                rng = np.random.default_rng()
            rng.shuffle(indices)
        
        for i in range(0, n, batch_size):
            batch_idx = indices[i:i + batch_size]
            yield {
                "state": self.states[batch_idx],
                "target_eef_left": self.target_eef_left[batch_idx],
                "target_eef_right": self.target_eef_right[batch_idx],
                "action": self.actions[batch_idx],
            }
