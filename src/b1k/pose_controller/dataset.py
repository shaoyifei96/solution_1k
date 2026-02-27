"""Dataset for whole-body pose controller training.

Extracts (current_state, future_eef_pose, action) tuples from BEHAVIOR-1K data.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
from pathlib import Path
from typing import Optional, List
import logging
from tqdm import tqdm

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


# =============================================================================
# STANDARD TRACK ALLOWED INDICES
# =============================================================================
# These are the ONLY proprioception indices allowed in the standard track.
# NOT ALLOWED (must be masked/excluded):
#   - joint_qpos[0:6], joint_qpos_sin[0:6], joint_qpos_cos[0:6] (base joints)
#   - robot_pos[140:143] (global position)
#   - robot_ori_cos[143:146], robot_ori_sin[146:149] (global orientation)
#   - robot_2d_ori[149:150], robot_2d_ori_cos[150:151], robot_2d_ori_sin[151:152]
#   - base_qpos[244:247], base_qpos_sin[247:250], base_qpos_cos[250:253]

# Allowed indices (concatenated into a smaller state vector)
# EXCLUDED from standard track allowed:
#   - joint_qeffort (28 dims) - too noisy
#   - robot_lin_vel (3 dims) - correlated to command
#   - robot_ang_vel (3 dims) - correlated to command
#   - gripper_left_qpos/qvel (4 dims) - not needed for EEF pose control
#   - gripper_right_qpos/qvel (4 dims) - not needed for EEF pose control
#   - base_qvel (3 dims) - correlated to command
ALLOWED_PROPRIO_SLICES = [
    # Joint positions (excluding first 6 base joints) -> indices 6:28 = 22 dims
    slice(6, 28),
    # Joint position sin (excluding first 6) -> indices 34:56 = 22 dims  
    slice(34, 56),
    # Joint position cos (excluding first 6) -> indices 62:84 = 22 dims
    slice(62, 84),
    # Joint velocities -> indices 84:112 = 28 dims
    slice(84, 112),
    # EXCLUDED: Joint efforts (112:140) - too noisy
    # EXCLUDED: Robot linear velocity (152:155) - correlated to command
    # EXCLUDED: Robot angular velocity (155:158) - correlated to command
    # Left arm qpos -> indices 158:165 = 7 dims
    slice(158, 165),
    # Left arm qpos sin -> indices 165:172 = 7 dims
    slice(165, 172),
    # Left arm qpos cos -> indices 172:179 = 7 dims
    slice(172, 179),
    # Left arm qvel -> indices 179:186 = 7 dims
    slice(179, 186),
    # EEF left pos -> indices 186:189 = 3 dims
    slice(186, 189),
    # EEF left quat -> indices 189:193 = 4 dims
    slice(189, 193),
    # EXCLUDED: Gripper left qpos (193:195) - not needed for EEF pose control
    # EXCLUDED: Gripper left qvel (195:197) - not needed for EEF pose control
    # Right arm qpos -> indices 197:204 = 7 dims
    slice(197, 204),
    # Right arm qpos sin -> indices 204:211 = 7 dims
    slice(204, 211),
    # Right arm qpos cos -> indices 211:218 = 7 dims
    slice(211, 218),
    # Right arm qvel -> indices 218:225 = 7 dims
    slice(218, 225),
    # EEF right pos -> indices 225:228 = 3 dims
    slice(225, 228),
    # EEF right quat -> indices 228:232 = 4 dims
    slice(228, 232),
    # EXCLUDED: Gripper right qpos (232:234) - not needed for EEF pose control
    # EXCLUDED: Gripper right qvel (234:236) - not needed for EEF pose control
    # Trunk qpos -> indices 236:240 = 4 dims
    slice(236, 240),
    # Trunk qvel -> indices 240:244 = 4 dims
    slice(240, 244),
    # EXCLUDED: Base qvel (253:256) - correlated to command
]

# Total allowed state dim = 22+22+22+28 + 7+7+7+7+3+4 + 7+7+7+7+3+4 + 4+4 = 172
ALLOWED_STATE_DIM = 172


def filter_state_standard_track(state: np.ndarray) -> np.ndarray:
    """
    Filter state vector to only include standard track allowed indices.
    
    Args:
        state: [..., 256] full proprioception
        
    Returns:
        filtered_state: [..., 217] standard track allowed proprioception
    """
    parts = [state[..., s] for s in ALLOWED_PROPRIO_SLICES]
    return np.concatenate(parts, axis=-1).astype(np.float32)


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


def compute_delta_action(current_state: np.ndarray, future_state: np.ndarray, action: np.ndarray) -> np.ndarray:
    """
    Compute delta action: target_joint_positions - current_joint_positions.
    
    The delta represents how much joint positions need to change from current
    to reach the target (future) configuration.
    
    Args:
        current_state: [256] current proprioception
        future_state: [256] future proprioception (at t + lookahead)
        action: [23] original action (needed for base velocity)
    
    Returns:
        action_delta: [23] = base_vel(3) + trunk_delta(4) + left_arm_delta(7) 
                            + left_grip_delta(1) + right_arm_delta(7) + right_grip_delta(1)
    
    Note: Base velocity is kept as-is from the action (already a velocity command).
          Only joint positions are converted to deltas.
    """
    # Base velocity - keep as-is from the original action (already velocity)
    base_vel = action[ActionIndices.BASE]
    
    # Trunk joint position delta
    trunk_delta = future_state[ProprioIndices.TRUNK_QPOS] - current_state[ProprioIndices.TRUNK_QPOS]
    
    # Left arm joint position delta
    left_arm_delta = future_state[ProprioIndices.ARM_LEFT_QPOS] - current_state[ProprioIndices.ARM_LEFT_QPOS]
    
    # Left gripper delta (sum of 2 finger positions -> 1 value)
    left_grip_current = current_state[ProprioIndices.GRIPPER_LEFT_QPOS].sum(keepdims=True)
    left_grip_future = future_state[ProprioIndices.GRIPPER_LEFT_QPOS].sum(keepdims=True)
    left_grip_delta = left_grip_future - left_grip_current
    
    # Right arm joint position delta
    right_arm_delta = future_state[ProprioIndices.ARM_RIGHT_QPOS] - current_state[ProprioIndices.ARM_RIGHT_QPOS]
    
    # Right gripper delta
    right_grip_current = current_state[ProprioIndices.GRIPPER_RIGHT_QPOS].sum(keepdims=True)
    right_grip_future = future_state[ProprioIndices.GRIPPER_RIGHT_QPOS].sum(keepdims=True)
    right_grip_delta = right_grip_future - right_grip_current
    
    return np.concatenate([
        base_vel,          # [0:3] - velocity, unchanged
        trunk_delta,       # [3:7]
        left_arm_delta,    # [7:14]
        left_grip_delta,   # [14:15]
        right_arm_delta,   # [15:22]
        right_grip_delta,  # [22:23]
    ], axis=-1).astype(np.float32)


class PoseControlDatasetLazy(Dataset):
    """
    Lazy-loading dataset for whole-body pose controller.
    
    Builds an index of (file, start_idx, end_idx) tuples and loads data on-demand.
    Much faster to initialize than loading everything into memory.
    """
    
    def __init__(
        self,
        data_root: str | Path,
        lookahead_steps: int = LOOKAHEAD_STEPS,
        max_tasks: Optional[int] = None,
        max_episodes_per_task: Optional[int] = None,
        cache_size: int = 10,  # Number of episodes to cache in memory
    ):
        self.data_root = Path(data_root)
        self.lookahead_steps = lookahead_steps
        
        # Build index: list of (parquet_file, episode_length)
        self.index = []  # List of (file_path, num_valid_samples)
        self.cumulative_lengths = [0]
        
        self._build_index(max_tasks, max_episodes_per_task)
        
        # LRU cache for loaded episodes
        self._cache = {}
        self._cache_order = []
        self._cache_size = cache_size
        
        logger.info(f"Indexed {len(self.index)} episodes, {len(self)} total samples")
    
    def _build_index(self, max_tasks: Optional[int], max_episodes: Optional[int]):
        """Build index of all parquet files without loading data."""
        import pandas as pd
        
        data_path = self.data_root / "data"
        task_dirs = sorted(data_path.glob("task-*"))
        
        if max_tasks:
            task_dirs = task_dirs[:max_tasks]
        
        for task_dir in tqdm(task_dirs, desc="Indexing tasks"):
            episode_files = sorted(task_dir.glob("episode_*.parquet"))
            
            if max_episodes:
                episode_files = episode_files[:max_episodes]
            
            for ep_file in episode_files:
                # Just read the length without loading all data
                # Use pyarrow for fast metadata reading
                try:
                    import pyarrow.parquet as pq
                    metadata = pq.read_metadata(ep_file)
                    episode_len = metadata.num_rows
                except:
                    # Fallback: read just one column
                    df = pd.read_parquet(ep_file, columns=["index"])
                    episode_len = len(df)
                
                # Number of valid samples = episode_len - lookahead
                valid_samples = max(0, episode_len - self.lookahead_steps)
                
                if valid_samples > 0:
                    self.index.append((ep_file, episode_len))
                    self.cumulative_lengths.append(
                        self.cumulative_lengths[-1] + valid_samples
                    )
    
    def __len__(self):
        return self.cumulative_lengths[-1]
    
    def _load_episode(self, file_path: Path) -> tuple[np.ndarray, np.ndarray]:
        """Load and cache an episode."""
        import pandas as pd
        
        # Check cache
        key = str(file_path)
        if key in self._cache:
            return self._cache[key]
        
        # Load data
        df = pd.read_parquet(file_path)
        states = np.stack(df["observation.state"].values).astype(np.float32)
        actions = np.stack(df["action"].values).astype(np.float32)
        
        # Update cache
        if len(self._cache_order) >= self._cache_size:
            # Remove oldest
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]
        
        self._cache[key] = (states, actions)
        self._cache_order.append(key)
        
        return states, actions
    
    def _find_episode(self, idx: int) -> tuple[int, int]:
        """Find which episode contains sample idx, and the offset within it."""
        # Binary search
        import bisect
        episode_idx = bisect.bisect_right(self.cumulative_lengths, idx) - 1
        offset = idx - self.cumulative_lengths[episode_idx]
        return episode_idx, offset
    
    def __getitem__(self, idx):
        episode_idx, offset = self._find_episode(idx)
        file_path, _ = self.index[episode_idx]
        
        states, actions = self._load_episode(file_path)
        
        current_state = states[offset]
        future_state = states[offset + self.lookahead_steps]
        action = actions[offset]
        
        # Extract EEF poses
        target_eef_left = np.concatenate([
            future_state[ProprioIndices.EEF_LEFT_POS],
            future_state[ProprioIndices.EEF_LEFT_QUAT],
        ])
        target_eef_right = np.concatenate([
            future_state[ProprioIndices.EEF_RIGHT_POS],
            future_state[ProprioIndices.EEF_RIGHT_QUAT],
        ])
        
        return {
            "state": torch.from_numpy(current_state),
            "target_eef_left": torch.from_numpy(target_eef_left),
            "target_eef_right": torch.from_numpy(target_eef_right),
            "action": torch.from_numpy(action),
        }


class PoseControlDataset(Dataset):
    """
    Dataset for whole-body pose controller (loads all data into memory).
    
    Each sample contains:
        - state: current proprioception [256]
        - target_eef_left: future left EEF pose [7]
        - target_eef_right: future right EEF pose [7]
        - action: action to take [23]
    
    The "target" EEF pose is extracted from lookahead_steps into the future,
    representing where the end-effector will be if this action is taken.
    
    Use PoseControlDatasetLazy for large datasets.
    """
    
    def __init__(
        self,
        data_root: str | Path,
        tasks: Optional[List[str]] = None,
        lookahead_steps: int = LOOKAHEAD_STEPS,
        normalize: bool = True,
        max_tasks: Optional[int] = None,
        max_episodes_per_task: Optional[int] = None,
    ):
        """
        Args:
            data_root: Path to BEHAVIOR-1K dataset root
            tasks: List of task names to load. If None, loads all.
            lookahead_steps: How many steps ahead to look for target EEF pose
            normalize: Whether to normalize inputs/outputs
            max_tasks: Limit number of tasks (for debugging)
            max_episodes_per_task: Limit episodes per task (for debugging)
        """
        self.data_root = Path(data_root)
        self.lookahead_steps = lookahead_steps
        self.normalize = normalize
        
        # Load all training samples
        self.samples = []
        self._load_data(tasks, max_tasks, max_episodes_per_task)
        
        logger.info(f"Loaded {len(self.samples)} training samples")
        
        # Compute normalization stats
        if normalize and len(self.samples) > 0:
            self._compute_norm_stats()
    
    def _load_data(self, tasks: Optional[List[str]], max_tasks: Optional[int], max_episodes: Optional[int]):
        """Load data from parquet files."""
        import pandas as pd
        
        data_path = self.data_root / "data"
        
        # Get task directories
        task_dirs = sorted(data_path.glob("task-*"))
        
        if max_tasks:
            task_dirs = task_dirs[:max_tasks]
        
        for task_dir in tqdm(task_dirs, desc="Loading tasks"):
            # Load episodes
            episode_files = sorted(task_dir.glob("episode_*.parquet"))
            
            if max_episodes:
                episode_files = episode_files[:max_episodes]
            
            for ep_file in episode_files:
                df = pd.read_parquet(ep_file)
                
                states = np.stack(df["observation.state"].values)
                actions = np.stack(df["action"].values)
                
                # Create training pairs with lookahead
                self._extract_pairs(states, actions)
    
    def _extract_pairs(self, states: np.ndarray, actions: np.ndarray):
        """Extract (state, target_eef, action) pairs from a trajectory."""
        T = len(states)
        
        for t in range(T - self.lookahead_steps):
            current_state = states[t]
            future_state = states[t + self.lookahead_steps]
            action = actions[t]
            
            # Target EEF = where the hand will be in lookahead_steps
            target_eef_left = extract_eef_pose(future_state, "left")
            target_eef_right = extract_eef_pose(future_state, "right")
            
            self.samples.append({
                "state": current_state.astype(np.float32),
                "target_eef_left": target_eef_left.astype(np.float32),
                "target_eef_right": target_eef_right.astype(np.float32),
                "action": action.astype(np.float32),
            })
    
    def _compute_norm_stats(self):
        """Compute mean/std for normalization."""
        states = np.stack([s["state"] for s in self.samples])
        actions = np.stack([s["action"] for s in self.samples])
        
        self.state_mean = states.mean(axis=0)
        self.state_std = states.std(axis=0) + 1e-6
        self.action_mean = actions.mean(axis=0)
        self.action_std = actions.std(axis=0) + 1e-6
        
        # EEF stats
        eef_left = np.stack([s["target_eef_left"] for s in self.samples])
        eef_right = np.stack([s["target_eef_right"] for s in self.samples])
        self.eef_left_mean = eef_left.mean(axis=0)
        self.eef_left_std = eef_left.std(axis=0) + 1e-6
        self.eef_right_mean = eef_right.mean(axis=0)
        self.eef_right_std = eef_right.std(axis=0) + 1e-6
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        state = torch.from_numpy(sample["state"])
        target_left = torch.from_numpy(sample["target_eef_left"])
        target_right = torch.from_numpy(sample["target_eef_right"])
        action = torch.from_numpy(sample["action"])
        
        if self.normalize:
            state = (state - torch.from_numpy(self.state_mean)) / torch.from_numpy(self.state_std)
            target_left = (target_left - torch.from_numpy(self.eef_left_mean)) / torch.from_numpy(self.eef_left_std)
            target_right = (target_right - torch.from_numpy(self.eef_right_mean)) / torch.from_numpy(self.eef_right_std)
            action = (action - torch.from_numpy(self.action_mean)) / torch.from_numpy(self.action_std)
        
        return {
            "state": state,
            "target_eef_left": target_left,
            "target_eef_right": target_right,
            "action": action,
        }


class PoseControlDatasetDelta(Dataset):
    """
    Dataset for whole-body pose controller with DELTA action targets.
    
    Instead of predicting absolute joint positions, the network learns to predict:
        delta = future_joint_positions - current_joint_positions
    
    At inference: target_joints = current_joints + predicted_delta
    
    Each sample contains:
        - state: current proprioception [217 if standard_track else 256]
        - target_eef_left: future left EEF pose [7]
        - target_eef_right: future right EEF pose [7]
        - action_delta: joint position deltas [23]
    """
    
    def __init__(
        self,
        data_root: str | Path,
        tasks: Optional[List[str]] = None,
        lookahead_steps: int = LOOKAHEAD_STEPS,
        normalize: bool = True,
        max_tasks: Optional[int] = None,
        max_episodes_per_task: Optional[int] = None,
        standard_track: bool = True,  # Filter to only standard track allowed indices
    ):
        """
        Args:
            data_root: Path to BEHAVIOR-1K dataset root
            tasks: List of task names to load. If None, loads all.
            lookahead_steps: How many steps ahead to look for target joint positions
            normalize: Whether to normalize inputs/outputs
            max_tasks: Limit number of tasks (for debugging)
            max_episodes_per_task: Limit episodes per task (for debugging)
            standard_track: If True, filter state to only include allowed indices (217 dims).
                           If False, use full 256-dim state (includes disallowed global info).
        """
        self.data_root = Path(data_root)
        self.lookahead_steps = lookahead_steps
        self.normalize = normalize
        self.standard_track = standard_track
        self.state_dim = ALLOWED_STATE_DIM if standard_track else 256
        
        # Load all training samples
        self.samples = []
        self._load_data(tasks, max_tasks, max_episodes_per_task)
        
        logger.info(f"Loaded {len(self.samples)} training samples (delta mode, state_dim={self.state_dim})")
        
        # Compute normalization stats
        if normalize and len(self.samples) > 0:
            self._compute_norm_stats()
    
    def _load_data(self, tasks: Optional[List[str]], max_tasks: Optional[int], max_episodes: Optional[int]):
        """Load data from parquet files."""
        import pandas as pd
        
        data_path = self.data_root / "data"
        
        # Get task directories
        task_dirs = sorted(data_path.glob("task-*"))
        
        if max_tasks:
            task_dirs = task_dirs[:max_tasks]
        
        for task_dir in tqdm(task_dirs, desc="Loading tasks (delta)"):
            # Load episodes
            episode_files = sorted(task_dir.glob("episode_*.parquet"))
            
            if max_episodes:
                episode_files = episode_files[:max_episodes]
            
            for ep_file in episode_files:
                df = pd.read_parquet(ep_file)
                
                states = np.stack(df["observation.state"].values)
                actions = np.stack(df["action"].values)
                
                # Create training pairs with delta targets
                self._extract_pairs(states, actions)
    
    def _extract_pairs(self, states: np.ndarray, actions: np.ndarray):
        """Extract (state, target_eef, action_delta) pairs from a trajectory."""
        T = len(states)
        
        for t in range(T - self.lookahead_steps):
            current_state_full = states[t]
            future_state_full = states[t + self.lookahead_steps]
            action = actions[t]
            
            # Target EEF = where the hand will be in lookahead_steps
            # (EEF positions are body-frame, always allowed)
            target_eef_left = extract_eef_pose(future_state_full, "left")
            target_eef_right = extract_eef_pose(future_state_full, "right")
            
            # Compute delta: how much joints need to change (base_vel stays as-is)
            # Note: compute_delta_action uses full state for joint position extraction
            action_delta = compute_delta_action(current_state_full, future_state_full, action)
            
            # Filter state to standard track allowed indices if requested
            if self.standard_track:
                current_state = filter_state_standard_track(current_state_full)
            else:
                current_state = current_state_full.astype(np.float32)
            
            self.samples.append({
                "state": current_state,
                "target_eef_left": target_eef_left.astype(np.float32),
                "target_eef_right": target_eef_right.astype(np.float32),
                "action_delta": action_delta,
            })
    
    def _compute_norm_stats(self):
        """Compute mean/std for normalization."""
        states = np.stack([s["state"] for s in self.samples])
        action_deltas = np.stack([s["action_delta"] for s in self.samples])
        
        self.state_mean = states.mean(axis=0)
        self.state_std = states.std(axis=0) + 1e-6
        self.action_delta_mean = action_deltas.mean(axis=0)
        self.action_delta_std = action_deltas.std(axis=0) + 1e-6
        
        # EEF stats
        eef_left = np.stack([s["target_eef_left"] for s in self.samples])
        eef_right = np.stack([s["target_eef_right"] for s in self.samples])
        self.eef_left_mean = eef_left.mean(axis=0)
        self.eef_left_std = eef_left.std(axis=0) + 1e-6
        self.eef_right_mean = eef_right.mean(axis=0)
        self.eef_right_std = eef_right.std(axis=0) + 1e-6
        
        logger.info(f"Delta action stats - mean: {self.action_delta_mean}, std: {self.action_delta_std}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        state = torch.from_numpy(sample["state"])
        target_left = torch.from_numpy(sample["target_eef_left"])
        target_right = torch.from_numpy(sample["target_eef_right"])
        action_delta = torch.from_numpy(sample["action_delta"])
        
        if self.normalize:
            state = (state - torch.from_numpy(self.state_mean)) / torch.from_numpy(self.state_std)
            target_left = (target_left - torch.from_numpy(self.eef_left_mean)) / torch.from_numpy(self.eef_left_std)
            target_right = (target_right - torch.from_numpy(self.eef_right_mean)) / torch.from_numpy(self.eef_right_std)
            action_delta = (action_delta - torch.from_numpy(self.action_delta_mean)) / torch.from_numpy(self.action_delta_std)
        
        return {
            "state": state,
            "target_eef_left": target_left,
            "target_eef_right": target_right,
            "action": action_delta,  # Named "action" for compatibility with training loop
        }


class PoseControlIterableDataset(IterableDataset):
    """
    Iterable version for streaming from BehaviorLeRobotDataset.
    
    Uses the existing data loading infrastructure from b1k training.
    """
    
    def __init__(
        self,
        data_config,  # openpi DataConfig
        lookahead_steps: int = LOOKAHEAD_STEPS,
    ):
        self.data_config = data_config
        self.lookahead_steps = lookahead_steps
        
        # Will be initialized in worker
        self._dataset = None
        self._buffer = []
    
    def _init_dataset(self):
        """Initialize the underlying BehaviorLeRobotDataset."""
        from omnigibson.learning.datas.lerobot_dataset import BehaviorLeRobotDataset
        
        self._dataset = BehaviorLeRobotDataset(
            repo_id=self.data_config.repo_id,
            root=self.data_config.behavior_dataset_root,
            modalities=["rgb"],
            local_only=True,
            delta_timestamps={
                "observation.state": [t / 30.0 for t in range(self.lookahead_steps + 1)],
                "action": [0.0],
            },
            shuffle=True,
        )
    
    def __iter__(self):
        if self._dataset is None:
            self._init_dataset()
        
        for item in self._dataset:
            # item["observation.state"] has shape [lookahead+1, 256]
            states = item["observation.state"]
            action = item["action"][0]  # Current action
            
            current_state = states[0]
            future_state = states[self.lookahead_steps]
            
            target_eef_left = torch.cat([
                future_state[ProprioIndices.EEF_LEFT_POS],
                future_state[ProprioIndices.EEF_LEFT_QUAT],
            ])
            target_eef_right = torch.cat([
                future_state[ProprioIndices.EEF_RIGHT_POS],
                future_state[ProprioIndices.EEF_RIGHT_QUAT],
            ])
            
            yield {
                "state": current_state,
                "target_eef_left": target_eef_left,
                "target_eef_right": target_eef_right,
                "action": action,
            }
