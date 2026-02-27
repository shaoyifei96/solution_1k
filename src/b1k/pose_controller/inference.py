"""Inference utilities for pose controller.

Use this at deployment time to get actions from desired EEF poses.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from b1k.pose_controller.model import WholeBodyPoseController, WholeBodyPoseControllerWithResidual
from b1k.pose_controller.dataset import ProprioIndices


class PoseControllerInference:
    """
    Wrapper for inference with trained pose controller.
    
    Usage:
        controller = PoseControllerInference.load("outputs/pose_controller/run_xxx")
        
        # At each step:
        action = controller.get_action(
            current_state,      # [256] proprioception
            target_eef_left,    # [7] pos + quat
            target_eef_right,   # [7] pos + quat
        )
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        norm_stats: dict,
        device: str = "cuda",
    ):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        # Convert norm stats to tensors
        self.state_mean = torch.tensor(norm_stats["state_mean"], dtype=torch.float32, device=self.device)
        self.state_std = torch.tensor(norm_stats["state_std"], dtype=torch.float32, device=self.device)
        self.action_mean = torch.tensor(norm_stats["action_mean"], dtype=torch.float32, device=self.device)
        self.action_std = torch.tensor(norm_stats["action_std"], dtype=torch.float32, device=self.device)
        self.eef_left_mean = torch.tensor(norm_stats["eef_left_mean"], dtype=torch.float32, device=self.device)
        self.eef_left_std = torch.tensor(norm_stats["eef_left_std"], dtype=torch.float32, device=self.device)
        self.eef_right_mean = torch.tensor(norm_stats["eef_right_mean"], dtype=torch.float32, device=self.device)
        self.eef_right_std = torch.tensor(norm_stats["eef_right_std"], dtype=torch.float32, device=self.device)
    
    @classmethod
    def load(cls, checkpoint_dir: str | Path, checkpoint_name: str = "best_model.pt") -> "PoseControllerInference":
        """Load controller from checkpoint directory."""
        checkpoint_dir = Path(checkpoint_dir)
        
        # Load config
        with open(checkpoint_dir / "config.json") as f:
            config = json.load(f)
        
        # Load norm stats
        with open(checkpoint_dir / "norm_stats.json") as f:
            norm_stats = json.load(f)
        
        # Create model
        use_residual = config.get("use_residual", False)
        ModelClass = WholeBodyPoseControllerWithResidual if use_residual else WholeBodyPoseController
        
        model = ModelClass(
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            dropout=0.0,
        )
        
        # Load weights
        checkpoint = torch.load(checkpoint_dir / checkpoint_name, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        
        return cls(model, norm_stats)
    
    def _to_tensor(self, x: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Convert input to tensor on correct device."""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        return x.to(self.device)
    
    @torch.no_grad()
    def get_action(
        self,
        current_state: np.ndarray | torch.Tensor,
        target_eef_left: np.ndarray | torch.Tensor,
        target_eef_right: np.ndarray | torch.Tensor,
    ) -> np.ndarray:
        """
        Get action for given state and target EEF poses.
        
        Args:
            current_state: [256] or [B, 256] proprioception
            target_eef_left: [7] or [B, 7] target left EEF (pos + quat)
            target_eef_right: [7] or [B, 7] target right EEF (pos + quat)
            
        Returns:
            action: [23] or [B, 23] action vector
        """
        # Handle single sample
        state = self._to_tensor(current_state)
        target_left = self._to_tensor(target_eef_left)
        target_right = self._to_tensor(target_eef_right)
        
        squeeze = False
        if state.dim() == 1:
            state = state.unsqueeze(0)
            target_left = target_left.unsqueeze(0)
            target_right = target_right.unsqueeze(0)
            squeeze = True
        
        # Normalize
        state_norm = (state - self.state_mean) / self.state_std
        target_left_norm = (target_left - self.eef_left_mean) / self.eef_left_std
        target_right_norm = (target_right - self.eef_right_mean) / self.eef_right_std
        
        # Forward
        action_norm = self.model(state_norm, target_left_norm, target_right_norm)
        
        # Denormalize
        action = action_norm * self.action_std + self.action_mean
        
        if squeeze:
            action = action.squeeze(0)
        
        return action.cpu().numpy()
    
    def get_action_from_obs(
        self,
        obs: dict,
        target_eef_left: np.ndarray,
        target_eef_right: np.ndarray,
    ) -> np.ndarray:
        """
        Get action from observation dict (as returned by environment).
        
        Args:
            obs: Observation dict containing 'state' or 'proprio' key
            target_eef_left: [7] target left EEF pose
            target_eef_right: [7] target right EEF pose
            
        Returns:
            action: [23] action vector
        """
        if "state" in obs:
            state = obs["state"]
        elif "proprio" in obs:
            state = obs["proprio"]
        else:
            raise ValueError("Observation must contain 'state' or 'proprio' key")
        
        return self.get_action(state, target_eef_left, target_eef_right)


def extract_current_eef_from_state(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract current EEF poses from state vector.
    
    Args:
        state: [256] proprioception vector
        
    Returns:
        eef_left: [7] left EEF pose (pos + quat)
        eef_right: [7] right EEF pose (pos + quat)
    """
    eef_left = np.concatenate([
        state[ProprioIndices.EEF_LEFT_POS],
        state[ProprioIndices.EEF_LEFT_QUAT],
    ])
    eef_right = np.concatenate([
        state[ProprioIndices.EEF_RIGHT_POS],
        state[ProprioIndices.EEF_RIGHT_QUAT],
    ])
    return eef_left, eef_right


def interpolate_eef_target(
    current_eef: np.ndarray,
    goal_eef: np.ndarray,
    alpha: float = 0.1,
) -> np.ndarray:
    """
    Interpolate between current and goal EEF pose.
    
    Use this to create intermediate targets for smoother motion.
    
    Args:
        current_eef: [7] current pose
        goal_eef: [7] goal pose
        alpha: interpolation factor (0 = current, 1 = goal)
        
    Returns:
        target_eef: [7] interpolated pose
    """
    # Simple linear interpolation for position
    target_pos = (1 - alpha) * current_eef[:3] + alpha * goal_eef[:3]
    
    # SLERP for quaternion
    from scipy.spatial.transform import Rotation, Slerp
    
    rots = Rotation.from_quat([current_eef[3:7], goal_eef[3:7]])
    slerp = Slerp([0, 1], rots)
    target_quat = slerp([alpha]).as_quat()[0]
    
    return np.concatenate([target_pos, target_quat])
