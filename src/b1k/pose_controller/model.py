"""Whole-body pose controller network.

Maps (current_state, target_eef_left, target_eef_right) → action[23]
"""

import torch
import torch.nn as nn


# =============================================================================
# Standard Track State Filtering (for inference with full 256-dim state)
# =============================================================================
# These slices extract allowed indices from the full 256-dim proprioception
# EXCLUDED: joint_qeffort, robot_lin/ang_vel, gripper_qpos/qvel, base_qvel
ALLOWED_PROPRIO_SLICES_TORCH = [
    slice(6, 28),     # joint_qpos excluding base (22)
    slice(34, 56),    # joint_qpos_sin excluding base (22)
    slice(62, 84),    # joint_qpos_cos excluding base (22)
    slice(84, 112),   # joint_qvel (28)
    # EXCLUDED: joint_qeffort (112:140) - too noisy
    # EXCLUDED: robot_lin_vel (152:155) - correlated to command
    # EXCLUDED: robot_ang_vel (155:158) - correlated to command
    slice(158, 165),  # arm_left_qpos (7)
    slice(165, 172),  # arm_left_qpos_sin (7)
    slice(172, 179),  # arm_left_qpos_cos (7)
    slice(179, 186),  # arm_left_qvel (7)
    slice(186, 189),  # eef_left_pos (3)
    slice(189, 193),  # eef_left_quat (4)
    # EXCLUDED: gripper_left_qpos (193:195) - not needed for EEF pose control
    # EXCLUDED: gripper_left_qvel (195:197) - not needed for EEF pose control
    slice(197, 204),  # arm_right_qpos (7)
    slice(204, 211),  # arm_right_qpos_sin (7)
    slice(211, 218),  # arm_right_qpos_cos (7)
    slice(218, 225),  # arm_right_qvel (7)
    slice(225, 228),  # eef_right_pos (3)
    slice(228, 232),  # eef_right_quat (4)
    # EXCLUDED: gripper_right_qpos (232:234) - not needed for EEF pose control
    # EXCLUDED: gripper_right_qvel (234:236) - not needed for EEF pose control
    slice(236, 240),  # trunk_qpos (4)
    slice(240, 244),  # trunk_qvel (4)
    # EXCLUDED: base_qvel (253:256) - correlated to command
]

# Total: 22+22+22+28 + 7+7+7+7+3+4 + 7+7+7+7+3+4 + 4+4 = 172
ALLOWED_STATE_DIM_TORCH = 172


def filter_state_standard_track_torch(state: torch.Tensor) -> torch.Tensor:
    """
    Filter full 256-dim state to 172-dim filtered state for pose controller.
    
    Use this at inference time when receiving full proprioception from simulator.
    
    Args:
        state: [..., 256] full proprioception tensor
        
    Returns:
        filtered_state: [..., 172] filtered proprioception
    """
    parts = [state[..., s] for s in ALLOWED_PROPRIO_SLICES_TORCH]
    return torch.cat(parts, dim=-1)


class WholeBodyPoseController(nn.Module):
    """
    Data-driven whole-body controller for R1Pro.
    
    Input:
        - current_state: [B, 256] full proprioception
        - target_eef_left: [B, 7] target left EEF pose (pos + quat)
        - target_eef_right: [B, 7] target right EEF pose (pos + quat)
    
    Output:
        - action: [B, 23] = base_vel(3) + trunk(4) + left_arm(7) + left_grip(1) + right_arm(7) + right_grip(1)
    
    The model learns implicitly from data:
        - When to move the base vs just the arms
        - How to coordinate trunk with arm motion
        - Joint limit awareness (from demonstrated trajectories)
    """
    
    def __init__(
        self, 
        state_dim: int = 256, 
        eef_dim: int = 7, 
        action_dim: int = 23, 
        hidden_dim: int = 512,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.eef_dim = eef_dim
        self.action_dim = action_dim
        
        input_dim = state_dim + eef_dim * 2  # 270
        
        # Build encoder layers
        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout) if i < num_layers - 1 else nn.Identity(),
            ])
            in_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Separate heads for different action components
        # This allows different output scales/activations if needed
        self.base_head = nn.Linear(hidden_dim, 3)       # velocity: vx, vy, vtheta
        self.trunk_head = nn.Linear(hidden_dim, 4)      # position: 4 joints
        self.left_arm_head = nn.Linear(hidden_dim, 7)   # position: 7 joints
        self.left_grip_head = nn.Linear(hidden_dim, 1)  # position: 1 joint
        self.right_arm_head = nn.Linear(hidden_dim, 7)  # position: 7 joints
        self.right_grip_head = nn.Linear(hidden_dim, 1) # position: 1 joint
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        state: torch.Tensor, 
        target_eef_left: torch.Tensor, 
        target_eef_right: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: [B, 256] current proprioception
            target_eef_left: [B, 7] target left EEF (pos + quat)
            target_eef_right: [B, 7] target right EEF (pos + quat)
            
        Returns:
            action: [B, 23] full action vector
        """
        # Concatenate inputs
        x = torch.cat([state, target_eef_left, target_eef_right], dim=-1)
        
        # Encode
        features = self.encoder(x)
        
        # Decode to action components
        base_vel = self.base_head(features)
        trunk_pos = self.trunk_head(features)
        left_arm_pos = self.left_arm_head(features)
        left_grip_pos = self.left_grip_head(features)
        right_arm_pos = self.right_arm_head(features)
        right_grip_pos = self.right_grip_head(features)
        
        # Concatenate in action order
        action = torch.cat([
            base_vel,       # [0:3]
            trunk_pos,      # [3:7]
            left_arm_pos,   # [7:14]
            left_grip_pos,  # [14:15]
            right_arm_pos,  # [15:22]
            right_grip_pos, # [22:23]
        ], dim=-1)
        
        return action
    
    def forward_with_components(
        self,
        state: torch.Tensor,
        target_eef_left: torch.Tensor,
        target_eef_right: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass returning action components separately."""
        x = torch.cat([state, target_eef_left, target_eef_right], dim=-1)
        features = self.encoder(x)
        
        return {
            "base_vel": self.base_head(features),
            "trunk_pos": self.trunk_head(features),
            "left_arm_pos": self.left_arm_head(features),
            "left_grip_pos": self.left_grip_head(features),
            "right_arm_pos": self.right_arm_head(features),
            "right_grip_pos": self.right_grip_head(features),
        }


class WholeBodyPoseControllerWithResidual(WholeBodyPoseController):
    """
    Variant that predicts residual from current joint positions.
    
    For joint position outputs (trunk, arms, grippers), predicts delta from current.
    For base velocity, predicts absolute velocity.
    """
    
    # Indices into the 256-dim state vector for current joint positions
    TRUNK_QPOS_IDX = slice(236, 240)      # 4 dims
    LEFT_ARM_QPOS_IDX = slice(158, 165)   # 7 dims  
    RIGHT_ARM_QPOS_IDX = slice(197, 204)  # 7 dims
    LEFT_GRIPPER_QPOS_IDX = slice(193, 195)   # 2 dims (but action is 1 dim)
    RIGHT_GRIPPER_QPOS_IDX = slice(232, 234)  # 2 dims (but action is 1 dim)
    
    def forward(
        self,
        state: torch.Tensor,
        target_eef_left: torch.Tensor,
        target_eef_right: torch.Tensor,
    ) -> torch.Tensor:
        """Forward with residual prediction for joint positions."""
        x = torch.cat([state, target_eef_left, target_eef_right], dim=-1)
        features = self.encoder(x)
        
        # Base velocity: absolute prediction
        base_vel = self.base_head(features)
        
        # Joint positions: predict residual and add to current
        trunk_delta = self.trunk_head(features)
        left_arm_delta = self.left_arm_head(features)
        left_grip_delta = self.left_grip_head(features)
        right_arm_delta = self.right_arm_head(features)
        right_grip_delta = self.right_grip_head(features)
        
        # Get current positions from state
        current_trunk = state[:, self.TRUNK_QPOS_IDX]
        current_left_arm = state[:, self.LEFT_ARM_QPOS_IDX]
        current_right_arm = state[:, self.RIGHT_ARM_QPOS_IDX]
        # For grippers, take first of the 2 dims as position
        current_left_grip = state[:, 193:194]
        current_right_grip = state[:, 232:233]
        
        # Add residuals
        trunk_pos = current_trunk + trunk_delta
        left_arm_pos = current_left_arm + left_arm_delta
        left_grip_pos = current_left_grip + left_grip_delta
        right_arm_pos = current_right_arm + right_arm_delta
        right_grip_pos = current_right_grip + right_grip_delta
        
        action = torch.cat([
            base_vel,
            trunk_pos,
            left_arm_pos,
            left_grip_pos,
            right_arm_pos,
            right_grip_pos,
        ], dim=-1)
        
        return action


class WholeBodyPoseControllerDelta(nn.Module):
    """
    Whole-body controller that outputs DELTA joint angles.
    
    Given (current_state, target_eef_left, target_eef_right), predicts:
        delta = target_joint_positions - current_joint_positions
    
    At inference: 
        target_joints = current_joints + predicted_delta
    
    Input:
        - current_state: [B, state_dim] proprioception (217 for standard track, 256 for full)
        - target_eef_left: [B, 7] target left EEF pose (pos + quat)
        - target_eef_right: [B, 7] target right EEF pose (pos + quat)
    
    Output:
        - action_delta: [B, 23] = base_vel(3) + trunk_delta(4) + left_arm_delta(7) 
                                 + left_grip_delta(1) + right_arm_delta(7) + right_grip_delta(1)
    
    Note: state_dim=172 for filtered state (excludes global pos, effort, gripper, velocities)
          state_dim=256 for full proprioception (not allowed in competition)
    """
    
    # Filtered state dimension (excludes effort, robot vel, gripper, base vel)
    FILTERED_STATE_DIM = 172
    
    def __init__(
        self, 
        state_dim: int = 172,  # Default to filtered state
        eef_dim: int = 7, 
        action_dim: int = 23, 
        hidden_dim: int = 512,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.eef_dim = eef_dim
        self.action_dim = action_dim
        
        input_dim = state_dim + eef_dim * 2  # 172 + 14 = 186 for filtered state
        
        # Build encoder layers
        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout) if i < num_layers - 1 else nn.Identity(),
            ])
            in_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Separate heads for different action components
        # Base velocity (NOT delta - already a velocity command)
        self.base_delta_head = nn.Linear(hidden_dim, 3)       # velocity: vx, vy, vtheta
        self.trunk_delta_head = nn.Linear(hidden_dim, 4)      # joint delta: 4 joints
        self.left_arm_delta_head = nn.Linear(hidden_dim, 7)   # joint delta: 7 joints
        self.left_grip_delta_head = nn.Linear(hidden_dim, 1)  # gripper delta: 1 value
        self.right_arm_delta_head = nn.Linear(hidden_dim, 7)  # joint delta: 7 joints
        self.right_grip_delta_head = nn.Linear(hidden_dim, 1) # gripper delta: 1 value
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values for stable training.
        
        Delta predictions should start near zero.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Initialize output heads with even smaller weights
        # so initial deltas are near zero
        for head in [self.base_delta_head, self.trunk_delta_head, 
                     self.left_arm_delta_head, self.left_grip_delta_head,
                     self.right_arm_delta_head, self.right_grip_delta_head]:
            nn.init.xavier_uniform_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)
    
    def forward(
        self, 
        state: torch.Tensor, 
        target_eef_left: torch.Tensor, 
        target_eef_right: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass - outputs base velocity + delta joint angles.
        
        Args:
            state: [B, 256] current proprioception
            target_eef_left: [B, 7] target left EEF (pos + quat)
            target_eef_right: [B, 7] target right EEF (pos + quat)
            
        Returns:
            action: [B, 23] = base_vel(3) + joint_deltas(20)
        """
        # Concatenate inputs
        x = torch.cat([state, target_eef_left, target_eef_right], dim=-1)
        
        # Encode
        features = self.encoder(x)
        
        # Decode: base velocity + joint deltas
        base_vel = self.base_delta_head(features)  # Actually velocity, not delta
        trunk_delta = self.trunk_delta_head(features)
        left_arm_delta = self.left_arm_delta_head(features)
        left_grip_delta = self.left_grip_delta_head(features)
        right_arm_delta = self.right_arm_delta_head(features)
        right_grip_delta = self.right_grip_delta_head(features)
        
        # Concatenate in action order
        action_delta = torch.cat([
            base_vel,          # [0:3] - velocity, not a delta
            trunk_delta,       # [3:7]
            left_arm_delta,    # [7:14]
            left_grip_delta,   # [14:15]
            right_arm_delta,   # [15:22]
            right_grip_delta,  # [22:23]
        ], dim=-1)
        
        return action_delta
    
    def predict_absolute(
        self,
        state: torch.Tensor,
        target_eef_left: torch.Tensor,
        target_eef_right: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict absolute target joint positions (with base velocity unchanged).
        
        Convenience method that adds predicted deltas to current joint positions.
        Base velocity is returned as-is since it's already a velocity command.
        
        Returns:
            action_absolute: [B, 23] = base_vel(3) + absolute joint positions(20)
        """
        # Indices for extracting current joint positions from state
        TRUNK_QPOS = slice(236, 240)
        LEFT_ARM_QPOS = slice(158, 165)
        RIGHT_ARM_QPOS = slice(197, 204)
        LEFT_GRIP_QPOS = slice(193, 195)
        RIGHT_GRIP_QPOS = slice(232, 234)
        
        # Get predicted output (base_vel + deltas)
        output = self.forward(state, target_eef_left, target_eef_right)
        
        # Base velocity stays as-is (already velocity command)
        base_vel = output[:, 0:3]
        
        # Extract current positions from state
        current_trunk = state[:, TRUNK_QPOS]
        current_left_arm = state[:, LEFT_ARM_QPOS]
        current_left_grip = state[:, LEFT_GRIP_QPOS].sum(dim=-1, keepdim=True)
        current_right_arm = state[:, RIGHT_ARM_QPOS]
        current_right_grip = state[:, RIGHT_GRIP_QPOS].sum(dim=-1, keepdim=True)
        
        # Add deltas to get absolute positions
        abs_trunk = current_trunk + output[:, 3:7]
        abs_left_arm = current_left_arm + output[:, 7:14]
        abs_left_grip = current_left_grip + output[:, 14:15]
        abs_right_arm = current_right_arm + output[:, 15:22]
        abs_right_grip = current_right_grip + output[:, 22:23]
        
        return torch.cat([
            base_vel,         # velocity, unchanged
            abs_trunk,
            abs_left_arm,
            abs_left_grip,
            abs_right_arm,
            abs_right_grip,
        ], dim=-1)

    # =========================================================================
    # Inference Methods (for use with full 256-dim state from simulator)
    # =========================================================================
    
    def forward_from_full_state(
        self,
        full_state: torch.Tensor,
        target_eef_left: torch.Tensor,
        target_eef_right: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with automatic state filtering.
        
        Use this at inference time when receiving full 256-dim proprioception
        from the simulator. Automatically filters to 217-dim standard track state.
        
        Args:
            full_state: [B, 256] full proprioception from simulator
            target_eef_left: [B, 7] target EEF pose (body frame)
            target_eef_right: [B, 7] target EEF pose (body frame)
            
        Returns:
            action_delta: [B, 23] = base_vel(3) + joint_deltas(20)
        """
        filtered_state = filter_state_standard_track_torch(full_state)
        return self.forward(filtered_state, target_eef_left, target_eef_right)
    
    def predict_absolute_from_full_state(
        self,
        full_state: torch.Tensor,
        target_eef_left: torch.Tensor,
        target_eef_right: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict absolute joint positions with automatic state filtering.
        
        Use this at inference time when receiving full 256-dim proprioception.
        Filters state, predicts deltas, then adds to current positions.
        
        NOTE: Current joint positions are extracted from the FULL state (not filtered),
        since we need the actual values to add deltas to.
        
        Args:
            full_state: [B, 256] full proprioception from simulator
            target_eef_left: [B, 7] target EEF pose (body frame)
            target_eef_right: [B, 7] target EEF pose (body frame)
            
        Returns:
            action_absolute: [B, 23] = base_vel(3) + absolute joint positions(20)
        """
        # Indices in FULL 256-dim state for current joint positions
        TRUNK_QPOS_FULL = slice(236, 240)
        LEFT_ARM_QPOS_FULL = slice(158, 165)
        RIGHT_ARM_QPOS_FULL = slice(197, 204)
        LEFT_GRIP_QPOS_FULL = slice(193, 195)
        RIGHT_GRIP_QPOS_FULL = slice(232, 234)
        
        # Get predicted output (base_vel + deltas) using filtered state
        output = self.forward_from_full_state(full_state, target_eef_left, target_eef_right)
        
        # Base velocity stays as-is (already velocity command)
        base_vel = output[:, 0:3]
        
        # Extract current positions from FULL state
        current_trunk = full_state[:, TRUNK_QPOS_FULL]
        current_left_arm = full_state[:, LEFT_ARM_QPOS_FULL]
        current_left_grip = full_state[:, LEFT_GRIP_QPOS_FULL].sum(dim=-1, keepdim=True)
        current_right_arm = full_state[:, RIGHT_ARM_QPOS_FULL]
        current_right_grip = full_state[:, RIGHT_GRIP_QPOS_FULL].sum(dim=-1, keepdim=True)
        
        # Add deltas to get absolute positions
        abs_trunk = current_trunk + output[:, 3:7]
        abs_left_arm = current_left_arm + output[:, 7:14]
        abs_left_grip = current_left_grip + output[:, 14:15]
        abs_right_arm = current_right_arm + output[:, 15:22]
        abs_right_grip = current_right_grip + output[:, 22:23]
        
        return torch.cat([
            base_vel,         # velocity, unchanged
            abs_trunk,
            abs_left_arm,
            abs_left_grip,
            abs_right_arm,
            abs_right_grip,
        ], dim=-1)
