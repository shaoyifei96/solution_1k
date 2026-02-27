"""Whole-body pose controller network in JAX/Flax.

Maps (current_state, target_eef_left, target_eef_right) → action[23]
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp


class WholeBodyPoseController(nnx.Module):
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
        rngs: nnx.Rngs,
        state_dim: int = 256, 
        eef_dim: int = 7, 
        action_dim: int = 23, 
        hidden_dim: int = 512,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        self.state_dim = state_dim
        self.eef_dim = eef_dim
        self.action_dim = action_dim
        
        input_dim = state_dim + eef_dim * 2  # 270
        
        # Build encoder layers
        self.encoder_layers = []
        self.encoder_norms = []
        self.encoder_dropouts = []
        
        in_dim = input_dim
        for i in range(num_layers):
            self.encoder_layers.append(nnx.Linear(in_dim, hidden_dim, rngs=rngs))
            self.encoder_norms.append(nnx.LayerNorm(hidden_dim, rngs=rngs))
            if i < num_layers - 1:
                self.encoder_dropouts.append(nnx.Dropout(dropout, rngs=rngs))
            else:
                self.encoder_dropouts.append(None)
            in_dim = hidden_dim
        
        # Separate heads for different action components
        self.base_head = nnx.Linear(hidden_dim, 3, rngs=rngs)       # velocity: vx, vy, vtheta
        self.trunk_head = nnx.Linear(hidden_dim, 4, rngs=rngs)      # position: 4 joints
        self.left_arm_head = nnx.Linear(hidden_dim, 7, rngs=rngs)   # position: 7 joints
        self.left_grip_head = nnx.Linear(hidden_dim, 1, rngs=rngs)  # position: 1 joint
        self.right_arm_head = nnx.Linear(hidden_dim, 7, rngs=rngs)  # position: 7 joints
        self.right_grip_head = nnx.Linear(hidden_dim, 1, rngs=rngs) # position: 1 joint
    
    def __call__(
        self, 
        state: jnp.ndarray, 
        target_eef_left: jnp.ndarray, 
        target_eef_right: jnp.ndarray,
        training: bool = True,
    ) -> jnp.ndarray:
        """
        Forward pass.
        
        Args:
            state: [B, 256] current proprioception
            target_eef_left: [B, 7] target left EEF (pos + quat)
            target_eef_right: [B, 7] target right EEF (pos + quat)
            training: whether in training mode (affects dropout)
            
        Returns:
            action: [B, 23] full action vector
        """
        # Concatenate inputs
        x = jnp.concatenate([state, target_eef_left, target_eef_right], axis=-1)
        
        # Encode
        for layer, norm, dropout in zip(self.encoder_layers, self.encoder_norms, self.encoder_dropouts):
            x = layer(x)
            x = norm(x)
            x = nnx.relu(x)
            if dropout is not None:
                x = dropout(x, deterministic=not training)
        
        features = x
        
        # Decode to action components
        base_vel = self.base_head(features)
        trunk_pos = self.trunk_head(features)
        left_arm_pos = self.left_arm_head(features)
        left_grip_pos = self.left_grip_head(features)
        right_arm_pos = self.right_arm_head(features)
        right_grip_pos = self.right_grip_head(features)
        
        # Concatenate in action order
        action = jnp.concatenate([
            base_vel,       # [0:3]
            trunk_pos,      # [3:7]
            left_arm_pos,   # [7:14]
            left_grip_pos,  # [14:15]
            right_arm_pos,  # [15:22]
            right_grip_pos, # [22:23]
        ], axis=-1)
        
        return action
    
    def forward_with_components(
        self,
        state: jnp.ndarray,
        target_eef_left: jnp.ndarray,
        target_eef_right: jnp.ndarray,
        training: bool = True,
    ) -> dict[str, jnp.ndarray]:
        """Forward pass returning action components separately."""
        x = jnp.concatenate([state, target_eef_left, target_eef_right], axis=-1)
        
        for layer, norm, dropout in zip(self.encoder_layers, self.encoder_norms, self.encoder_dropouts):
            x = layer(x)
            x = norm(x)
            x = nnx.relu(x)
            if dropout is not None:
                x = dropout(x, deterministic=not training)
        
        features = x
        
        return {
            "base_vel": self.base_head(features),
            "trunk_pos": self.trunk_head(features),
            "left_arm_pos": self.left_arm_head(features),
            "left_grip_pos": self.left_grip_head(features),
            "right_arm_pos": self.right_arm_head(features),
            "right_grip_pos": self.right_grip_head(features),
        }


def count_parameters(model: nnx.Module) -> int:
    """Count the number of trainable parameters."""
    params = nnx.state(model, nnx.Param)
    total = 0
    for p in jax.tree_util.tree_leaves(params):
        total += p.size
    return total
