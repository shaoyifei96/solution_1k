# Whole-Body Pose Controller for R1Pro

## Overview

This is a data-driven whole-body controller that maps desired end-effector (EEF) poses to full robot actions. It was trained on BEHAVIOR-1K demonstrations and learns to:

- Predict base velocity (when/how to move the mobile base)
- Predict trunk joint positions
- Predict arm joint positions for both arms
- Predict gripper positions

**Key advantage:** Instead of using IK solvers with heuristics for base motion, this model implicitly learns from data when to move the base vs. just the arms.

---

## Model Details

| Property | Value |
|----------|-------|
| Architecture | 4-layer MLP with LayerNorm |
| Hidden dim | 512 |
| Parameters | 942,615 (~4 MB) |
| Input dim | 270 (state=256 + left_eef=7 + right_eef=7) |
| Output dim | 23 (action) |
| Inference time | < 1ms on CPU/GPU |
| GPU memory | < 10 MB |

---

## Checkpoint Location

Best model saved at:
```
/vast/projects/kumar/lab/yishao/b1k_2/outputs/pose_controller/run_20260118_162502/best_model.pt
```

The checkpoint contains:
- `model_state_dict`: PyTorch model weights
- `norm_stats`: Normalization statistics for inputs/outputs
- `config`: Model hyperparameters

---

## Input Format

### 1. Current State (256-dim vector)
The full R1Pro proprioception vector. Key indices:

| Component | Indices | Dims | Description |
|-----------|---------|------|-------------|
| Left arm qpos | 158:165 | 7 | Left arm joint positions |
| Right arm qpos | 197:204 | 7 | Right arm joint positions |
| Trunk qpos | 236:240 | 4 | Trunk joint positions |
| Base qpos | 244:247 | 3 | Base pose (x, y, theta) |
| Left EEF pos | 186:189 | 3 | Current left EEF position |
| Left EEF quat | 189:193 | 4 | Current left EEF quaternion |
| Right EEF pos | 225:228 | 3 | Current right EEF position |
| Right EEF quat | 228:232 | 4 | Current right EEF quaternion |

### 2. Target Left EEF Pose (7-dim)
```
[x, y, z, qx, qy, qz, qw]  # position + quaternion
```

### 3. Target Right EEF Pose (7-dim)
```
[x, y, z, qx, qy, qz, qw]  # position + quaternion
```

---

## Output Format

### Action (23-dim vector)

| Component | Indices | Dims | Type | Description |
|-----------|---------|------|------|-------------|
| Base velocity | 0:3 | 3 | Velocity | vx, vy, vtheta |
| Trunk | 3:7 | 4 | Position | 4 trunk joints |
| Left arm | 7:14 | 7 | Position | 7 arm joints |
| Left gripper | 14:15 | 1 | Position | Gripper opening |
| Right arm | 15:22 | 7 | Position | 7 arm joints |
| Right gripper | 22:23 | 1 | Position | Gripper opening |

**Note:** Base is velocity control, all other joints are position control.

---

## Usage Example (PyTorch)

```python
import torch
import torch.nn as nn
import numpy as np

# ============================================
# 1. Define the model architecture
# ============================================
class WholeBodyPoseController(nn.Module):
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
        
        input_dim = state_dim + eef_dim * 2  # 270
        
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
        
        self.base_head = nn.Linear(hidden_dim, 3)
        self.trunk_head = nn.Linear(hidden_dim, 4)
        self.left_arm_head = nn.Linear(hidden_dim, 7)
        self.left_grip_head = nn.Linear(hidden_dim, 1)
        self.right_arm_head = nn.Linear(hidden_dim, 7)
        self.right_grip_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, state, target_eef_left, target_eef_right):
        x = torch.cat([state, target_eef_left, target_eef_right], dim=-1)
        features = self.encoder(x)
        
        action = torch.cat([
            self.base_head(features),
            self.trunk_head(features),
            self.left_arm_head(features),
            self.left_grip_head(features),
            self.right_arm_head(features),
            self.right_grip_head(features),
        ], dim=-1)
        
        return action


# ============================================
# 2. Load the checkpoint
# ============================================
checkpoint_path = "/path/to/best_model.pt"
checkpoint = torch.load(checkpoint_path, map_location="cpu")

model = WholeBodyPoseController(
    hidden_dim=checkpoint["config"]["hidden_dim"],
    num_layers=checkpoint["config"]["num_layers"],
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Optional: move to GPU
# model = model.cuda()

# Get normalization stats
norm_stats = checkpoint.get("norm_stats", None)


# ============================================
# 3. Run inference
# ============================================
def get_action(
    model: WholeBodyPoseController,
    current_state: np.ndarray,      # [256]
    target_left_eef: np.ndarray,    # [7] = pos(3) + quat(4)
    target_right_eef: np.ndarray,   # [7] = pos(3) + quat(4)
    norm_stats: dict = None,
) -> np.ndarray:
    """
    Get action from pose controller.
    
    Args:
        model: Loaded WholeBodyPoseController
        current_state: Current proprioception (256-dim)
        target_left_eef: Desired left EEF pose [x,y,z,qx,qy,qz,qw]
        target_right_eef: Desired right EEF pose [x,y,z,qx,qy,qz,qw]
        norm_stats: Optional normalization stats from checkpoint
    
    Returns:
        action: 23-dim action vector
    """
    with torch.no_grad():
        # Convert to tensors
        state = torch.from_numpy(current_state).float().unsqueeze(0)
        left = torch.from_numpy(target_left_eef).float().unsqueeze(0)
        right = torch.from_numpy(target_right_eef).float().unsqueeze(0)
        
        # Normalize inputs if stats available
        if norm_stats is not None:
            state = (state - norm_stats["state_mean"]) / norm_stats["state_std"]
            left = (left - norm_stats["eef_mean"]) / norm_stats["eef_std"]
            right = (right - norm_stats["eef_mean"]) / norm_stats["eef_std"]
        
        # Forward pass
        action = model(state, left, right)
        
        # Denormalize output if stats available
        if norm_stats is not None:
            action = action * norm_stats["action_std"] + norm_stats["action_mean"]
        
        return action.squeeze(0).numpy()


# ============================================
# 4. Example: Move left hand to target
# ============================================
# Get current state from robot/simulation
current_state = np.zeros(256)  # Replace with actual state

# Define target EEF poses
target_left = np.array([0.5, 0.2, 0.8, 0, 0, 0, 1])   # [x,y,z, qx,qy,qz,qw]
target_right = np.array([0.5, -0.2, 0.8, 0, 0, 0, 1]) # Keep right arm still

# Get action
action = get_action(model, current_state, target_left, target_right, norm_stats)

# Extract components
base_vel = action[0:3]      # [vx, vy, vtheta]
trunk_pos = action[3:7]     # 4 joints
left_arm = action[7:14]     # 7 joints
left_grip = action[14:15]   # 1 joint
right_arm = action[15:22]   # 7 joints
right_grip = action[22:23]  # 1 joint

# Apply to robot
# robot.set_base_velocity(base_vel)
# robot.set_joint_positions(trunk=trunk_pos, left_arm=left_arm, right_arm=right_arm, ...)
```

---

## Integration with BEHAVIOR-1K Evaluation

In the evaluation loop, you can use this to replace IK-based pose control:

```python
from b1k.pose_controller.inference import PoseControllerInference

# Load controller
pose_ctrl = PoseControllerInference.load(
    "/path/to/best_model.pt",
    device="cuda"  # or "cpu"
)

# In the control loop:
def step(obs, target_left_eef, target_right_eef):
    # Get current proprioception from observation
    current_state = obs["state"]  # [256]
    
    # Get action from pose controller
    action = pose_ctrl.get_action(
        current_state,
        target_left_eef,
        target_right_eef
    )
    
    return action
```

---

## Training Details

- **Dataset:** BEHAVIOR-1K demonstrations (50 tasks × 200 episodes)
- **Samples:** ~119 million state-action pairs
- **Lookahead:** 15 steps (0.5 sec at 30Hz)
- **Best epoch:** 1 (val_loss=0.2263)
- **Validation MAE:**
  - Base: 0.130 m/s
  - Trunk: 0.031 rad
  - Left arm: 0.119 rad
  - Right arm: 0.084 rad

---

## Notes

1. **Target poses should be in robot base frame** (same coordinate system as the state vector EEF poses)

2. **Quaternion format:** [qx, qy, qz, qw] (scalar-last convention)

3. **Lookahead:** The model was trained with 15-step lookahead, meaning targets should be ~0.5 seconds ahead of current position for smooth motion

4. **Gripper control:** The model predicts gripper positions but was not specifically optimized for grasp timing. You may want to use a separate gripper controller for precise manipulation.

5. **Base motion:** The model learns when to move the base from data. For targets within arm reach, it may output near-zero base velocity.

---

## Troubleshooting

**Q: Actions look jerky/unstable**
- Ensure target poses are in the correct coordinate frame
- Low-pass filter the outputs if needed
- Check that quaternions are normalized

**Q: Base moves when it shouldn't**
- The model learned base motion patterns from demonstrations
- For fine manipulation, you may want to mask base velocity to zero

**Q: Arm doesn't reach target**
- The model is trained on demonstrated trajectories, not arbitrary targets
- Targets far outside the training distribution may not work well
- Consider using multiple steps to reach distant targets

---

## Files

```
b1k_2/src/b1k/pose_controller/
├── model.py           # PyTorch model definition
├── model_jax.py       # JAX/Flax model definition
├── dataset.py         # PyTorch dataset
├── dataset_jax.py     # NumPy dataset for JAX
├── inference.py       # Inference wrapper class

b1k_2/scripts/
├── train_pose_controller.py      # PyTorch training
├── train_pose_controller_jax.py  # JAX training

b1k_2/outputs/pose_controller/
└── run_20260118_162502/
    ├── best_model.pt    # Best validation checkpoint
    └── final_model.pt   # Final epoch checkpoint
```
