# Whole-Body Pose Controller for R1Pro (Delta Mode)

## Overview

This is a data-driven whole-body controller that maps desired end-effector (EEF) poses to **delta joint angles**. It was trained on BEHAVIOR-1K demonstrations and learns to:

- Predict base velocity (when/how to move the mobile base)
- Predict **delta** trunk joint positions (change from current)
- Predict **delta** arm joint positions for both arms
- Predict **delta** gripper positions

**Key advantages:**
1. Instead of using IK solvers with heuristics for base motion, this model implicitly learns from data when to move the base vs. just the arms.
2. **Delta mode** predicts small changes from current pose, making outputs more stable and interpretable.
3. **Filtered 172-dim state** excludes noisy/correlated signals (effort, robot velocities, gripper states).

---

## Model Details

| Property | Value |
|----------|-------|
| Architecture | 4-layer MLP with LayerNorm |
| Hidden dim | 512 |
| Parameters | 899,607 (~3.6 MB) |
| Input dim | 186 (state=172 + left_eef=7 + right_eef=7) |
| Output dim | 23 (base_vel + joint_deltas) |
| State dim | 172 (filtered from 256, excludes effort/gripper/velocities) |
| Inference time | < 1ms on CPU/GPU |
| GPU memory | < 10 MB |

---

## Training Results

**Latest Run:** `run_20260126_153106`

| Metric | Value |
|--------|-------|
| Training samples | 5,544,658 |
| Validation samples | 616,073 |
| Dataset | 50 tasks × 10 episodes each |
| Lookahead | 1 step |
| Epochs | 10 |
| Best val loss | 2.5645 |
| Val MAE - Base | 0.173 |
| Val MAE - Trunk | 0.201 |
| Val MAE - L-Arm | 0.210 |
| Val MAE - R-Arm | 0.215 |

---

## Checkpoint Location

Best model saved at:
```
/vast/projects/kumar/lab/yishao/b1k_2/outputs/pose_controller/run_20260126_153106/best_model.pt
```

The checkpoint contains:
- `model_state_dict`: PyTorch model weights
- `epoch`, `val_loss`: Training metadata

Normalization stats saved separately:
```
/vast/projects/kumar/lab/yishao/b1k_2/outputs/pose_controller/run_20260126_153106/norm_stats.json
```

---

## Input Format

### 1. Current State (172-dim filtered vector)

The model uses a **filtered 172-dim state** from the full 256-dim proprioception. 

**Included components:**
| Component | Original Indices | Dims | Description |
|-----------|-----------------|------|-------------|
| joint_qpos | 6:28 | 22 | Joint positions (excl. base) |
| joint_qpos_sin | 34:56 | 22 | Joint position sin |
| joint_qpos_cos | 62:84 | 22 | Joint position cos |
| joint_qvel | 84:112 | 28 | Joint velocities |
| arm_left_qpos | 158:165 | 7 | Left arm joint positions |
| arm_left_qpos_sin | 165:172 | 7 | Left arm sin |
| arm_left_qpos_cos | 172:179 | 7 | Left arm cos |
| arm_left_qvel | 179:186 | 7 | Left arm velocities |
| eef_left_pos | 186:189 | 3 | Left EEF position (body frame) |
| eef_left_quat | 189:193 | 4 | Left EEF quaternion |
| arm_right_qpos | 197:204 | 7 | Right arm joint positions |
| arm_right_qpos_sin | 204:211 | 7 | Right arm sin |
| arm_right_qpos_cos | 211:218 | 7 | Right arm cos |
| arm_right_qvel | 218:225 | 7 | Right arm velocities |
| eef_right_pos | 225:228 | 3 | Right EEF position (body frame) |
| eef_right_quat | 228:232 | 4 | Right EEF quaternion |
| trunk_qpos | 236:240 | 4 | Trunk joint positions |
| trunk_qvel | 240:244 | 4 | Trunk joint velocities |

**Excluded (for stability):**
- `joint_qeffort` (28 dims) - too noisy
- `robot_lin_vel` (3 dims) - correlated to command
- `robot_ang_vel` (3 dims) - correlated to command
- `gripper_left_qpos/qvel` (4 dims) - not needed for EEF pose control
- `gripper_right_qpos/qvel` (4 dims) - not needed for EEF pose control
- `base_qvel` (3 dims) - correlated to command
- Global position/orientation (not allowed in standard track)

### 2. Target Left EEF Pose (7-dim)
```
[x, y, z, qx, qy, qz, qw]  # position + quaternion (body frame)
```

### 3. Target Right EEF Pose (7-dim)
```
[x, y, z, qx, qy, qz, qw]  # position + quaternion (body frame)
```

---

## Output Format (Delta Mode)

### Action Delta (23-dim vector)

| Component | Indices | Dims | Type | Description |
|-----------|---------|------|------|-------------|
| Base velocity | 0:3 | 3 | Velocity | vx, vy, vtheta (unchanged, already velocity) |
| Trunk delta | 3:7 | 4 | Delta | Δ for 4 trunk joints |
| Left arm delta | 7:14 | 7 | Delta | Δ for 7 arm joints |
| Left gripper delta | 14:15 | 1 | Delta | Δ for gripper |
| Right arm delta | 15:22 | 7 | Delta | Δ for 7 arm joints |
| Right gripper delta | 22:23 | 1 | Delta | Δ for gripper |

**To get absolute positions:** `target_joints = current_joints + delta`

---

## Usage Example (PyTorch)

```python
import torch
import json
import numpy as np

# Import the model
from b1k.pose_controller.model import (
    WholeBodyPoseControllerDelta,
    filter_state_standard_track_torch
)

# ============================================
# 1. Load the checkpoint
# ============================================
checkpoint_path = "/vast/projects/kumar/lab/yishao/b1k_2/outputs/pose_controller/run_20260126_153106/best_model.pt"
norm_stats_path = "/vast/projects/kumar/lab/yishao/b1k_2/outputs/pose_controller/run_20260126_153106/norm_stats.json"

checkpoint = torch.load(checkpoint_path, map_location="cpu")

model = WholeBodyPoseControllerDelta(
    state_dim=172,  # Filtered state
    hidden_dim=512,
    num_layers=4,
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Load normalization stats
with open(norm_stats_path) as f:
    norm_stats = json.load(f)

# Convert to tensors
state_mean = torch.tensor(norm_stats["state_mean"])
state_std = torch.tensor(norm_stats["state_std"])
eef_left_mean = torch.tensor(norm_stats["eef_left_mean"])
eef_left_std = torch.tensor(norm_stats["eef_left_std"])
eef_right_mean = torch.tensor(norm_stats["eef_right_mean"])
eef_right_std = torch.tensor(norm_stats["eef_right_std"])
action_delta_mean = torch.tensor(norm_stats["action_delta_mean"])
action_delta_std = torch.tensor(norm_stats["action_delta_std"])


# ============================================
# 2. Inference with full 256-dim state
# ============================================
def get_action_delta(
    model,
    full_state: np.ndarray,         # [256] from simulator
    target_left_eef: np.ndarray,    # [7] = pos(3) + quat(4)
    target_right_eef: np.ndarray,   # [7] = pos(3) + quat(4)
) -> np.ndarray:
    """
    Get delta action from pose controller.
    
    Args:
        model: Loaded WholeBodyPoseControllerDelta
        full_state: Full 256-dim proprioception from simulator
        target_left_eef: Desired left EEF pose [x,y,z,qx,qy,qz,qw]
        target_right_eef: Desired right EEF pose [x,y,z,qx,qy,qz,qw]
    
    Returns:
        action_delta: 23-dim action = base_vel(3) + joint_deltas(20)
    """
    with torch.no_grad():
        # Convert to tensors
        state_full = torch.from_numpy(full_state).float().unsqueeze(0)
        left = torch.from_numpy(target_left_eef).float().unsqueeze(0)
        right = torch.from_numpy(target_right_eef).float().unsqueeze(0)
        
        # Filter state (256 -> 172)
        state = filter_state_standard_track_torch(state_full)
        
        # Normalize inputs
        state = (state - state_mean) / state_std
        left = (left - eef_left_mean) / eef_left_std
        right = (right - eef_right_mean) / eef_right_std
        
        # Forward pass
        action_delta = model(state, left, right)
        
        # Denormalize output
        action_delta = action_delta * action_delta_std + action_delta_mean
        
        return action_delta.squeeze(0).numpy()


# ============================================
# 3. Convert delta to absolute positions
# ============================================
def delta_to_absolute(
    full_state: np.ndarray,  # [256]
    action_delta: np.ndarray,  # [23]
) -> np.ndarray:
    """
    Convert delta action to absolute joint positions.
    
    Base velocity is unchanged (already velocity command).
    Joint positions = current + delta.
    """
    # Indices in full 256-dim state
    TRUNK_QPOS = slice(236, 240)
    LEFT_ARM_QPOS = slice(158, 165)
    RIGHT_ARM_QPOS = slice(197, 204)
    LEFT_GRIP_QPOS = slice(193, 195)
    RIGHT_GRIP_QPOS = slice(232, 234)
    
    # Current positions
    current_trunk = full_state[TRUNK_QPOS]
    current_left_arm = full_state[LEFT_ARM_QPOS]
    current_right_arm = full_state[RIGHT_ARM_QPOS]
    current_left_grip = full_state[LEFT_GRIP_QPOS].sum()
    current_right_grip = full_state[RIGHT_GRIP_QPOS].sum()
    
    # Add deltas
    action_abs = np.zeros(23)
    action_abs[0:3] = action_delta[0:3]  # Base velocity unchanged
    action_abs[3:7] = current_trunk + action_delta[3:7]
    action_abs[7:14] = current_left_arm + action_delta[7:14]
    action_abs[14:15] = current_left_grip + action_delta[14:15]
    action_abs[15:22] = current_right_arm + action_delta[15:22]
    action_abs[22:23] = current_right_grip + action_delta[22:23]
    
    return action_abs


# ============================================
# 4. Example usage
# ============================================
# Get current state from robot/simulation
current_state = np.zeros(256)  # Replace with actual state

# Define target EEF poses (body frame)
target_left = np.array([0.5, 0.2, 0.8, 0, 0, 0, 1])   # [x,y,z, qx,qy,qz,qw]
target_right = np.array([0.5, -0.2, 0.8, 0, 0, 0, 1])

# Get delta action
action_delta = get_action_delta(model, current_state, target_left, target_right)

# Convert to absolute (optional)
action_absolute = delta_to_absolute(current_state, action_delta)

# Extract components
base_vel = action_absolute[0:3]      # [vx, vy, vtheta]
trunk_pos = action_absolute[3:7]     # 4 joints
left_arm = action_absolute[7:14]     # 7 joints
left_grip = action_absolute[14:15]   # 1 joint
right_arm = action_absolute[15:22]   # 7 joints
right_grip = action_absolute[22:23]  # 1 joint
```

---

## Using the Model's Built-in Methods

The model also has convenience methods for inference:

```python
# Forward with automatic state filtering (256 -> 172)
action_delta = model.forward_from_full_state(full_state, target_left, target_right)

# Get absolute positions directly
action_absolute = model.predict_absolute_from_full_state(full_state, target_left, target_right)
```

---

## Training Command

```bash
uv run scripts/train_pose_controller.py \
    --data_root /path/to/b1k_full \
    --output_dir outputs/pose_controller \
    --lookahead 1 \
    --batch_size 512 \
    --epochs 10 \
    --lr 1e-4 \
    --hidden_dim 512 \
    --num_layers 4 \
    --use_delta \
    --max_episodes 10
```

---

## Files

```
b1k_2/src/b1k/pose_controller/
├── model.py           # PyTorch model (WholeBodyPoseControllerDelta)
├── dataset.py         # Dataset with delta computation and state filtering

b1k_2/scripts/
├── train_pose_controller.py      # Training script
├── visualize_state_inputs.py     # State visualization

b1k_2/outputs/pose_controller/
└── run_20260126_153106/
    ├── best_model.pt       # Best validation checkpoint
    ├── final_model.pt      # Final epoch checkpoint
    ├── config.json         # Training config
    └── norm_stats.json     # Normalization statistics
```

---

## Notes

1. **EEF poses are in body frame** (relative to robot base, not global)

2. **Quaternion format:** [qx, qy, qz, qw] (scalar-last convention)

3. **Lookahead=1:** This model uses 1-step lookahead, meaning targets are very close to current pose

4. **Delta output:** The model predicts small deltas. For large motions, call multiple times iteratively.

5. **State filtering happens automatically** when using `forward_from_full_state()` or `predict_absolute_from_full_state()`
