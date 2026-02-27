#!/usr/bin/env python3
"""
Visualize the 217-dim filtered state inputs that the pose controller network sees.

Takes 1 episode from each task, combines with different colors, and creates:
- 3D plots for position/velocity data (3-dim)
- Subplots for higher-dim data (joint positions, velocities, etc.)

Usage:
    conda activate b1k_solution
    python visualize_state_inputs.py --data_dir /path/to/data --output_dir ./viz_outputs
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# =============================================================================
# State Component Definitions (matching the 172-dim filtered state)
# =============================================================================
# These are the indices in the ORIGINAL 256-dim state that get extracted
# EXCLUDED: joint_qeffort, robot_lin/ang_vel, gripper_qpos/qvel, base_qvel
ALLOWED_PROPRIO_SLICES = [
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
    # EXCLUDED: gripper_left_qpos (193:195) - not needed
    # EXCLUDED: gripper_left_qvel (195:197) - not needed
    slice(197, 204),  # arm_right_qpos (7)
    slice(204, 211),  # arm_right_qpos_sin (7)
    slice(211, 218),  # arm_right_qpos_cos (7)
    slice(218, 225),  # arm_right_qvel (7)
    slice(225, 228),  # eef_right_pos (3)
    slice(228, 232),  # eef_right_quat (4)
    # EXCLUDED: gripper_right_qpos (232:234) - not needed
    # EXCLUDED: gripper_right_qvel (234:236) - not needed
    slice(236, 240),  # trunk_qpos (4)
    slice(240, 244),  # trunk_qvel (4)
    # EXCLUDED: base_qvel (253:256) - correlated to command
]

# Component names and their dimensions in the filtered 172-dim state
# (name, start_idx_in_filtered, dim)
STATE_COMPONENTS = [
    ("joint_qpos", 0, 22),
    ("joint_qpos_sin", 22, 22),
    ("joint_qpos_cos", 44, 22),
    ("joint_qvel", 66, 28),
    # After joint_qvel at 66+28=94, next is arm_left_qpos
    ("arm_left_qpos", 94, 7),
    ("arm_left_qpos_sin", 101, 7),
    ("arm_left_qpos_cos", 108, 7),
    ("arm_left_qvel", 115, 7),
    ("eef_left_pos", 122, 3),
    ("eef_left_quat", 125, 4),
    ("arm_right_qpos", 129, 7),
    ("arm_right_qpos_sin", 136, 7),
    ("arm_right_qpos_cos", 143, 7),
    ("arm_right_qvel", 150, 7),
    ("eef_right_pos", 157, 3),
    ("eef_right_quat", 160, 4),
    ("trunk_qpos", 164, 4),
    ("trunk_qvel", 168, 4),
]


def filter_state_standard_track(state: np.ndarray) -> np.ndarray:
    """Filter 256-dim state to 172-dim filtered state."""
    parts = [state[..., s] for s in ALLOWED_PROPRIO_SLICES]
    return np.concatenate(parts, axis=-1)


def load_one_episode_per_task(data_dir: str, max_tasks: int = 50) -> Dict[str, np.ndarray]:
    """
    Load one episode from each task.
    
    Returns:
        Dictionary mapping task name -> filtered state array [T, 172]
    """
    data_path = Path(data_dir)
    task_dirs = sorted([d for d in data_path.iterdir() if d.is_dir() and d.name.startswith("task-")])
    
    if max_tasks:
        task_dirs = task_dirs[:max_tasks]
    
    task_data = {}
    
    for task_dir in task_dirs:
        parquet_files = sorted(task_dir.glob("*.parquet"))
        if not parquet_files:
            print(f"  Skipping {task_dir.name}: no parquet files")
            continue
        
        # Load first episode
        first_file = parquet_files[0]
        df = pd.read_parquet(first_file)
        
        # Extract state
        state_col = "observation.state"
        if state_col not in df.columns:
            print(f"  Skipping {task_dir.name}: no {state_col} column")
            continue
        
        states = np.stack(df[state_col].values)  # [T, 256]
        filtered_states = filter_state_standard_track(states)  # [T, 217]
        
        task_data[task_dir.name] = filtered_states
        print(f"  Loaded {task_dir.name}: {len(filtered_states)} timesteps")
    
    return task_data


def get_component(data: np.ndarray, name: str) -> np.ndarray:
    """Extract a named component from the filtered 217-dim state."""
    for comp_name, start, dim in STATE_COMPONENTS:
        if comp_name == name:
            return data[:, start:start+dim]
    raise ValueError(f"Unknown component: {name}")


def get_colors(n: int) -> List[Tuple[float, float, float, float]]:
    """Get n distinct colors."""
    cmap = plt.cm.get_cmap('tab20' if n <= 20 else 'hsv')
    return [cmap(i / n) for i in range(n)]


def plot_3d_component(
    task_data: Dict[str, np.ndarray],
    component_name: str,
    output_dir: str,
    labels: List[str] = None,
):
    """Plot a 3-dim component in 3D space with different colors per task."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = get_colors(len(task_data))
    
    for (task_name, data), color in zip(task_data.items(), colors):
        comp = get_component(data, component_name)
        if comp.shape[1] != 3:
            print(f"  Warning: {component_name} has {comp.shape[1]} dims, expected 3")
            continue
        ax.scatter(comp[:, 0], comp[:, 1], comp[:, 2], 
                  c=[color], alpha=0.5, s=5, label=task_name)
    
    if labels is None:
        labels = ["X", "Y", "Z"]
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    ax.set_title(f"{component_name} (3D)")
    
    # Legend outside
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=6, ncol=2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{component_name}_3d.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {component_name}_3d.png")


def plot_subplots_component(
    task_data: Dict[str, np.ndarray],
    component_name: str,
    output_dir: str,
    joint_names: List[str] = None,
):
    """Plot each dimension of a component as a separate subplot (time series)."""
    # Get dimension
    for comp_name, start, dim in STATE_COMPONENTS:
        if comp_name == component_name:
            n_dims = dim
            break
    else:
        raise ValueError(f"Unknown component: {component_name}")
    
    # Determine subplot layout
    n_cols = min(4, n_dims)
    n_rows = (n_dims + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    if n_dims == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    colors = get_colors(len(task_data))
    
    for d in range(n_dims):
        row, col = d // n_cols, d % n_cols
        ax = axes[row, col]
        
        for (task_name, data), color in zip(task_data.items(), colors):
            comp = get_component(data, component_name)
            # Normalize x-axis to 0-100 (percentage of episode)
            t = np.linspace(0, 100, len(comp))
            ax.plot(t, comp[:, d], c=color, alpha=0.6, linewidth=0.8)
        
        if joint_names and d < len(joint_names):
            ax.set_title(f"{joint_names[d]}", fontsize=10)
        else:
            ax.set_title(f"dim {d}", fontsize=10)
        ax.set_xlabel("Episode %")
        ax.set_xlim(0, 100)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for d in range(n_dims, n_rows * n_cols):
        row, col = d // n_cols, d % n_cols
        axes[row, col].set_visible(False)
    
    fig.suptitle(f"{component_name} ({n_dims} dims)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{component_name}_subplots.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {component_name}_subplots.png")


def plot_4d_quat_component(
    task_data: Dict[str, np.ndarray],
    component_name: str,
    output_dir: str,
):
    """Plot quaternion (4D) as separate w,x,y,z subplots."""
    plot_subplots_component(task_data, component_name, output_dir, 
                           joint_names=["w", "x", "y", "z"])


def plot_2d_component(
    task_data: Dict[str, np.ndarray],
    component_name: str,
    output_dir: str,
    labels: List[str] = None,
):
    """Plot a 2-dim component as scatter plot."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = get_colors(len(task_data))
    
    for (task_name, data), color in zip(task_data.items(), colors):
        comp = get_component(data, component_name)
        ax.scatter(comp[:, 0], comp[:, 1], c=[color], alpha=0.5, s=5, label=task_name)
    
    if labels is None:
        labels = ["dim 0", "dim 1"]
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title(f"{component_name} (2D)")
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{component_name}_2d.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {component_name}_2d.png")


# Joint names for R1Pro robot
JOINT_NAMES_22 = [
    "trunk_0", "trunk_1", "trunk_2", "trunk_3",
    "left_arm_0", "left_arm_1", "left_arm_2", "left_arm_3", "left_arm_4", "left_arm_5", "left_arm_6",
    "right_arm_0", "right_arm_1", "right_arm_2", "right_arm_3", "right_arm_4", "right_arm_5", "right_arm_6",
    "left_gripper", "left_gripper_mimic",
    "right_gripper", "right_gripper_mimic",
]

JOINT_NAMES_28 = [
    "base_x", "base_y", "base_rz",
    "trunk_0", "trunk_1", "trunk_2", "trunk_3",
    "left_arm_0", "left_arm_1", "left_arm_2", "left_arm_3", "left_arm_4", "left_arm_5", "left_arm_6",
    "right_arm_0", "right_arm_1", "right_arm_2", "right_arm_3", "right_arm_4", "right_arm_5", "right_arm_6",
    "left_gripper", "left_gripper_mimic",
    "right_gripper", "right_gripper_mimic",
]

ARM_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3", "wrist_4"]


def main():
    parser = argparse.ArgumentParser(description="Visualize pose controller state inputs")
    parser.add_argument("--data_dir", type=str, 
                       default="/vast/projects/kumar/lab/yishao/data/b1k_full/data",
                       help="Path to data directory")
    parser.add_argument("--output_dir", type=str,
                       default="/vast/projects/kumar/lab/yishao/b1k_2/viz_outputs",
                       help="Output directory for visualizations")
    parser.add_argument("--max_tasks", type=int, default=50,
                       help="Maximum number of tasks to load")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading data from {args.data_dir}")
    task_data = load_one_episode_per_task(args.data_dir, args.max_tasks)
    print(f"Loaded {len(task_data)} tasks\n")
    
    print("Generating visualizations...")
    
    # ==========================================================================
    # 3D Plots (for 3-dim components)
    # ==========================================================================
    print("\n--- 3D Position Plots ---")
    
    # EEF left position (body frame, 3D)
    plot_3d_component(task_data, "eef_left_pos", args.output_dir,
                     labels=["x", "y", "z"])
    
    # EEF right position (body frame, 3D)
    plot_3d_component(task_data, "eef_right_pos", args.output_dir,
                     labels=["x", "y", "z"])
    
    # ==========================================================================
    # Quaternion Plots (4-dim as subplots)
    # ==========================================================================
    print("\n--- Quaternion Plots (4D) ---")
    
    plot_4d_quat_component(task_data, "eef_left_quat", args.output_dir)
    plot_4d_quat_component(task_data, "eef_right_quat", args.output_dir)
    
    # ==========================================================================
    # Subplot Plots (high-dim joint data)
    # ==========================================================================
    print("\n--- High-Dimensional Joint Plots ---")
    
    # Joint positions (22-dim)
    plot_subplots_component(task_data, "joint_qpos", args.output_dir, 
                           joint_names=JOINT_NAMES_22)
    plot_subplots_component(task_data, "joint_qpos_sin", args.output_dir,
                           joint_names=JOINT_NAMES_22)
    plot_subplots_component(task_data, "joint_qpos_cos", args.output_dir,
                           joint_names=JOINT_NAMES_22)
    
    # Joint velocities (28-dim)
    plot_subplots_component(task_data, "joint_qvel", args.output_dir,
                           joint_names=JOINT_NAMES_28)
    
    # Arm-specific plots (7-dim each)
    print("\n--- Arm-Specific Plots ---")
    
    plot_subplots_component(task_data, "arm_left_qpos", args.output_dir,
                           joint_names=ARM_JOINT_NAMES)
    plot_subplots_component(task_data, "arm_left_qpos_sin", args.output_dir,
                           joint_names=ARM_JOINT_NAMES)
    plot_subplots_component(task_data, "arm_left_qpos_cos", args.output_dir,
                           joint_names=ARM_JOINT_NAMES)
    plot_subplots_component(task_data, "arm_left_qvel", args.output_dir,
                           joint_names=ARM_JOINT_NAMES)
    
    plot_subplots_component(task_data, "arm_right_qpos", args.output_dir,
                           joint_names=ARM_JOINT_NAMES)
    plot_subplots_component(task_data, "arm_right_qpos_sin", args.output_dir,
                           joint_names=ARM_JOINT_NAMES)
    plot_subplots_component(task_data, "arm_right_qpos_cos", args.output_dir,
                           joint_names=ARM_JOINT_NAMES)
    plot_subplots_component(task_data, "arm_right_qvel", args.output_dir,
                           joint_names=ARM_JOINT_NAMES)
    
    # Trunk plots (4-dim)
    print("\n--- Trunk Plots ---")
    
    plot_subplots_component(task_data, "trunk_qpos", args.output_dir,
                           joint_names=["trunk_0", "trunk_1", "trunk_2", "trunk_3"])
    plot_subplots_component(task_data, "trunk_qvel", args.output_dir,
                           joint_names=["trunk_0_vel", "trunk_1_vel", "trunk_2_vel", "trunk_3_vel"])
    
    print(f"\n=== Done! All visualizations saved to {args.output_dir} ===")
    print(f"Total: {len(os.listdir(args.output_dir))} files generated")


if __name__ == "__main__":
    main()
