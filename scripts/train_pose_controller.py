#!/usr/bin/env python3
"""Training script for whole-body pose controller.

Usage:
    python scripts/train_pose_controller.py \
        --data_root /path/to/b1k_full \
        --output_dir outputs/pose_controller \
        --lookahead 15 \
        --batch_size 256 \
        --epochs 100
"""

import argparse
import logging
import os
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from b1k.pose_controller.model import WholeBodyPoseController, WholeBodyPoseControllerWithResidual, WholeBodyPoseControllerDelta
from b1k.pose_controller.dataset import (
    PoseControlDataset, PoseControlDatasetLazy, PoseControlDatasetDelta, 
    ActionIndices, ALLOWED_STATE_DIM, filter_state_standard_track
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train whole-body pose controller")
    
    # Data
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to BEHAVIOR-1K dataset root")
    parser.add_argument("--lookahead", type=int, default=15,
                        help="Lookahead steps for target EEF pose (default: 15 = 0.5s)")
    parser.add_argument("--max_tasks", type=int, default=None,
                        help="Max tasks to load (for debugging)")
    parser.add_argument("--max_episodes", type=int, default=None,
                        help="Max episodes per task (for debugging)")
    parser.add_argument("--lazy_load", action="store_true",
                        help="Use lazy loading (faster init, slower training)")
    parser.add_argument("--fast", action="store_true",
                        help="Quick test: load only 2 tasks, 5 episodes each")
    parser.add_argument("--no_standard_track", action="store_true",
                        help="Use full 256-dim state (includes disallowed global info). "
                             "Default is standard track mode (217-dim filtered state).")
    
    # Model
    parser.add_argument("--hidden_dim", type=int, default=512,
                        help="Hidden dimension")
    parser.add_argument("--num_layers", type=int, default=4,
                        help="Number of MLP layers")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    parser.add_argument("--use_residual", action="store_true",
                        help="Use residual prediction for joint positions")
    parser.add_argument("--use_delta", action="store_true",
                        help="Predict delta joint angles (target - current) instead of absolute")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Validation split ratio")
    
    # Loss weights for different action components
    parser.add_argument("--weight_base", type=float, default=1.0)
    parser.add_argument("--weight_trunk", type=float, default=1.0)
    parser.add_argument("--weight_arm", type=float, default=1.0)
    parser.add_argument("--weight_gripper", type=float, default=0.5)
    
    # Output
    parser.add_argument("--output_dir", type=str, default="outputs/pose_controller")
    parser.add_argument("--save_every", type=int, default=10,
                        help="Save checkpoint every N epochs")
    
    # Misc
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    
    return parser.parse_args()


def weighted_mse_loss(pred: torch.Tensor, target: torch.Tensor, weights: dict) -> torch.Tensor:
    """Compute weighted MSE loss for different action components."""
    loss = 0.0
    
    # Base velocity [0:3]
    loss += weights["base"] * F.mse_loss(
        pred[:, ActionIndices.BASE], 
        target[:, ActionIndices.BASE]
    )
    
    # Trunk [3:7]
    loss += weights["trunk"] * F.mse_loss(
        pred[:, ActionIndices.TORSO],
        target[:, ActionIndices.TORSO]
    )
    
    # Left arm [7:14]
    loss += weights["arm"] * F.mse_loss(
        pred[:, ActionIndices.LEFT_ARM],
        target[:, ActionIndices.LEFT_ARM]
    )
    
    # Left gripper [14:15]
    loss += weights["gripper"] * F.mse_loss(
        pred[:, ActionIndices.LEFT_GRIPPER],
        target[:, ActionIndices.LEFT_GRIPPER]
    )
    
    # Right arm [15:22]
    loss += weights["arm"] * F.mse_loss(
        pred[:, ActionIndices.RIGHT_ARM],
        target[:, ActionIndices.RIGHT_ARM]
    )
    
    # Right gripper [22:23]
    loss += weights["gripper"] * F.mse_loss(
        pred[:, ActionIndices.RIGHT_GRIPPER],
        target[:, ActionIndices.RIGHT_GRIPPER]
    )
    
    return loss


def train_epoch(model, dataloader, optimizer, device, weights):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        state = batch["state"].to(device)
        target_left = batch["target_eef_left"].to(device)
        target_right = batch["target_eef_right"].to(device)
        action_gt = batch["action"].to(device)
        
        optimizer.zero_grad()
        
        action_pred = model(state, target_left, target_right)
        loss = weighted_mse_loss(action_pred, action_gt, weights)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


@torch.no_grad()
def validate(model, dataloader, device, weights):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    # Per-component errors
    errors = {
        "base": 0.0,
        "trunk": 0.0,
        "left_arm": 0.0,
        "right_arm": 0.0,
        "left_grip": 0.0,
        "right_grip": 0.0,
    }
    
    for batch in tqdm(dataloader, desc="Validating", leave=False):
        state = batch["state"].to(device)
        target_left = batch["target_eef_left"].to(device)
        target_right = batch["target_eef_right"].to(device)
        action_gt = batch["action"].to(device)
        
        action_pred = model(state, target_left, target_right)
        loss = weighted_mse_loss(action_pred, action_gt, weights)
        
        # Per-component MAE
        errors["base"] += F.l1_loss(
            action_pred[:, ActionIndices.BASE],
            action_gt[:, ActionIndices.BASE]
        ).item()
        errors["trunk"] += F.l1_loss(
            action_pred[:, ActionIndices.TORSO],
            action_gt[:, ActionIndices.TORSO]
        ).item()
        errors["left_arm"] += F.l1_loss(
            action_pred[:, ActionIndices.LEFT_ARM],
            action_gt[:, ActionIndices.LEFT_ARM]
        ).item()
        errors["right_arm"] += F.l1_loss(
            action_pred[:, ActionIndices.RIGHT_ARM],
            action_gt[:, ActionIndices.RIGHT_ARM]
        ).item()
        errors["left_grip"] += F.l1_loss(
            action_pred[:, ActionIndices.LEFT_GRIPPER],
            action_gt[:, ActionIndices.LEFT_GRIPPER]
        ).item()
        errors["right_grip"] += F.l1_loss(
            action_pred[:, ActionIndices.RIGHT_GRIPPER],
            action_gt[:, ActionIndices.RIGHT_GRIPPER]
        ).item()
        
        total_loss += loss.item()
        num_batches += 1
    
    for k in errors:
        errors[k] /= num_batches
    
    return total_loss / num_batches, errors


def main():
    args = parse_args()
    
    # Handle --fast mode
    if args.fast:
        args.max_tasks = 2
        args.max_episodes = 5
        args.epochs = 5
        logger.info("FAST MODE: 2 tasks, 5 episodes each, 5 epochs")
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {output_dir}")
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load dataset
    logger.info("Loading dataset...")
    
    # Determine standard track mode
    standard_track = not args.no_standard_track
    if standard_track:
        logger.info("Using FILTERED STATE mode: 172-dim state (no effort, gripper, velocities)")
        state_dim = 172
    else:
        logger.info("Using FULL STATE mode: 256-dim state (includes all proprioception)")
        state_dim = 256
    
    if args.use_delta:
        # Delta mode: predict (target_joints - current_joints)
        logger.info("Using DELTA mode: predicting joint angle deltas")
        dataset = PoseControlDatasetDelta(
            data_root=args.data_root,
            lookahead_steps=args.lookahead,
            normalize=True,
            max_tasks=args.max_tasks,
            max_episodes_per_task=args.max_episodes,
            standard_track=standard_track,
        )
        normalize = True
    elif args.lazy_load:
        dataset = PoseControlDatasetLazy(
            data_root=args.data_root,
            lookahead_steps=args.lookahead,
            max_tasks=args.max_tasks,
            max_episodes_per_task=args.max_episodes,
        )
        # For lazy dataset, normalization happens in training loop
        normalize = False
    else:
        dataset = PoseControlDataset(
            data_root=args.data_root,
            lookahead_steps=args.lookahead,
            normalize=True,
            max_tasks=args.max_tasks,
            max_episodes_per_task=args.max_episodes,
        )
        normalize = True
    
    # Train/val split
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # Create model
    if args.use_delta:
        ModelClass = WholeBodyPoseControllerDelta
        logger.info(f"Using WholeBodyPoseControllerDelta model (state_dim={state_dim})")
    elif args.use_residual:
        ModelClass = WholeBodyPoseControllerWithResidual
        logger.info("Using WholeBodyPoseControllerWithResidual model")
    else:
        ModelClass = WholeBodyPoseController
        logger.info("Using WholeBodyPoseController model")
    
    # Pass state_dim for delta model
    if args.use_delta:
        model = ModelClass(
            state_dim=state_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(device)
    else:
        model = ModelClass(
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")
    
    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Loss weights
    weights = {
        "base": args.weight_base,
        "trunk": args.weight_trunk,
        "arm": args.weight_arm,
        "gripper": args.weight_gripper,
    }
    
    # Save config
    config = vars(args).copy()
    config["num_params"] = num_params
    config["train_samples"] = len(train_dataset)
    config["val_samples"] = len(val_dataset)
    
    import json
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Save normalization stats (only for non-lazy dataset)
    if hasattr(dataset, 'state_mean'):
        norm_stats = {
            "state_mean": dataset.state_mean.tolist(),
            "state_std": dataset.state_std.tolist(),
            "eef_left_mean": dataset.eef_left_mean.tolist(),
            "eef_left_std": dataset.eef_left_std.tolist(),
            "eef_right_mean": dataset.eef_right_mean.tolist(),
            "eef_right_std": dataset.eef_right_std.tolist(),
        }
        # Delta mode uses action_delta stats, regular mode uses action stats
        if hasattr(dataset, 'action_delta_mean'):
            norm_stats["action_delta_mean"] = dataset.action_delta_mean.tolist()
            norm_stats["action_delta_std"] = dataset.action_delta_std.tolist()
        if hasattr(dataset, 'action_mean'):
            norm_stats["action_mean"] = dataset.action_mean.tolist()
            norm_stats["action_std"] = dataset.action_std.tolist()
        with open(output_dir / "norm_stats.json", "w") as f:
            json.dump(norm_stats, f)
    else:
        logger.warning("Lazy dataset: normalization stats not saved (compute separately)")
    
    # Training loop
    best_val_loss = float("inf")
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, weights)
        val_loss, val_errors = validate(model, val_loader, device, weights)
        scheduler.step()
        
        logger.info(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )
        logger.info(
            f"  Val MAE - Base: {val_errors['base']:.4f}, "
            f"Trunk: {val_errors['trunk']:.4f}, "
            f"L-Arm: {val_errors['left_arm']:.4f}, "
            f"R-Arm: {val_errors['right_arm']:.4f}"
        )
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            }, output_dir / "best_model.pt")
            logger.info(f"  Saved best model (val_loss: {val_loss:.4f})")
        
        # Periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            }, output_dir / f"checkpoint_epoch{epoch+1}.pt")
    
    # Save final model
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "val_loss": val_loss,
    }, output_dir / "final_model.pt")
    
    logger.info(f"Training complete! Best val loss: {best_val_loss:.4f}")
    logger.info(f"Models saved to: {output_dir}")


if __name__ == "__main__":
    main()
