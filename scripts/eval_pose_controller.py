#!/usr/bin/env python3
"""Evaluate trained pose controller.

Usage:
    python scripts/eval_pose_controller.py \
        --checkpoint outputs/pose_controller/run_xxx/best_model.pt \
        --data_root /path/to/b1k_full
"""

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from b1k.pose_controller.model import WholeBodyPoseController, WholeBodyPoseControllerWithResidual
from b1k.pose_controller.dataset import PoseControlDataset, ActionIndices

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--lookahead", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, dataloader, device, norm_stats):
    model.eval()
    
    all_preds = []
    all_targets = []
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        state = batch["state"].to(device)
        target_left = batch["target_eef_left"].to(device)
        target_right = batch["target_eef_right"].to(device)
        action_gt = batch["action"].to(device)
        
        action_pred = model(state, target_left, target_right)
        
        all_preds.append(action_pred.cpu())
        all_targets.append(action_gt.cpu())
    
    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    # Denormalize
    action_mean = torch.tensor(norm_stats["action_mean"])
    action_std = torch.tensor(norm_stats["action_std"])
    
    preds_denorm = preds * action_std + action_mean
    targets_denorm = targets * action_std + action_mean
    
    # Compute metrics
    results = {}
    
    # Overall
    results["mse"] = F.mse_loss(preds_denorm, targets_denorm).item()
    results["mae"] = F.l1_loss(preds_denorm, targets_denorm).item()
    
    # Per-component
    components = {
        "base": ActionIndices.BASE,
        "trunk": ActionIndices.TORSO,
        "left_arm": ActionIndices.LEFT_ARM,
        "left_gripper": ActionIndices.LEFT_GRIPPER,
        "right_arm": ActionIndices.RIGHT_ARM,
        "right_gripper": ActionIndices.RIGHT_GRIPPER,
    }
    
    for name, idx in components.items():
        results[f"{name}_mae"] = F.l1_loss(
            preds_denorm[:, idx],
            targets_denorm[:, idx]
        ).item()
        results[f"{name}_mse"] = F.mse_loss(
            preds_denorm[:, idx],
            targets_denorm[:, idx]
        ).item()
    
    return results


def main():
    args = parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    run_dir = checkpoint_path.parent
    
    # Load config
    with open(run_dir / "config.json") as f:
        config = json.load(f)
    
    # Load norm stats
    with open(run_dir / "norm_stats.json") as f:
        norm_stats = json.load(f)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Load model
    use_residual = config.get("use_residual", False)
    ModelClass = WholeBodyPoseControllerWithResidual if use_residual else WholeBodyPoseController
    
    model = ModelClass(
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=0.0,  # No dropout at eval
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Load dataset
    dataset = PoseControlDataset(
        data_root=args.data_root,
        lookahead_steps=args.lookahead,
        normalize=True,
    )
    
    # Use same stats as training
    dataset.state_mean = np.array(norm_stats["state_mean"])
    dataset.state_std = np.array(norm_stats["state_std"])
    dataset.action_mean = np.array(norm_stats["action_mean"])
    dataset.action_std = np.array(norm_stats["action_std"])
    dataset.eef_left_mean = np.array(norm_stats["eef_left_mean"])
    dataset.eef_left_std = np.array(norm_stats["eef_left_std"])
    dataset.eef_right_mean = np.array(norm_stats["eef_right_mean"])
    dataset.eef_right_std = np.array(norm_stats["eef_right_std"])
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )
    
    # Evaluate
    results = evaluate(model, dataloader, device, norm_stats)
    
    # Print results
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Overall MSE: {results['mse']:.6f}")
    print(f"Overall MAE: {results['mae']:.6f}")
    print()
    print("Per-component MAE:")
    print(f"  Base velocity:   {results['base_mae']:.6f}")
    print(f"  Trunk position:  {results['trunk_mae']:.6f}")
    print(f"  Left arm:        {results['left_arm_mae']:.6f}")
    print(f"  Right arm:       {results['right_arm_mae']:.6f}")
    print(f"  Left gripper:    {results['left_gripper_mae']:.6f}")
    print(f"  Right gripper:   {results['right_gripper_mae']:.6f}")
    print("=" * 50)
    
    # Save results
    with open(run_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {run_dir / 'eval_results.json'}")


if __name__ == "__main__":
    main()
