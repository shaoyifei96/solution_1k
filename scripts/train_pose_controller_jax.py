#!/usr/bin/env python3
"""Training script for whole-body pose controller using JAX/Flax.

Usage:
    python scripts/train_pose_controller_jax.py \
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
from functools import partial

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import numpy as np
from tqdm import tqdm

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from b1k.pose_controller.model_jax import WholeBodyPoseController, count_parameters
from b1k.pose_controller.dataset_jax import (
    PoseControlDataset, 
    PoseControlDataLoader,
    ActionIndices,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train whole-body pose controller (JAX)")
    
    # Data
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to BEHAVIOR-1K dataset root")
    parser.add_argument("--lookahead", type=int, default=15,
                        help="Lookahead steps for target EEF pose (default: 15 = 0.5s)")
    parser.add_argument("--max_tasks", type=int, default=None,
                        help="Max tasks to load (for debugging)")
    parser.add_argument("--max_episodes", type=int, default=None,
                        help="Max episodes per task (for debugging)")
    parser.add_argument("--streaming", action="store_true",
                        help="Use streaming data loader (memory efficient)")
    parser.add_argument("--fast", action="store_true",
                        help="Quick test: load only 2 tasks, 5 episodes each")
    
    # Model
    parser.add_argument("--hidden_dim", type=int, default=512,
                        help="Hidden dimension")
    parser.add_argument("--num_layers", type=int, default=4,
                        help="Number of MLP layers")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=1000)
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
    parser.add_argument("--seed", type=int, default=42)
    
    return parser.parse_args()


def weighted_mse_loss(pred: jnp.ndarray, target: jnp.ndarray, weights: dict) -> jnp.ndarray:
    """Compute weighted MSE loss for different action components."""
    loss = 0.0
    
    # Base velocity [0:3]
    loss += weights["base"] * jnp.mean(jnp.square(pred[:, 0:3] - target[:, 0:3]))
    
    # Trunk [3:7]
    loss += weights["trunk"] * jnp.mean(jnp.square(pred[:, 3:7] - target[:, 3:7]))
    
    # Left arm [7:14]
    loss += weights["arm"] * jnp.mean(jnp.square(pred[:, 7:14] - target[:, 7:14]))
    
    # Left gripper [14:15]
    loss += weights["gripper"] * jnp.mean(jnp.square(pred[:, 14:15] - target[:, 14:15]))
    
    # Right arm [15:22]
    loss += weights["arm"] * jnp.mean(jnp.square(pred[:, 15:22] - target[:, 15:22]))
    
    # Right gripper [22:23]
    loss += weights["gripper"] * jnp.mean(jnp.square(pred[:, 22:23] - target[:, 22:23]))
    
    return loss


def compute_mae_components(pred: jnp.ndarray, target: jnp.ndarray) -> dict:
    """Compute MAE for different action components."""
    return {
        "base": jnp.mean(jnp.abs(pred[:, 0:3] - target[:, 0:3])),
        "trunk": jnp.mean(jnp.abs(pred[:, 3:7] - target[:, 3:7])),
        "left_arm": jnp.mean(jnp.abs(pred[:, 7:14] - target[:, 7:14])),
        "right_arm": jnp.mean(jnp.abs(pred[:, 15:22] - target[:, 15:22])),
    }


@partial(jax.jit, static_argnums=(4,))
def train_step(
    model: WholeBodyPoseController,
    opt_state: optax.OptState,
    batch: dict,
    weights: dict,
    optimizer: optax.GradientTransformation,
):
    """Single training step."""
    
    def loss_fn(model):
        pred = model(
            batch["state"],
            batch["target_eef_left"],
            batch["target_eef_right"],
            training=True,
        )
        loss = weighted_mse_loss(pred, batch["action"], weights)
        return loss, pred
    
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, pred), grads = grad_fn(model)
    
    # Update parameters
    params = nnx.state(model, nnx.Param)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    nnx.update(model, new_params)
    
    return loss, new_opt_state


@jax.jit
def eval_step(
    model: WholeBodyPoseController,
    batch: dict,
    weights: dict,
):
    """Single evaluation step."""
    pred = model(
        batch["state"],
        batch["target_eef_left"],
        batch["target_eef_right"],
        training=False,
    )
    loss = weighted_mse_loss(pred, batch["action"], weights)
    mae = compute_mae_components(pred, batch["action"])
    return loss, mae


def save_checkpoint(model: WholeBodyPoseController, output_dir: Path, name: str):
    """Save model checkpoint."""
    import pickle
    
    # Get parameters as pytree
    params = nnx.state(model, nnx.Param)
    
    # Convert to numpy for serialization
    params_np = jax.tree.map(lambda x: np.array(x), params)
    
    checkpoint = {
        "params": params_np,
        "config": {
            "state_dim": model.state_dim,
            "eef_dim": model.eef_dim,
            "action_dim": model.action_dim,
        }
    }
    
    checkpoint_path = output_dir / f"{name}.pkl"
    with open(checkpoint_path, "wb") as f:
        pickle.dump(checkpoint, f)
    
    logger.info(f"Saved checkpoint to {checkpoint_path}")


def load_checkpoint(checkpoint_path: Path) -> dict:
    """Load model checkpoint."""
    import pickle
    
    with open(checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)
    
    return checkpoint


def main():
    args = parse_args()
    
    # Fast mode overrides
    if args.fast:
        args.max_tasks = 2
        args.max_episodes = 5
        args.epochs = 5
        logger.info(f"FAST MODE: {args.max_tasks} tasks, {args.max_episodes} episodes each, {args.epochs} epochs")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Check JAX devices
    devices = jax.devices()
    logger.info(f"JAX devices: {devices}")
    logger.info(f"JAX default backend: {jax.default_backend()}")
    
    # Set random seed
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    
    # Load dataset
    logger.info("Loading dataset...")
    if args.streaming:
        # Streaming loader for large datasets
        train_loader = PoseControlDataLoader(
            data_root=args.data_root,
            batch_size=args.batch_size,
            lookahead_steps=args.lookahead,
            max_tasks=args.max_tasks,
            max_episodes_per_task=args.max_episodes,
            shuffle=True,
            seed=args.seed,
        )
        val_loader = None  # No validation with streaming
        logger.info(f"Using streaming loader with {len(train_loader.episode_files)} episodes")
    else:
        # In-memory dataset for smaller data
        dataset = PoseControlDataset(
            data_root=args.data_root,
            lookahead_steps=args.lookahead,
            max_tasks=args.max_tasks,
            max_episodes_per_task=args.max_episodes,
        )
        logger.info(f"Loaded {len(dataset)} samples")
        
        # Split into train/val
        train_data, val_data = dataset.split(val_ratio=args.val_split, seed=args.seed)
        logger.info(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")
    
    # Create model
    rngs = nnx.Rngs(args.seed)
    model = WholeBodyPoseController(
        rngs=rngs,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    
    num_params = count_parameters(model)
    logger.info(f"Model parameters: {num_params:,}")
    
    # Create optimizer with warmup + cosine decay
    if args.streaming:
        total_steps = len(train_loader) * args.epochs
    else:
        total_steps = (len(train_data) // args.batch_size + 1) * args.epochs
    
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.lr,
        warmup_steps=args.warmup_steps,
        decay_steps=total_steps,
        end_value=args.lr * 0.01,
    )
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(schedule, weight_decay=args.weight_decay),
    )
    
    params = nnx.state(model, nnx.Param)
    opt_state = optimizer.init(params)
    
    # Loss weights
    weights = {
        "base": args.weight_base,
        "trunk": args.weight_trunk,
        "arm": args.weight_arm,
        "gripper": args.weight_gripper,
    }
    
    # Training loop
    best_val_loss = float("inf")
    
    for epoch in range(1, args.epochs + 1):
        # Training
        train_losses = []
        
        if args.streaming:
            batches = train_loader
        else:
            batches = train_data.get_batches(args.batch_size, shuffle=True, rng=rng)
        
        for batch in tqdm(batches, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            # Convert to JAX arrays
            batch_jax = {k: jnp.array(v) for k, v in batch.items()}
            
            loss, opt_state = train_step(model, opt_state, batch_jax, weights, optimizer)
            train_losses.append(float(loss))
        
        train_loss = np.mean(train_losses)
        
        # Validation
        if not args.streaming:
            val_losses = []
            val_maes = {"base": [], "trunk": [], "left_arm": [], "right_arm": []}
            
            for batch in val_data.get_batches(args.batch_size, shuffle=False):
                batch_jax = {k: jnp.array(v) for k, v in batch.items()}
                loss, mae = eval_step(model, batch_jax, weights)
                val_losses.append(float(loss))
                for k, v in mae.items():
                    val_maes[k].append(float(v))
            
            val_loss = np.mean(val_losses)
            val_mae_avg = {k: np.mean(v) for k, v in val_maes.items()}
            
            logger.info(
                f"Epoch {epoch}/{args.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"LR: {schedule(opt_state[1].count):.2e}"
            )
            logger.info(
                f"  Val MAE - Base: {val_mae_avg['base']:.4f}, "
                f"Trunk: {val_mae_avg['trunk']:.4f}, "
                f"L-Arm: {val_mae_avg['left_arm']:.4f}, "
                f"R-Arm: {val_mae_avg['right_arm']:.4f}"
            )
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, output_dir, "best_model")
                logger.info(f"  Saved best model (val_loss: {val_loss:.4f})")
        else:
            logger.info(f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f}")
            # Save periodically
            if epoch % args.save_every == 0:
                save_checkpoint(model, output_dir, f"checkpoint_epoch{epoch}")
    
    # Save final model
    save_checkpoint(model, output_dir, "final_model")
    
    if args.streaming:
        logger.info(f"Training complete! Models saved to: {output_dir}")
    else:
        logger.info(f"Training complete! Best val loss: {best_val_loss:.4f}")
        logger.info(f"Models saved to: {output_dir}")


if __name__ == "__main__":
    main()
