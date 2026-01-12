"""B1K policy wrapper with action compression, rolling inpainting, and predicate consensus voting."""

import logging
import os
import pickle
import numpy as np
import torch
import dataclasses
from collections import deque
from typing import Dict, List, Optional

from openpi_client.base_policy import BasePolicy
from openpi_client.image_tools import resize_with_pad
from b1k.policies.b1k_policy import extract_state_from_proprio
from b1k.models.pi_behavior_config import TASK_NUM_PREDICATES, MAX_NUM_PREDICATES
from b1k.shared.correction_rules import apply_correction_rules, check_gripper_variation
from omnigibson.learning.utils.eval_utils import PROPRIOCEPTION_INDICES

logger = logging.getLogger(__name__)

RESIZE_SIZE = 224


@dataclasses.dataclass
class B1KWrapperConfig:
    """Configuration for B1K policy wrapper execution parameters."""
    actions_to_execute: int = 26
    actions_to_keep: int = 4
    execute_in_n_steps: int = 20
    time_threshold_inpaint: float = 0.3
    num_steps: int = 20
    apply_eval_tricks: bool = True
    
    # Predicate consensus settings
    predicate_data_path: str = "data/predicate_data"  # Path to predicate pkl files
    predicate_history_len: int = 3  # Number of predictions to consider for consensus
    predicate_votes_to_done: int = 2  # Votes needed to transition predicate 0→1
    predicate_allow_backward: bool = True  # Allow predicates to go back 1→0
    predicate_votes_to_undo: int = 3  # Votes needed to transition predicate 1→0 (if allowed)


class B1KPolicyWrapper():
    """B1K policy wrapper for PI_BEHAVIOR models with action compression, rolling inpainting, and predicate consensus voting."""
    
    # Class-level cache for predicate names per task
    _predicate_names_cache: Dict[int, List[str]] = {}
    
    def __init__(
        self, 
        policy: BasePolicy,
        text_prompt: str = "PI_BEHAVIOR model (task-conditioned)",  # Not used, kept for compatibility
        action_horizon: int = 30,
        task_id: int | None = None,
        config: B1KWrapperConfig = None,
        checkpoint_switcher = None,
    ) -> None:
        self.base_policy = policy
        self.policy = policy
        self.checkpoint_switcher = checkpoint_switcher
        self.text_prompt = text_prompt
        self.action_horizon = action_horizon
        self.config = config if config is not None else B1KWrapperConfig()
        
        # Validate configuration
        if self.config.actions_to_execute + self.config.actions_to_keep > self.action_horizon:
            raise ValueError(
                f"actions_to_execute + actions_to_keep exceeds action_horizon"
            )
        
        # PI_BEHAVIOR specific (always True for B1K)
        self.task_id = task_id
        
        # Predicate consensus state
        # current_predicate_states: [MAX_NUM_PREDICATES] bool array, True = object is done
        self.current_predicate_states = np.zeros(MAX_NUM_PREDICATES, dtype=bool)
        # Per-predicate prediction history: list of deques, one per predicate
        self.predicate_prediction_history = [
            deque([], maxlen=self.config.predicate_history_len)
            for _ in range(MAX_NUM_PREDICATES)
        ]
        # Predicate names for current task (loaded from pkl files)
        self.predicate_names: List[str] = []
        if task_id is not None:
            self._load_predicate_names(task_id)
        
        # Control loop variables
        self.last_actions = None
        self.action_index = 0
        self.step_count = 0
        self.prediction_count = 0
        self.next_initial_actions = None
    
    def reset(self):
        """Reset policy state."""
        self.policy.reset()
        self.last_actions = None
        self.action_index = 0
        self.step_count = 0
        self.prediction_count = 0
        self.next_initial_actions = None
        # Reset predicate consensus state
        self.current_predicate_states = np.zeros(MAX_NUM_PREDICATES, dtype=bool)
        for hist in self.predicate_prediction_history:
            hist.clear()
        logger.info(f"Policy reset - Task ID: {self.task_id}, Action horizon: {self.action_horizon}")
    
    def _load_predicate_names(self, task_id: int) -> None:
        """Load predicate names from the task's state_action_vectors.pkl file."""
        if task_id in B1KPolicyWrapper._predicate_names_cache:
            self.predicate_names = B1KPolicyWrapper._predicate_names_cache[task_id]
            return
        
        pkl_path = os.path.join(
            self.config.predicate_data_path, 
            f"task_{task_id:04d}_state_action_vectors.pkl"
        )
        
        if not os.path.exists(pkl_path):
            logger.warning(f"Predicate data file not found: {pkl_path}")
            self.predicate_names = [f"predicate_{i}" for i in range(TASK_NUM_PREDICATES[task_id])]
            return
        
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            
            # index_to_item: {0: 'item_name:object', 1: 'item_name:object', ...}
            index_to_item = data.get('index_to_item', {})
            num_items = data.get('num_items', 0)
            
            # Build ordered list of predicate names
            self.predicate_names = []
            for i in range(num_items):
                name = index_to_item.get(i, f"predicate_{i}")
                # Clean up the name (remove ':object' suffix if present)
                if name.endswith(':object'):
                    name = name[:-7]
                self.predicate_names.append(name)
            
            # Cache for future use
            B1KPolicyWrapper._predicate_names_cache[task_id] = self.predicate_names
            
            logger.info(f"Loaded {len(self.predicate_names)} predicate names for task {task_id}: {self.predicate_names}")
            
        except Exception as e:
            logger.warning(f"Failed to load predicate names from {pkl_path}: {e}")
            self.predicate_names = [f"predicate_{i}" for i in range(TASK_NUM_PREDICATES[task_id])]
    
    def format_predicate_states(self) -> str:
        """Format current predicate states with names for logging."""
        if self.task_id is None:
            return "No task"
        
        num_predicates = TASK_NUM_PREDICATES[self.task_id]
        parts = []
        for i in range(num_predicates):
            name = self.predicate_names[i] if i < len(self.predicate_names) else f"pred_{i}"
            state = "✓" if self.current_predicate_states[i] else "✗"
            parts.append(f"{name}:{state}")
        
        done_count = int(np.sum(self.current_predicate_states[:num_predicates]))
        return f"[{done_count}/{num_predicates}] " + " | ".join(parts)
    
    def _handle_task_change(self, new_task_id):
        """Handle task ID change by switching checkpoint and resetting state."""
        if self.task_id != new_task_id:
            old_task_id = self.task_id
            self.task_id = new_task_id
            
            logger.info(f"🔄 Task change detected: {old_task_id} → {new_task_id} (predicates: {TASK_NUM_PREDICATES[new_task_id]})")
            
            # Load predicate names for new task
            self._load_predicate_names(new_task_id)
            
            if self.checkpoint_switcher:
                new_policy = self.checkpoint_switcher.get_policy_for_task(new_task_id)
                if new_policy is not self.policy:
                    logger.info(f"📦 Switching checkpoint: task {old_task_id} → {new_task_id}")
                    self.base_policy = new_policy
                    self.policy = new_policy
                    self.policy.reset()
            
            # Reset predicate consensus state on task change
            self.current_predicate_states = np.zeros(MAX_NUM_PREDICATES, dtype=bool)
            for hist in self.predicate_prediction_history:
                hist.clear()
            self.last_actions = None
            self.action_index = 0
            self.next_initial_actions = None

    def process_obs(self, obs: dict) -> dict:
        """Process observation to match model input format."""
        prop_state = obs["robot_r1::proprio"]
        
        head_original = obs["robot_r1::robot_r1:zed_link:Camera:0::rgb"][..., :3]
        left_original = obs["robot_r1::robot_r1:left_realsense_link:Camera:0::rgb"][..., :3]
        right_original = obs["robot_r1::robot_r1:right_realsense_link:Camera:0::rgb"][..., :3]
        
        # Resize images
        head_resized = resize_with_pad(head_original, RESIZE_SIZE, RESIZE_SIZE)
        left_resized = resize_with_pad(left_original, RESIZE_SIZE, RESIZE_SIZE)
        right_resized = resize_with_pad(right_original, RESIZE_SIZE, RESIZE_SIZE)
        
        return {
            "observation/egocentric_camera": head_resized,
            "observation/wrist_image_left": left_resized,
            "observation/wrist_image_right": right_resized,
            "observation/state": prop_state,
            "prompt": self.text_prompt,
        }
    
    def update_predicate_states(self, predicate_logits):
        """Update predicate states using per-predicate consensus voting.
        
        Each predicate independently transitions (not linear progress!):
        - 0→1 (not done → done): when votes_to_done predictions agree it's done
        - 1→0 (done → not done): when votes_to_undo predictions agree it's not done (if allowed)
        
        Args:
            predicate_logits: [MAX_NUM_PREDICATES] logits from model (apply sigmoid for probabilities)
        """
        if self.task_id is None:
            return
        
        num_predicates = TASK_NUM_PREDICATES[self.task_id]
        
        # Apply sigmoid to get binary predictions (threshold at 0.5 = logit 0)
        predicted_done = predicate_logits > 0  # logit > 0 means sigmoid > 0.5
        
        # Update each predicate independently
        for i in range(num_predicates):
            pred = bool(predicted_done[i])
            self.predicate_prediction_history[i].append(pred)
            
            history = self.predicate_prediction_history[i]
            if len(history) < self.config.predicate_history_len:
                continue
            
            # Count votes for done (True) and not done (False)
            votes_done = sum(1 for p in history if p)
            votes_not_done = len(history) - votes_done
            
            # Get predicate name for logging
            pred_name = self.predicate_names[i] if i < len(self.predicate_names) else f"pred_{i}"
            
            if not self.current_predicate_states[i]:
                # Currently not done (0), check if should transition to done (1)
                if votes_done >= self.config.predicate_votes_to_done:
                    self.current_predicate_states[i] = True
                    self.predicate_prediction_history[i].clear()
                    logger.info(f"✅ [{pred_name}] → DONE (task {self.task_id}, step {self.step_count})")
                    logger.info(f"   Current: {self.format_predicate_states()}")
            else:
                # Currently done (1), check if should transition back to not done (0)
                if self.config.predicate_allow_backward:
                    if votes_not_done >= self.config.predicate_votes_to_undo:
                        self.current_predicate_states[i] = False
                        self.predicate_prediction_history[i].clear()
                        logger.info(f"↩️  [{pred_name}] → NOT DONE (task {self.task_id}, step {self.step_count})")
                        logger.info(f"   Current: {self.format_predicate_states()}")
    
    def prepare_batch_for_pi_behavior(self, batch):
        """Prepare batch for PI_BEHAVIOR model by adding task_id and predicate states."""
        task_id = self.task_id if self.task_id is not None else -1
        batch_copy = batch.copy()
        if "prompt" in batch_copy:
            del batch_copy["prompt"]
        
        # Task ID only (no stage - predicates are the state now)
        batch_copy["tokenized_prompt"] = np.array([task_id], dtype=np.int32)
        batch_copy["tokenized_prompt_mask"] = np.array([True], dtype=bool)
        
        # Add predicate states for predicate-based conditioning
        num_predicates = TASK_NUM_PREDICATES[task_id] if task_id >= 0 else 1
        predicate_mask = np.zeros(MAX_NUM_PREDICATES, dtype=bool)
        predicate_mask[:num_predicates] = True
        
        batch_copy["predicate_states"] = self.current_predicate_states.copy()
        batch_copy["predicate_mask"] = predicate_mask
        
        return batch_copy
    
    def _interpolate_actions(self, actions, target_steps):
        """Interpolate actions using cubic spline."""
        from scipy.interpolate import interp1d
        
        original_indices = np.linspace(0, len(actions)-1, len(actions))
        target_indices = np.linspace(0, len(actions)-1, target_steps)
        
        interpolated = np.zeros((target_steps, actions.shape[1]))
        for dim in range(actions.shape[1]):
            f = interp1d(original_indices, actions[:, dim], kind='cubic')
            interpolated[:, dim] = f(target_indices)
        
        return interpolated

    def act(self, obs: dict) -> torch.Tensor:
        """Main action function."""
        
        # Extract task_id from observations
        if "task_id" in obs:
            new_task_id = int(obs["task_id"][0])
            self._handle_task_change(new_task_id)
        
        raw_state = obs["robot_r1::proprio"]
        current_state = extract_state_from_proprio(raw_state)
        
        # Check if we need new actions
        if self.last_actions is None or self.action_index >= self.config.execute_in_n_steps:
            
            # Process observation
            model_input = self.process_obs(obs)
            model_input = self.prepare_batch_for_pi_behavior(model_input)
            
            # Add rolling inpainting if available
            if self.next_initial_actions is not None and ("initial_actions" not in model_input or model_input["initial_actions"] is None):
                model_input["initial_actions"] = self.next_initial_actions
            
            # Get prediction
            if "initial_actions" in model_input and model_input["initial_actions"] is not None:
                output = self.policy.infer(model_input, initial_actions=model_input["initial_actions"])
            else:
                output = self.policy.infer(model_input)
            
            actions = output["actions"]
            
            # Ensure correct shape
            if len(actions.shape) == 3:
                actions = actions[0]
            if actions.shape[1] > 23:
                actions = actions[:, :23]
            
            # Apply eval tricks if enabled
            should_compress = self.config.execute_in_n_steps < self.config.actions_to_execute
            
            if self.config.apply_eval_tricks:
                if self.task_id is not None:
                    actions_before = actions.copy()
                    actions, _ = apply_correction_rules(
                        self.task_id, 0, current_state, actions  # Stage no longer used
                    )
                    
                    # Log if actions were modified
                    if not np.allclose(actions_before, actions, rtol=1e-3):
                        max_diff = np.max(np.abs(actions_before - actions))
                        logger.info(f"🔧 Correction rule: Actions modified (max diff: {max_diff:.4f}, task {self.task_id})")
                
                if should_compress:
                    has_high_variation, mean_var, max_var = check_gripper_variation(
                        actions, self.config.actions_to_execute
                    )
                    if has_high_variation:
                        should_compress = False
                        logger.info(f"🔧 Gripper variation: Compression disabled (mean: {mean_var:.4f}, max: {max_var:.4f})")
            
            # Determine execution parameters
            actions_to_execute = self.config.actions_to_execute if should_compress else self.config.execute_in_n_steps
            execute_steps = self.config.execute_in_n_steps
            
            # Save actions for next inpainting (before compression)
            inpainting_start = actions_to_execute
            inpainting_end = inpainting_start + self.config.actions_to_keep
            
            if len(actions) >= inpainting_end:
                self.next_initial_actions = actions[inpainting_start:inpainting_end].copy()
            else:
                self.next_initial_actions = None
            
            # Extract and compress actions
            self.last_actions = actions[:actions_to_execute].copy()
            
            if should_compress:
                compressed_actions = self._interpolate_actions(self.last_actions, execute_steps)
                compression_factor = actions_to_execute / execute_steps
                compressed_actions[:, :3] *= compression_factor  # Scale velocities
                self.last_actions = compressed_actions
            
            self.action_index = 0
            self.prediction_count += 1
            
            # Log prediction details (at lower frequency, every 10 predictions)
            if self.prediction_count % 10 == 0:
                compression_status = f"compressed {actions_to_execute}→{execute_steps}" if should_compress else f"uncompressed ({execute_steps})"
                logger.info(f"🎯 Prediction #{self.prediction_count} | Actions: {compression_status} | Inpainting: {self.next_initial_actions is not None}")
            
            # Update predicate states based on model predictions
            if "predicate_logits" in output:
                self.update_predicate_states(output["predicate_logits"])
        
        # Get current action from sequence
        if self.action_index >= len(self.last_actions):
            self.action_index = 0
            
        current_action = self.last_actions[self.action_index]
        self.action_index += 1
        self.step_count += 1
        
        # Log progress every 100 steps (with predicate names)
        if self.step_count % 100 == 0:
            logger.info(f"📊 Step {self.step_count} | Task: {self.task_id} | {self.format_predicate_states()} | Predictions: {self.prediction_count}")
        # Convert to torch tensor
        action_tensor = torch.from_numpy(current_action).float()
        if len(action_tensor) > 23:
            action_tensor = action_tensor[:23]
        
        return action_tensor

