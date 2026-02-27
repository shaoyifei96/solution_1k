"""Whole-body pose controller for R1Pro robot.

This module provides a data-driven controller that maps:
    (current_state, target_eef_pose) → action

The controller learns from BEHAVIOR-1K demonstrations how to coordinate
base motion, trunk, and arm movements to achieve desired end-effector poses.
"""

from b1k.pose_controller.model import WholeBodyPoseController
from b1k.pose_controller.dataset import PoseControlDataset, LOOKAHEAD_STEPS

__all__ = [
    "WholeBodyPoseController",
    "PoseControlDataset", 
    "LOOKAHEAD_STEPS",
]
