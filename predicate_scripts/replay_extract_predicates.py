#!/usr/bin/env python3
"""
Replay HDF5 demonstrations and extract predicate states at each timestep.

Saves predicate data in Parquet format (efficient) and optionally JSONL (human-readable).

Usage:
    python replay_extract_predicates.py --file <path_to_hdf5>
    python replay_extract_predicates.py --dir <path_to_folder>
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch as th
import pandas as pd

# OmniGibson imports
from omnigibson.envs import DataPlaybackWrapper
from omnigibson.macros import gm
from omnigibson.utils.config_utils import TorchEncoder
import omnigibson as og

from bddl.condition_evaluation import (
        NQuantifier, Universal, Existential, Conjunction, Disjunction,
        Negation, HEAD, ForPairs, ForNPairs
    )
from bddl.logic_base import BinaryAtomicFormula, UnaryAtomicFormula
# Disable rendering for speed
gm.RENDER_VIEWER_CAMERA = False
gm.DEFAULT_VIEWER_WIDTH = 1280
gm.DEFAULT_VIEWER_HEIGHT = 720
gm.HEADLESS = True

# Disable transition rules for playback
gm.ENABLE_TRANSITION_RULES = False


# =============================================================================
# Helper functions for BDDL body parsing
# =============================================================================

def body_to_name(body):
    """Convert BDDL body to a readable name like 'inside(obj1, obj2)'."""
    def flatten(x):
        if isinstance(x, (list, tuple)):
            if len(x) == 0:
                return ''
            # First element is predicate name, rest are arguments
            pred_name = str(x[0])
            args = [str(a).lstrip('?') for a in x[1:] if a]
            if args:
                return f"{pred_name}({', '.join(args)})"
            return pred_name
        return str(x).lstrip('?')
    return flatten(body)


def get_category(body):
    """Extract the iterator category from body like ['?type.n.01', '-', 'type.n.01']"""
    if isinstance(body, (list, tuple)) and len(body) >= 1:
        iterable = body[0]
        if isinstance(iterable, (list, tuple)) and len(iterable) >= 3:
            return iterable[2]  # The type, e.g., 'table.n.02'
    return None


def get_category_forn(body):
    """Extract the iterator category from forn body like [['1'], ['?type', '-', 'type'], ['pred', ...]]"""
    # forn body: [['N'], ['?type', '-', 'type'], ['predicate', ...]]
    if isinstance(body, (list, tuple)) and len(body) >= 2:
        # Skip the count element [['1']]
        iterable = body[1] if len(body) > 1 else body[0]
        if isinstance(iterable, (list, tuple)) and len(iterable) >= 3:
            return iterable[2]  # The type, e.g., 'candy_cane.n.01'
    return None


def get_predicate_info(body):
    """Extract predicate name and args from body"""
    if isinstance(body, (list, tuple)) and len(body) >= 2:
        pred_part = body[1] if len(body) > 1 else body[0]
        if isinstance(pred_part, (list, tuple)) and len(pred_part) >= 1:
            pred_name = pred_part[0]
            args = [str(a).lstrip('?') for a in pred_part[1:] if a]
            return pred_name, args
    return None, []


def get_predicate_info_forn(body):
    """Extract predicate name and args from forn body [['N'], ['?type', '-', 'type'], ['predicate', ...]]"""
    if isinstance(body, (list, tuple)) and len(body) >= 3:
        pred_part = body[2]  # The predicate is the third element
        if isinstance(pred_part, (list, tuple)) and len(pred_part) >= 1:
            pred_name = pred_part[0]
            args = [str(a).lstrip('?') for a in pred_part[1:] if a]
            return pred_name, args
    return None, []


def get_param_label(body):
    """Extract param_label from body for scope lookup"""
    if isinstance(body, (list, tuple)) and len(body) >= 1:
        iterable = body[0]
        if isinstance(iterable, (list, tuple)) and len(iterable) >= 1:
            if isinstance(iterable[0], str):
                return iterable[0].strip('?')
    return None


def get_param_label_forn(body):
    """Extract param_label from forn body [['N'], ['?type', '-', 'type'], ...]"""
    if isinstance(body, (list, tuple)) and len(body) >= 2:
        iterable = body[1]
        if isinstance(iterable, (list, tuple)) and len(iterable) >= 1:
            if isinstance(iterable[0], str):
                return iterable[0].strip('?')
    return None


def get_instance_name(child, i, param_label):
    """
    Get grounded instance name from child expression.
    
    Args:
        child: Child expression with scope and/or input1
        i: Index of the child (fallback)
        param_label: Parameter label for scope lookup (e.g., 'obj' from '?obj')
    
    Returns:
        str: Instance name like 'table.n.02_1' or fallback 'instance_0'
    """
    instance_name = f"instance_{i}"
    if param_label and hasattr(child, 'scope') and isinstance(child.scope, dict):
        grounded = child.scope.get(param_label)
        if grounded and isinstance(grounded, str):
            instance_name = grounded
        else:
            # Fallback: search scope for matching key
            for sk, sv in child.scope.items():
                if isinstance(sv, str) and param_label and sv.startswith(param_label + '_'):
                    instance_name = sv
                    break
    # If still generic, try input1 for atomic formulas
    if instance_name.startswith("instance_") and hasattr(child, 'input1'):
        instance_name = child.input1
    return instance_name


def predicate_to_str(pred):
    """
    Convert a predicate (HEAD object) to a human-readable string.
    
    Args:
        pred: A bddl.condition_evaluation.HEAD object
        
    Returns:
        str: Human-readable predicate like "inside(apple_1, fridge_1)"
    """
    try:
        # pred.body contains the parsed BDDL expression
        # e.g., ['inside', '?apple', '?fridge'] or similar
        # pred.children[0] is the actual atomic formula with STATE_NAME
        if hasattr(pred, 'children') and len(pred.children) > 0:
            child = pred.children[0]
            if hasattr(child, 'STATE_NAME') and child.STATE_NAME:
                # Binary or unary atomic formula
                state_name = child.STATE_NAME
                if hasattr(child, 'input1') and hasattr(child, 'input2'):
                    return f"{state_name}({child.input1}, {child.input2})"
                elif hasattr(child, 'input1'):
                    return f"{state_name}({child.input1})"
                elif hasattr(child, 'input'):
                    return f"{state_name}({child.input})"
        
        # Fallback: try to reconstruct from body
        if hasattr(pred, 'body') and pred.body:
            def flatten(x):
                if isinstance(x, (list, tuple)):
                    return ' '.join(flatten(i) for i in x)
                return str(x).lstrip('?')
            return flatten(pred.body)
        
        # Last resort
        return repr(pred)
    except Exception:
        return repr(pred)


def extract_predicate_tree(expr, _current_depth=0, max_depth=10):
    """
    Extract predicate evaluation as a proper hierarchical tree structure.
    
    Returns a dict representing the predicate tree with all nested quantifiers,
    counts, instances, and satisfaction status preserved in structure.
    
    Args:
        expr: A BDDL Expression object (HEAD, NQuantifier, Universal, etc.)
        _current_depth: Internal counter for recursion depth
        max_depth: Maximum recursion depth (default: 10)
        
    Returns:
        dict: Hierarchical representation of the predicate, e.g.:
            {
                "type": "exists",
                "category": "table.n.02",
                "satisfied": True,
                "count": 1,
                "total": 2,
                "threshold": 1,
                "instances": {
                    "table.n.02_1": {
                        "satisfied": True,
                        "nested": [
                            {
                                "type": "forall",
                                "category": "pillar_candle.n.01",
                                "predicate": "ontop",
                                "count": 2,
                                "threshold": 2,
                                "satisfied": True
                            }
                        ]
                    }
                }
            }
    """

    
    if max_depth >= 0 and _current_depth > max_depth:
        return {"type": "max_depth_reached"}
    
    result = {}
    
    def iterate_children_with_recursion(expr, param_label, skip_atomic=True):
        """
        Shared child iteration logic for quantifiers (forall, exists, forn).
        Returns a dict mapping instance_name -> instance_data with optional nested tree.
        
        Args:
            expr: The quantifier expression with children and child_values
            param_label: The parameter label for scope lookup (e.g., 'obj' from '?obj')
            skip_atomic: If True, skip adding 'nested' for atomic predicates
        """
        instances = {}
        if not expr.children:
            return instances
        
        for i, child in enumerate(expr.children):
            # Get grounded instance name from scope or input1
            instance_name = f"instance_{i}"
            if param_label and hasattr(child, 'scope') and isinstance(child.scope, dict):
                grounded = child.scope.get(param_label)
                if grounded and isinstance(grounded, str):
                    instance_name = grounded
                else:
                    # Fallback: search scope for matching key
                    for sk, sv in child.scope.items():
                        if isinstance(sv, str) and param_label and sv.startswith(param_label + '_'):
                            instance_name = sv
                            break
            # If still generic, try input1 for atomic formulas
            if instance_name.startswith("instance_") and hasattr(child, 'input1'):
                instance_name = child.input1
            
            # Store per-instance satisfaction
            instance_data = {
                "satisfied": bool(expr.child_values[i]) if expr.child_values and i < len(expr.child_values) else False
            }
            
            # Recurse into nested quantifiers
            nested = extract_predicate_tree(child, _current_depth + 1, max_depth)
            skip_types = ["empty_head", "max_depth_reached", None]
            if skip_atomic:
                skip_types.append("atomic")
            if nested and nested.get("type") not in skip_types:
                instance_data["nested"] = nested
            
            instances[instance_name] = instance_data
        
        return instances
    
    if isinstance(expr, HEAD):
        # HEAD wraps a single child expression
        child = expr.children[0] if expr.children else None
        if child is not None:
            return extract_predicate_tree(child, _current_depth, max_depth)
        return {"type": "empty_head"}
    
    elif isinstance(expr, (BinaryAtomicFormula, UnaryAtomicFormula)):
        # Atomic predicate - leaf node
        result["type"] = "atomic"
        if hasattr(expr, 'STATE_NAME'):
            result["predicate"] = expr.STATE_NAME
        if hasattr(expr, 'input1'):
            result["arg1"] = expr.input1
        if hasattr(expr, 'input2'):
            result["arg2"] = expr.input2
        elif hasattr(expr, 'input'):
            result["arg1"] = expr.input
        # Get value from parent's child_values if available
        return result
    
    elif isinstance(expr, NQuantifier):
        # forn N: requires exactly N of the children to be true
        result["type"] = "forn"
        result["n"] = expr.N
        # forn has different body structure: [['N'], ['?type', '-', 'type'], ['predicate', ...]]
        category = get_category_forn(expr.body)
        if category:
            result["category"] = category
        pred_name, pred_args = get_predicate_info_forn(expr.body)
        if pred_name:
            result["predicate"] = pred_name
            if pred_args:
                result["args"] = pred_args
        
        count = sum(expr.child_values) if expr.child_values else 0
        result["count"] = int(count)
        result["threshold"] = expr.N
        result["satisfied"] = (count >= expr.N)
        
        # Use shared child iteration helper
        param_label = get_param_label_forn(expr.body)
        result["instances"] = iterate_children_with_recursion(expr, param_label, skip_atomic=True)
        
        return result
    
    elif isinstance(expr, Universal):
        # forall: requires all children to be true
        result["type"] = "forall"
        category = get_category(expr.body)
        if category:
            result["category"] = category
        pred_name, pred_args = get_predicate_info(expr.body)
        if pred_name:
            result["predicate"] = pred_name
            if pred_args:
                result["args"] = pred_args
        
        count = sum(expr.child_values) if expr.child_values else 0
        threshold = len(expr.children)
        result["count"] = int(count)
        result["threshold"] = threshold
        result["satisfied"] = (count == threshold)
        
        # Use shared child iteration helper
        param_label = get_param_label(expr.body)
        result["instances"] = iterate_children_with_recursion(expr, param_label, skip_atomic=True)
        
        return result
    
    elif isinstance(expr, Existential):
        # exists: requires at least 1 child to be true
        result["type"] = "exists"
        category = get_category(expr.body)
        if category:
            result["category"] = category
        
        count = sum(expr.child_values) if expr.child_values else 0
        total = len(expr.children) if expr.children else 0
        result["count"] = int(count)
        result["total"] = total
        result["threshold"] = 1
        result["satisfied"] = (count >= 1)
        
        # Use shared child iteration helper (include atomic for exists since children might be simple predicates)
        param_label = get_param_label(expr.body)
        result["instances"] = iterate_children_with_recursion(expr, param_label, skip_atomic=False)
        
        return result
    
    elif isinstance(expr, ForPairs):
        # forpairs: bijective matching
        result["type"] = "forpairs"
        if hasattr(expr, 'body') and isinstance(expr.body, (list, tuple)):
            # Extract both categories
            if len(expr.body) >= 2:
                cat1 = get_category([expr.body[0]])
                cat2 = get_category([expr.body[1]])
                if cat1:
                    result["category1"] = cat1
                if cat2:
                    result["category2"] = cat2
            pred_name, pred_args = get_predicate_info(expr.body)
            if pred_name:
                result["predicate"] = pred_name
        
        if hasattr(expr, 'child_values') and expr.child_values is not None:
            import numpy as np
            pair_count = int(np.sum(expr.child_values))
            total_pairs = expr.child_values.size
            L = min(expr.child_values.shape) if len(expr.child_values.shape) == 2 else 0
            result["count"] = pair_count
            result["total"] = total_pairs
            result["threshold"] = L
            row_satisfied = int(np.sum(np.any(expr.child_values, axis=1)))
            col_satisfied = int(np.sum(np.any(expr.child_values, axis=0)))
            result["satisfied"] = (row_satisfied >= L and col_satisfied >= L)
        
        return result
    
    elif isinstance(expr, ForNPairs):
        # fornpairs N: N pairs must satisfy
        result["type"] = "fornpairs"
        result["n"] = expr.N
        if hasattr(expr, 'body') and isinstance(expr.body, (list, tuple)):
            if len(expr.body) >= 2:
                cat1 = get_category([expr.body[0]])
                cat2 = get_category([expr.body[1]])
                if cat1:
                    result["category1"] = cat1
                if cat2:
                    result["category2"] = cat2
            pred_name, pred_args = get_predicate_info(expr.body)
            if pred_name:
                result["predicate"] = pred_name
        
        if hasattr(expr, 'child_values') and expr.child_values is not None:
            import numpy as np
            pair_count = int(np.sum(expr.child_values))
            total_pairs = expr.child_values.size
            result["count"] = pair_count
            result["total"] = total_pairs
            result["threshold"] = expr.N
            row_satisfied = int(np.sum(np.any(expr.child_values, axis=1)))
            col_satisfied = int(np.sum(np.any(expr.child_values, axis=0)))
            result["satisfied"] = (row_satisfied >= expr.N and col_satisfied >= expr.N)
        
        return result
    
    elif isinstance(expr, Conjunction):
        # and: all must be true
        result["type"] = "and"
        result["satisfied"] = all(expr.child_values) if expr.child_values else False
        result["children"] = []
        for child in expr.children:
            child_tree = extract_predicate_tree(child, _current_depth + 1, max_depth)
            if child_tree:
                result["children"].append(child_tree)
        return result
    
    elif isinstance(expr, Disjunction):
        # or: at least one must be true
        result["type"] = "or"
        result["satisfied"] = any(expr.child_values) if expr.child_values else False
        result["children"] = []
        for child in expr.children:
            child_tree = extract_predicate_tree(child, _current_depth + 1, max_depth)
            if child_tree:
                result["children"].append(child_tree)
        return result
    
    elif isinstance(expr, Negation):
        # not: negation of child
        result["type"] = "not"
        child = expr.children[0] if expr.children else None
        if child is not None:
            result["child"] = extract_predicate_tree(child, _current_depth + 1, max_depth)
            # Satisfaction is inverted
            child_sat = result["child"].get("satisfied", False) if result["child"] else False
            result["satisfied"] = not child_sat
        return result
    
    return result


def get_predicate_states_hierarchical(env):
    """
    Get predicate states as a list of hierarchical tree structures.
    
    Returns:
        list: List of predicate tree dicts, one per goal condition
    """
    goal_conditions = env.task.activity_goal_conditions
    predicates = []
    
    for pred in goal_conditions:
        # Evaluate to populate child_values throughout the tree
        value = pred.evaluate()
        
        # Extract as tree structure
        tree = extract_predicate_tree(pred)
        tree["_raw_description"] = predicate_to_str(pred)
        tree["_satisfied"] = bool(value)
        
        predicates.append(tree)
    
    return predicates


def get_goal_progress(env):
    """
    Get overall goal progress.
    
    Returns:
        dict with 'satisfied_count', 'total_count', 'progress'
    """
    from bddl.activity import evaluate_goal_conditions
    
    try:
        done, goal_status = evaluate_goal_conditions(env.task.activity_goal_conditions)
        satisfied = len(goal_status.get("satisfied", []))
        unsatisfied = len(goal_status.get("unsatisfied", []))
        total = satisfied + unsatisfied
        progress = satisfied / total if total > 0 else 0.0
        
        return {
            "done": done,
            "satisfied_count": satisfied,
            "unsatisfied_count": unsatisfied,
            "total_count": total,
            "progress": progress,
        }
    except Exception as e:
        return {"error": str(e)}


def replay_and_extract_predicates(hdf_input_path, output_dir=None):
    """
    Replay a single HDF5 file and extract predicate states at each timestep.
    
    Args:
        hdf_input_path: Path to the HDF5 file to replay
        output_dir: Output directory (default: same as input file)
    
    Returns:
        Path to output parquet file
    """
    hdf_input_path = Path(hdf_input_path)
    
    if output_dir is None:
        output_dir = hdf_input_path.parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Output file paths
    base_name = hdf_input_path.stem
    parquet_path = output_dir / f"{base_name}_predicates.parquet"
    jsonl_path = output_dir / f"{base_name}_predicates.jsonl"
    
    print(f"Processing: {hdf_input_path}", flush=True)
    print(f"Output: {parquet_path}", flush=True)
    
    # Create playback environment
    # We need include_task=True to get access to predicates
    # output_path is required by the API even though we don't need it (it's designed
    # to record new data while replaying). The wrapper may still create this file.
    tmp_output_path = output_dir / f"{base_name}_replay_tmp.hdf5"
    env = DataPlaybackWrapper.create_from_hdf5(
        input_path=str(hdf_input_path),
        output_path=str(tmp_output_path),
        robot_obs_modalities=[],  # No visual obs needed
        n_render_iterations=1,
        only_successes=False,
        include_task=True,  # Required for predicates!
        include_task_obs=False,
        include_robot_control=False,
        include_contacts=True,  # Required for accurate predicate evaluation
    )
    
    # Get task info
    task_name = env.task.activity_name if hasattr(env.task, 'activity_name') else "unknown"
    print(f"Task: {task_name}")
    
    # Collect all predicate data
    all_records = []
    
    n_episodes = env.input_hdf5["data"].attrs["n_episodes"]
    print(f"Episodes to process: {n_episodes}")
    
    for episode_id in range(n_episodes):
        print(f"  Episode {episode_id}/{n_episodes-1}...")
        
        data_grp = env.input_hdf5["data"]
        if f"demo_{episode_id}" not in data_grp:
            print(f"    Skipping - demo_{episode_id} not found")
            continue
            
        traj_grp = data_grp[f"demo_{episode_id}"]
        
        try:
            # Load episode data
            from omnigibson.utils.python_utils import h5py_group_to_torch
            traj_data = h5py_group_to_torch(traj_grp)
            state = traj_data["state"]
            state_size = traj_data["state_size"]
            n_steps = len(state)
        except Exception as e:
            print(f"    Error loading episode: {e}")
            continue
        
        # Reset and restore initial state
        env.scene.restore(env.scene_file, update_initial_file=True)
        
        # Reset object attributes from stored metadata
        if "init_metadata" in traj_data:
            init_metadata = traj_data["init_metadata"]
            with og.sim.stopped():
                for i, obj in enumerate(env.scene.objects):
                    for attr, vals in init_metadata.items():
                        if i < len(vals):
                            val = vals[i]
                            setattr(obj, attr, val.item() if val.ndim == 0 else val)
        
        env.reset()
        
        # Build list of steps to sample: first, every 90 frames, and last
        sample_interval = 90
        step_indices = [0] + list(range(sample_interval, n_steps, sample_interval))
        if step_indices[-1] != n_steps - 1:
            step_indices.append(n_steps - 1)
        
        for step_idx in step_indices:
            og.sim.load_state(state[step_idx, :int(state_size[step_idx])], serialized=True)
            # Step a few times to let physics settle and contacts update
            # for _ in range(1):
            og.sim.step()
            
            progress = get_goal_progress(env)
            all_records.append({
                "episode_id": episode_id,
                "step": step_idx,
                "task_name": task_name,
                "progress": progress.get("progress", 0.0),
                "done": progress.get("done", False),
                "satisfied_count": progress.get("satisfied_count", 0),
                "total_count": progress.get("total_count", 0),
                "predicates": get_predicate_states_hierarchical(env),
            })
        
        print(f"    Processed {n_steps} steps ({len(step_indices)} samples)")
    
    # Convert to DataFrame and save
    df = pd.DataFrame(all_records)
    
    # Save as Parquet (efficient binary format)
    df.to_parquet(parquet_path, index=False)
    print(f"Saved Parquet: {parquet_path}")
    
    # Also save as JSONL for human readability
    with open(jsonl_path, 'w') as f:
        for record in all_records:
            f.write(json.dumps(record, cls=TorchEncoder) + '\n')
    print(f"Saved JSONL: {jsonl_path}")
    
    # Cleanup temp file created by DataPlaybackWrapper
    if tmp_output_path.exists():
        tmp_output_path.unlink()
    
    # Properly stop and clear environment before next file
    og.sim.stop()
    og.clear()
    
    return parquet_path


def main():
    parser = argparse.ArgumentParser(
        description="Replay HDF5 and extract predicate states",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single file
  python replay_extract_predicates.py --file demo.hdf5
  
  # Process directory
  python replay_extract_predicates.py --dir /path/to/demos
  
  # Filter by pattern
  python replay_extract_predicates.py --dir /path/to/demos --pattern task_name
"""
    )
    parser.add_argument("--file", type=str, help="Single HDF5 file to process")
    parser.add_argument("--dir", type=str, help="Directory containing HDF5 files")
    parser.add_argument("--output-dir", type=str, default=None, 
                        help="Output directory (default: same as input)")
    parser.add_argument("--pattern", type=str, default=None,
                        help="Only process files matching pattern")
    
    args = parser.parse_args()
    
    if args.file:
        hdf_files = [args.file]
    elif args.dir:
        dir_path = Path(args.dir)
        hdf_files = sorted(dir_path.rglob("*.hdf5"))
        if args.pattern:
            hdf_files = [f for f in hdf_files if args.pattern in str(f)]
        print(f"Found {len(hdf_files)} HDF5 files")
    else:
        parser.print_help()
        print("\nError: Either --file or --dir must be specified")
        return
    
    # Process each file
    for hdf_file in hdf_files:
        try:
            replay_and_extract_predicates(hdf_file, args.output_dir)
        except Exception as e:
            print(f"Error processing {hdf_file}: {e}")
            import traceback
            traceback.print_exc()
            # Cleanup after error to ensure next file can be processed
            try:
                og.sim.stop()
                og.clear()
            except Exception:
                pass
    
    og.shutdown()
    print("\nDone!")


if __name__ == "__main__":
    main()
