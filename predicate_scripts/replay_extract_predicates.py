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

# Disable rendering for speed
gm.RENDER_VIEWER_CAMERA = False
gm.DEFAULT_VIEWER_WIDTH = 1280
gm.DEFAULT_VIEWER_HEIGHT = 720
gm.HEADLESS = True

# Disable transition rules for playback
gm.ENABLE_TRANSITION_RULES = False


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


def extract_predicate_details(expr, prefix="", expand_instances=False, max_depth=2, 
                               quantifier_types=None, _current_depth=0):
    """
    Recursively extract predicate evaluation details from a BDDL expression tree.
    
    After evaluate() is called on the expression, child_values contains the evaluation
    results. This function extracts count/threshold info for quantifiers.
    
    Args:
        expr: A BDDL Expression object (HEAD, NQuantifier, Universal, etc.)
        prefix: Optional prefix for nested expressions
        expand_instances: If True, include per-instance boolean values (can be many columns)
        max_depth: Maximum recursion depth for nested expressions (0=no recursion, -1=unlimited)
        quantifier_types: Set of quantifier types to expand. Options: 
                         {'forn', 'forall', 'exists', 'fornpairs', 'forpairs', 'and', 'or', 'not'}
                         Default (None) = {'forn', 'forall', 'exists', 'fornpairs', 'forpairs'}
        _current_depth: Internal counter for recursion depth
        
    Returns:
        dict: Mapping of predicate keys to their evaluation details including:
            - For simple predicates: just the boolean value
            - For count predicates (forn, forall, exists): count, threshold, satisfied
    """
    from bddl.condition_evaluation import (
        NQuantifier, Universal, Existential, Conjunction, Disjunction,
        Negation, HEAD, ForPairs, ForNPairs
    )
    
    # Default quantifier types to expand
    if quantifier_types is None:
        quantifier_types = {'forn', 'forall', 'exists', 'fornpairs', 'forpairs'}
    
    # Check depth limit
    if max_depth >= 0 and _current_depth > max_depth:
        return {}
    
    results = {}
    expr_type = type(expr).__name__
    
    # Get the base predicate name from body - use human-readable format
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
    
    # Handle different expression types
    if isinstance(expr, HEAD):
        # HEAD wraps a single child expression
        child = expr.children[0] if expr.children else None
        if child is not None:
            child_results = extract_predicate_details(
                child, prefix, expand_instances, max_depth, quantifier_types, _current_depth
            )
            results.update(child_results)
            
            # Also store the overall HEAD evaluation
            pred_name = body_to_name(expr.body)
            if pred_name and pred_name not in results:
                results[pred_name] = bool(expr.child_values[0]) if expr.child_values else None
                
    elif isinstance(expr, NQuantifier) and 'forn' in quantifier_types:
        # forn N: requires exactly N of the children to be true
        # child_values already populated after evaluate()
        pred_name = body_to_name(expr.body) if hasattr(expr, 'body') else f"forn_{id(expr)}"
        count = sum(expr.child_values) if expr.child_values else 0
        threshold = expr.N
        results[f"forn_{pred_name}::count"] = int(count)
        results[f"forn_{pred_name}::threshold"] = threshold
        results[f"forn_{pred_name}::satisfied"] = (count >= threshold)
        
        # Optionally extract individual child predicates
        if expand_instances:
            for i, (child, val) in enumerate(zip(expr.children, expr.child_values or [])):
                if hasattr(child, 'input1'):
                    # Atomic formula - get specific instance
                    instance_name = child.input1 if hasattr(child, 'input1') else f"child_{i}"
                    results[f"forn_{pred_name}::{instance_name}"] = bool(val)
                
    elif isinstance(expr, Universal) and 'forall' in quantifier_types:
        # forall: requires all children to be true
        pred_name = body_to_name(expr.body) if hasattr(expr, 'body') else f"forall_{id(expr)}"
        count = sum(expr.child_values) if expr.child_values else 0
        threshold = len(expr.children)
        results[f"forall_{pred_name}::count"] = int(count)
        results[f"forall_{pred_name}::threshold"] = threshold  
        results[f"forall_{pred_name}::satisfied"] = (count == threshold)
        
        # Optionally extract individual child predicates  
        if expand_instances:
            for i, (child, val) in enumerate(zip(expr.children, expr.child_values or [])):
                if hasattr(child, 'input1'):
                    instance_name = child.input1
                    results[f"forall_{pred_name}::{instance_name}"] = bool(val)
        
    elif isinstance(expr, Existential) and 'exists' in quantifier_types:
        # exists: requires at least 1 child to be true
        pred_name = body_to_name(expr.body) if hasattr(expr, 'body') else f"exists_{id(expr)}"
        count = sum(expr.child_values) if expr.child_values else 0
        results[f"exists_{pred_name}::count"] = int(count)
        results[f"exists_{pred_name}::threshold"] = 1
        results[f"exists_{pred_name}::satisfied"] = (count >= 1)
        
    elif isinstance(expr, ForNPairs) and 'fornpairs' in quantifier_types:
        # fornpairs N: N pairs must satisfy the predicate
        pred_name = body_to_name(expr.body) if hasattr(expr, 'body') else f"fornpairs_{id(expr)}"
        # child_values is a 2D array for pairs
        if hasattr(expr, 'child_values') and expr.child_values is not None:
            import numpy as np
            # Count total satisfied pairs (True cells in the matrix)
            pair_count = int(np.sum(expr.child_values))
            total_pairs = expr.child_values.size
            results[f"fornpairs_{pred_name}::count"] = pair_count
            results[f"fornpairs_{pred_name}::total"] = total_pairs
            results[f"fornpairs_{pred_name}::threshold"] = expr.N
            # For satisfaction, need N rows and N cols to have at least one match each
            row_satisfied = int(np.sum(np.any(expr.child_values, axis=1)))
            col_satisfied = int(np.sum(np.any(expr.child_values, axis=0)))
            results[f"fornpairs_{pred_name}::satisfied"] = (row_satisfied >= expr.N and col_satisfied >= expr.N)
            
    elif isinstance(expr, ForPairs) and 'forpairs' in quantifier_types:
        # forpairs: all pairs must satisfy (bijective matching)
        pred_name = body_to_name(expr.body) if hasattr(expr, 'body') else f"forpairs_{id(expr)}"
        if hasattr(expr, 'child_values') and expr.child_values is not None:
            import numpy as np
            # Count total satisfied pairs (True cells in the matrix)
            pair_count = int(np.sum(expr.child_values))
            total_pairs = expr.child_values.size
            L = min(len(expr.children), len(expr.children[0])) if expr.children else 0
            results[f"forpairs_{pred_name}::count"] = pair_count
            results[f"forpairs_{pred_name}::total"] = total_pairs
            results[f"forpairs_{pred_name}::threshold"] = L
            # For satisfaction, need L rows and L cols to have at least one match each
            row_satisfied = int(np.sum(np.any(expr.child_values, axis=1)))
            col_satisfied = int(np.sum(np.any(expr.child_values, axis=0)))
            results[f"forpairs_{pred_name}::satisfied"] = (row_satisfied >= L and col_satisfied >= L)
            
    elif isinstance(expr, Conjunction) and 'and' in quantifier_types:
        # and: all must be true - recurse into children
        for i, child in enumerate(expr.children):
            child_results = extract_predicate_details(
                child, f"{prefix}and_{i}_", expand_instances, max_depth, 
                quantifier_types, _current_depth + 1
            )
            results.update(child_results)
            
    elif isinstance(expr, Disjunction) and 'or' in quantifier_types:
        # or: at least one must be true - recurse into children
        for i, child in enumerate(expr.children):
            child_results = extract_predicate_details(
                child, f"{prefix}or_{i}_", expand_instances, max_depth,
                quantifier_types, _current_depth + 1
            )
            results.update(child_results)
            
    elif isinstance(expr, Negation) and 'not' in quantifier_types:
        # not: negation of child
        child = expr.children[0] if expr.children else None
        if child is not None:
            child_results = extract_predicate_details(
                child, f"{prefix}not_", expand_instances, max_depth,
                quantifier_types, _current_depth + 1
            )
            # Invert the values for negated predicates
            for k, v in child_results.items():
                if isinstance(v, bool):
                    results[f"not_{k}"] = not v
                else:
                    results[f"not_{k}"] = v
                    
    else:
        # Atomic formula (BinaryAtomicFormula, UnaryAtomicFormula)
        # These are handled by the parent predicate_to_str, so we don't need to duplicate
        pass
            
    return results


def get_detailed_predicate_states(env):
    """
    Get detailed predicate evaluation states directly from the BDDL expression tree.
    
    This extracts count/threshold information for quantified predicates (forn, forall, exists)
    which is computed during online evaluation.
    
    Returns:
        dict: Mapping from predicate keys to values including:
            - Simple predicates: boolean
            - Count predicates: ::count, ::threshold, ::satisfied
    """
    detailed_states = {}
    
    # Iterate through all grounded goal state options
    for option_idx, option in enumerate(env.task.ground_goal_state_options):
        for pred_idx, pred in enumerate(option):
            # First evaluate to populate child_values throughout the tree
            pred.evaluate()
            
            # Then extract detailed information
            details = extract_predicate_details(pred)
            for key, value in details.items():
                full_key = f"{option_idx}:::{key}"
                detailed_states[full_key] = value
    
    return detailed_states


def get_predicate_states(env, detailed=True, expand_instances=False, max_depth=2, 
                          quantifier_types=None):
    """
    Get current state of all goal predicates from the ORIGINAL goal conditions
    (with forall/forn/exists quantifiers intact).
    
    Args:
        env: The environment with task and activity_goal_conditions
        detailed: If True, include count/threshold for quantified predicates
        expand_instances: If True, include per-instance boolean values
        max_depth: Maximum recursion depth (-1 for unlimited)
        quantifier_types: Set of quantifier types to expand (default: forn, forall, exists, fornpairs, forpairs)
    
    Returns:
        dict: Mapping from predicate string to boolean value (and counts if detailed=True)
    """
    predicate_states = {}
    
    # Use activity_goal_conditions which has the original structure with quantifiers
    # NOT ground_goal_state_options which is already expanded
    goal_conditions = env.task.activity_goal_conditions
    
    for pred in goal_conditions:
        # Evaluate to populate child_values throughout the tree
        value = pred.evaluate()
        pred_str = predicate_to_str(pred)
        predicate_states[pred_str] = bool(value)
        
        # Extract count/threshold info from quantifiers (forall, forn, exists, etc.)
        if detailed:
            details = extract_predicate_details(
                pred, 
                expand_instances=expand_instances,
                max_depth=max_depth,
                quantifier_types=quantifier_types
            )
            for detail_key, detail_value in details.items():
                predicate_states[detail_key] = detail_value
    
    return predicate_states


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


def replay_and_extract_predicates(hdf_input_path, output_dir=None, detailed=True,
                                   expand_instances=False, max_depth=2, quantifier_types=None):
    """
    Replay a single HDF5 file and extract predicate states at each timestep.
    
    Args:
        hdf_input_path: Path to the HDF5 file to replay
        output_dir: Output directory (default: same as input file)
        detailed: If True, include count/threshold info for quantified predicates (forn, forall, etc.)
        expand_instances: If True, include per-instance boolean values (can generate many columns)
        max_depth: Maximum recursion depth for nested expressions (default: 2, -1 for unlimited)
        quantifier_types: Set of quantifier types to expand (default: forn, forall, exists, fornpairs, forpairs)
    
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
        
        # Load initial state
        og.sim.load_state(state[0, :int(state_size[0])], serialized=True)
        og.sim.step()
        
        # Record initial predicate state
        pred_states = get_predicate_states(
            env, detailed=detailed, expand_instances=expand_instances,
            max_depth=max_depth, quantifier_types=quantifier_types
        )
        progress = get_goal_progress(env)
        
        record = {
            "episode_id": episode_id,
            "step": 0,
            "task_name": task_name,
            "progress": progress.get("progress", 0.0),
            "done": progress.get("done", False),
            "satisfied_count": progress.get("satisfied_count", 0),
            "total_count": progress.get("total_count", 0),
        }
        record.update(pred_states)
        all_records.append(record)
        
        # Process each step (sample every 90 frames = 3 seconds at 30Hz)
        sample_interval = 90
        for step_idx in range(sample_interval, n_steps, sample_interval):
            # Load state
            og.sim.load_state(state[step_idx, :int(state_size[step_idx])], serialized=True)
            og.sim.step()
            
            # Get predicate states
            pred_states = get_predicate_states(
                env, detailed=detailed, expand_instances=expand_instances,
                max_depth=max_depth, quantifier_types=quantifier_types
            )
            progress = get_goal_progress(env)
            
            record = {
                "episode_id": episode_id,
                "step": step_idx,
                "task_name": task_name,
                "progress": progress.get("progress", 0.0),
                "done": progress.get("done", False),
                "satisfied_count": progress.get("satisfied_count", 0),
                "total_count": progress.get("total_count", 0),
            }
            record.update(pred_states)
            all_records.append(record)
        
        n_samples = 1 + len(range(sample_interval, n_steps, sample_interval))
        print(f"    Processed {n_steps} steps ({n_samples} samples)")
    
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
  # Basic extraction with count/threshold info (default)
  python replay_extract_predicates.py --file demo.hdf5
  
  # Minimal output - only boolean values
  python replay_extract_predicates.py --file demo.hdf5 --no-detailed
  
  # Include per-instance values (more columns)
  python replay_extract_predicates.py --file demo.hdf5 --expand-instances
  
  # Only expand forn predicates, not forall/exists
  python replay_extract_predicates.py --file demo.hdf5 --quantifiers forn
  
  # Limit recursion depth for nested expressions
  python replay_extract_predicates.py --file demo.hdf5 --max-depth 1
"""
    )
    parser.add_argument("--file", type=str, help="Single HDF5 file to process")
    parser.add_argument("--dir", type=str, help="Directory containing HDF5 files")
    parser.add_argument("--output-dir", type=str, default=None, 
                        help="Output directory (default: same as input)")
    parser.add_argument("--pattern", type=str, default=None,
                        help="Only process files matching pattern")
    
    # Predicate expansion controls
    parser.add_argument("--detailed", action="store_true", default=True,
                        help="Include count/threshold for quantified predicates (default: True)")
    parser.add_argument("--no-detailed", action="store_false", dest="detailed",
                        help="Only include simple boolean predicate values")
    parser.add_argument("--expand-instances", action="store_true", default=False,
                        help="Include per-instance boolean values (can generate many columns)")
    parser.add_argument("--max-depth", type=int, default=2,
                        help="Max recursion depth for nested expressions (default: 2, -1 for unlimited)")
    parser.add_argument("--quantifiers", type=str, nargs="+", 
                        default=["forn", "forall", "exists", "fornpairs", "forpairs"],
                        help="Quantifier types to expand (default: forn forall exists fornpairs forpairs). "
                             "Options: forn, forall, exists, fornpairs, forpairs, and, or, not")
    
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
    
    # Convert quantifiers to set
    quantifier_types = set(args.quantifiers)
    
    # Process each file
    for hdf_file in hdf_files:
        try:
            replay_and_extract_predicates(
                hdf_file, 
                args.output_dir, 
                detailed=args.detailed,
                expand_instances=args.expand_instances,
                max_depth=args.max_depth,
                quantifier_types=quantifier_types
            )
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
