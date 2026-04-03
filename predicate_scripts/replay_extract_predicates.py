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
            child_tree = extract_predicate_tree(child, _current_depth, max_depth)
            # Attach satisfaction from HEAD's child_values (populated by evaluate())
            if expr.child_values is not None and len(expr.child_values) > 0:
                child_tree["satisfied"] = bool(expr.child_values[0])
            return child_tree
        return {"type": "empty_head"}
    
    elif isinstance(expr, (BinaryAtomicFormula, UnaryAtomicFormula)):
        # Atomic predicate - leaf node
        # Note: 'satisfied' is set by the PARENT from its child_values after recursion
        # (HEAD, Negation, Conjunction, Disjunction all attach satisfaction to children)
        result["type"] = "atomic"
        if hasattr(expr, 'STATE_NAME'):
            result["predicate"] = expr.STATE_NAME
        if hasattr(expr, 'input1'):
            result["arg1"] = expr.input1
        if hasattr(expr, 'input2'):
            result["arg2"] = expr.input2
        elif hasattr(expr, 'input'):
            result["arg1"] = expr.input
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
        for i, child in enumerate(expr.children):
            child_tree = extract_predicate_tree(child, _current_depth + 1, max_depth)
            if child_tree:
                # Attach satisfaction status from parent's child_values
                if expr.child_values is not None and i < len(expr.child_values):
                    child_tree["satisfied"] = bool(expr.child_values[i])
                result["children"].append(child_tree)
        return result
    
    elif isinstance(expr, Disjunction):
        # or: at least one must be true
        result["type"] = "or"
        result["satisfied"] = any(expr.child_values) if expr.child_values else False
        result["children"] = []
        for i, child in enumerate(expr.children):
            child_tree = extract_predicate_tree(child, _current_depth + 1, max_depth)
            if child_tree:
                # Attach satisfaction status from parent's child_values
                if expr.child_values is not None and i < len(expr.child_values):
                    child_tree["satisfied"] = bool(expr.child_values[i])
                result["children"].append(child_tree)
        return result
        return result
    
    elif isinstance(expr, Negation):
        # not: negation of child
        result["type"] = "not"
        child = expr.children[0] if expr.children else None
        if child is not None:
            result["child"] = extract_predicate_tree(child, _current_depth + 1, max_depth)
            # Get child satisfaction from expr.child_values (populated by evaluate())
            # This is more reliable than trying to get it from the recursively-built child dict
            if expr.child_values is not None and len(expr.child_values) > 0:
                child_sat = bool(expr.child_values[0])
                result["child"]["satisfied"] = child_sat  # Also store in child for consistency
                result["satisfied"] = not child_sat
            else:
                # Fallback: try to get from child dict (may be unreliable for atomic)
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
        pred.evaluate()
        
        # Extract as tree structure (satisfaction is attached by parent wrappers)
        tree = extract_predicate_tree(pred)
        tree["_raw_description"] = predicate_to_str(pred)
        
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


def _is_already_complete(jsonl_path):
    """Check if a JSONL output file already has a completed extraction."""
    if not jsonl_path.exists():
        return False
    try:
        with open(jsonl_path, 'r') as f:
            lines = f.readlines()
        if lines:
            last_record = json.loads(lines[-1])
            return last_record.get("done", False)
    except Exception:
        pass
    return False


def _process_hdf5_with_env(env, hdf5_path, output_dir, sim_steps, sample_interval, force=False):
    """
    Process a single HDF5 file using an already-loaded environment.

    The env must already have the correct scene loaded (same task).
    This avoids the expensive scene teardown/reload between same-task episodes.

    Args:
        env: DataPlaybackWrapper with scene already loaded
        hdf5_path: Path to HDF5 file to process
        output_dir: Output directory
        sim_steps: Number of simulation steps after loading state
        sample_interval: Sample every N frames
        force: Force re-processing

    Returns:
        Path to output parquet file, or None if skipped
    """
    import h5py as h5
    from omnigibson.utils.python_utils import h5py_group_to_torch, create_object_from_init_info
    import torch as th

    hdf5_path = Path(hdf5_path)
    output_dir = Path(output_dir)
    base_name = hdf5_path.stem
    parquet_path = output_dir / f"{base_name}_predicates.parquet"
    jsonl_path = output_dir / f"{base_name}_predicates.jsonl"

    if not force and _is_already_complete(jsonl_path):
        print(f"Skipping (already complete): {hdf5_path}", flush=True)
        return str(parquet_path)

    print(f"Processing: {hdf5_path}", flush=True)

    task_name = env.task.activity_name if hasattr(env.task, 'activity_name') else "unknown"
    all_records = []

    # Incremental writing: open JSONL in append mode so crashes don't lose prior steps
    jsonl_f = open(jsonl_path, 'w')

    with h5.File(str(hdf5_path), 'r') as f:
        data_grp = f["data"]
        scene_file = json.loads(data_grp.attrs["scene_file"])
        n_episodes = data_grp.attrs["n_episodes"]

        for episode_id in range(n_episodes):
            if f"demo_{episode_id}" not in data_grp:
                continue

            traj_grp = data_grp[f"demo_{episode_id}"]

            # Load transitions (object slicing/dicing/cooking that create new objects mid-episode)
            transitions = {}
            if "transitions" in traj_grp.attrs:
                try:
                    transitions = json.loads(traj_grp.attrs["transitions"])
                except Exception:
                    pass

            try:
                traj_data = h5py_group_to_torch(traj_grp)
                state = traj_data["state"]
                state_size = traj_data["state_size"]
                n_steps = len(state)
            except Exception as e:
                print(f"    Error loading episode {episode_id}: {e}")
                continue

            # Reset scene to this episode's initial state.
            # Pre-clean: remove transition objects individually with error tolerance.
            try:
                scene = og.sim.scenes[0]
                init_obj_names = set(scene_file["objects_info"]["init_info"].keys())
                extra_objs = [
                    scene.object_registry("name", name)
                    for name in list(scene.object_registry.get_dict("name").keys())
                    if name not in init_obj_names
                ]
                for obj in extra_objs:
                    if obj is not None:
                        try:
                            scene.remove_object(obj)
                        except (KeyError, Exception):
                            pass
                # Also clear any extra systems
                init_sys_names = set(scene_file["state"]["registry"]["system_registry"].keys())
                for sys_name in list(scene.active_systems.keys()):
                    if sys_name not in init_sys_names:
                        try:
                            scene.clear_system(sys_name)
                        except Exception:
                            pass
                env.scene.restore(scene_file, update_initial_file=True)

                if "init_metadata" in traj_data:
                    init_metadata = traj_data["init_metadata"]
                    with og.sim.stopped():
                        for i, obj in enumerate(env.scene.objects):
                            for attr, vals in init_metadata.items():
                                if i < len(vals):
                                    val = vals[i]
                                    setattr(obj, attr, val.item() if val.ndim == 0 else val)

                env.reset()
            except Exception as e:
                print(f"    [{base_name}] ep {episode_id}: scene reset FAILED: {e}", flush=True)
                continue

            # Sort transition steps for this episode
            transition_steps = sorted(int(k) for k in transitions.keys())
            applied_transitions = set()

            step_indices = [0] + list(range(sample_interval, n_steps, sample_interval))
            if step_indices[-1] != n_steps - 1:
                step_indices.append(n_steps - 1)

            ep_step_count = 0
            for step_idx in step_indices:
                try:
                    # Apply any transitions that should have occurred before this step
                    scene = og.sim.scenes[0]
                    for t_step in transition_steps:
                        if t_step > step_idx:
                            break
                        if t_step in applied_transitions:
                            continue
                        applied_transitions.add(t_step)
                        cur_transitions = transitions[str(t_step)]
                        for add_sys_name in cur_transitions["systems"]["add"]:
                            scene.get_system(add_sys_name, force_init=True)
                        for remove_sys_name in cur_transitions["systems"]["remove"]:
                            scene.clear_system(remove_sys_name)
                        for remove_obj_name in cur_transitions["objects"]["remove"]:
                            obj = scene.object_registry("name", remove_obj_name)
                            if obj is not None:
                                scene.remove_object(obj)
                        for j, add_obj_info in enumerate(cur_transitions["objects"]["add"]):
                            obj = create_object_from_init_info(add_obj_info)
                            scene.add_object(obj)
                            obj.set_position(th.ones(3) * 100.0 + th.ones(3) * 5 * j)
                        og.sim.step()

                    og.sim.load_state(state[step_idx, :int(state_size[step_idx])], serialized=True)
                    for _ in range(sim_steps):
                        og.sim.step()

                    predicates = get_predicate_states_hierarchical(env)
                except Exception as e:
                    print(f"    [{base_name}] ep {episode_id} step {step_idx}: FAILED ({e})", flush=True)
                    continue

                total_count = len(predicates)
                satisfied_count = sum(
                    1 for p in predicates if p.get("satisfied", p.get("_satisfied", False))
                )
                progress_val = satisfied_count / total_count if total_count > 0 else 0.0
                done = (satisfied_count == total_count) and total_count > 0

                record = {
                    "episode_id": episode_id,
                    "step": step_idx,
                    "task_name": task_name,
                    "progress": progress_val,
                    "done": done,
                    "satisfied_count": satisfied_count,
                    "total_count": total_count,
                    "predicates": predicates,
                }
                all_records.append(record)

                # Write immediately and flush — survives process crash
                jsonl_f.write(json.dumps(record, cls=TorchEncoder) + '\n')
                jsonl_f.flush()
                ep_step_count += 1

            print(f"    [{base_name}] ep {episode_id}: {n_steps} steps ({ep_step_count}/{len(step_indices)} samples ok)", flush=True)

    jsonl_f.close()

    df = pd.DataFrame(all_records)
    df.to_parquet(parquet_path, index=False)

    print(f"  Saved: {parquet_path}")
    return str(parquet_path)


def replay_and_extract_predicates(hdf_input_path, output_dir=None, sim_steps=5, sample_interval=1, force=False):
    """
    Replay a single HDF5 file and extract predicate states at each timestep.
    Creates and tears down the environment for this single file.

    For processing multiple files from the same task, use replay_and_extract_task_batch() instead.
    """
    hdf_input_path = Path(hdf_input_path)

    if output_dir is None:
        output_dir = hdf_input_path.parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = hdf_input_path.stem
    jsonl_path = output_dir / f"{base_name}_predicates.jsonl"

    if not force and _is_already_complete(jsonl_path):
        print(f"Skipping (already complete): {hdf_input_path}", flush=True)
        return str(output_dir / f"{base_name}_predicates.parquet")

    tmp_output_path = output_dir / f"{base_name}_replay_tmp.hdf5"
    env = DataPlaybackWrapper.create_from_hdf5(
        input_path=str(hdf_input_path),
        output_path=str(tmp_output_path),
        robot_obs_modalities=[],
        n_render_iterations=1,
        only_successes=False,
        include_task=True,
        include_task_obs=False,
        include_robot_control=False,
        include_contacts=True,
    )

    result = _process_hdf5_with_env(env, hdf_input_path, output_dir, sim_steps, sample_interval, force)

    if tmp_output_path.exists():
        tmp_output_path.unlink()
    og.sim.stop()
    og.clear()

    return result


def replay_and_extract_task_batch(hdf5_files, output_dir, sim_steps=5, sample_interval=1, force=False):
    """
    Process multiple HDF5 files from the same task, loading the scene only once.

    This avoids the ~120-150s scene load overhead per file. For a task with 200
    episodes, this saves ~7-8 hours of startup time compared to per-file processing.

    Args:
        hdf5_files: List of HDF5 file paths (must all be from the same task)
        output_dir: Output directory for predicate files
        sim_steps: Number of simulation steps after loading state
        sample_interval: Sample every N frames
        force: Force re-processing
    """
    if not hdf5_files:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter out already-completed files (unless force)
    if not force:
        pending = []
        for f in hdf5_files:
            jsonl_path = output_dir / f"{Path(f).stem}_predicates.jsonl"
            if _is_already_complete(jsonl_path):
                print(f"Skipping (already complete): {f}", flush=True)
            else:
                pending.append(f)
        if not pending:
            print("All files already complete, nothing to do.")
            return
    else:
        pending = list(hdf5_files)

    first_file = pending[0]
    print(f"Loading scene from: {first_file}", flush=True)
    print(f"Total files to process: {len(pending)}", flush=True)

    tmp_output_path = output_dir / "batch_replay_tmp.hdf5"
    env = DataPlaybackWrapper.create_from_hdf5(
        input_path=str(first_file),
        output_path=str(tmp_output_path),
        robot_obs_modalities=[],
        n_render_iterations=1,
        only_successes=False,
        include_task=True,
        include_task_obs=False,
        include_robot_control=False,
        include_contacts=True,
    )

    task_name = env.task.activity_name if hasattr(env.task, 'activity_name') else "unknown"
    print(f"Task: {task_name} — scene loaded, processing {len(pending)} files")

    for i, hdf5_path in enumerate(pending):
        try:
            print(f"[{i+1}/{len(pending)}] {Path(hdf5_path).name}", flush=True)
            _process_hdf5_with_env(env, hdf5_path, output_dir, sim_steps, sample_interval, force)
        except Exception as e:
            print(f"Error processing {hdf5_path}: {e}")
            import traceback
            traceback.print_exc()

    if tmp_output_path.exists():
        tmp_output_path.unlink()
    og.sim.stop()
    og.clear()
    print(f"Task {task_name}: batch complete ({len(pending)} files)")


def main():
    parser = argparse.ArgumentParser(
        description="Replay HDF5 and extract predicate states",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single file
  python replay_extract_predicates.py --file demo.hdf5

  # Process all files in a task directory (scene loaded once)
  python replay_extract_predicates.py --task-dir /path/to/task-0004

  # Process directory (auto-groups by task subdirectory)
  python replay_extract_predicates.py --dir /path/to/rawdata

  # Filter by pattern
  python replay_extract_predicates.py --dir /path/to/demos --pattern task-0004
"""
    )
    parser.add_argument("--file", type=str, help="Single HDF5 file to process")
    parser.add_argument("--task-dir", type=str,
                        help="Task directory containing HDF5 files (scene loaded once)")
    parser.add_argument("--dir", type=str,
                        help="Directory containing task subdirs (auto-groups by task)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: same as input)")
    parser.add_argument("--pattern", type=str, default=None,
                        help="Only process files matching pattern")
    parser.add_argument("--sim-steps", type=int, default=5,
                        help="Number of simulation steps after loading state (default: 5)")
    parser.add_argument("--sample-interval", type=int, default=1,
                        help="Sample every N frames (default: 1 = every frame)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-processing even if output already exists")
    parser.add_argument("--start", type=int, default=0,
                        help="Start index for file list (for splitting across workers)")
    parser.add_argument("--count", type=int, default=0,
                        help="Number of files to process (0 = all remaining from --start)")

    args = parser.parse_args()

    def _slice_files(files):
        """Apply --start/--count slicing to a sorted file list."""
        files = sorted(files)
        if args.start > 0:
            files = files[args.start:]
        if args.count > 0:
            files = files[:args.count]
        return files

    if args.file:
        # Single file mode (original behavior)
        replay_and_extract_predicates(
            args.file, args.output_dir,
            sim_steps=args.sim_steps, sample_interval=args.sample_interval, force=args.force,
        )

    elif args.task_dir:
        # Task-batch mode: all HDF5s in one task dir, scene loaded once
        task_path = Path(args.task_dir)
        hdf5_files = sorted(f for f in task_path.glob("*.hdf5") if "_replay_tmp" not in f.name)
        if args.pattern:
            hdf5_files = [f for f in hdf5_files if args.pattern in str(f)]
        hdf5_files = _slice_files(hdf5_files)
        print(f"Found {len(hdf5_files)} HDF5 files in {task_path} (start={args.start}, count={args.count or 'all'})")
        output_dir = args.output_dir or str(task_path)
        replay_and_extract_task_batch(
            hdf5_files, output_dir,
            sim_steps=args.sim_steps, sample_interval=args.sample_interval, force=args.force,
        )

    elif args.dir:
        # Directory mode: auto-group by task subdirectory
        from collections import defaultdict
        dir_path = Path(args.dir)
        all_files = sorted(dir_path.rglob("*.hdf5"))
        if args.pattern:
            all_files = [f for f in all_files if args.pattern in str(f)]
        print(f"Found {len(all_files)} HDF5 files")

        # Group by parent directory (= task directory)
        task_groups = defaultdict(list)
        for f in all_files:
            task_groups[f.parent].append(f)

        print(f"Grouped into {len(task_groups)} tasks")
        for task_dir in sorted(task_groups.keys()):
            files = task_groups[task_dir]
            output_dir = args.output_dir or str(task_dir)
            print(f"\n{'='*60}")
            print(f"Task: {task_dir.name} ({len(files)} files)")
            print(f"{'='*60}")
            try:
                replay_and_extract_task_batch(
                    files, output_dir,
                    sim_steps=args.sim_steps, sample_interval=args.sample_interval, force=args.force,
                )
            except Exception as e:
                print(f"Error processing task {task_dir.name}: {e}")
                import traceback
                traceback.print_exc()
                try:
                    og.sim.stop()
                    og.clear()
                except Exception:
                    pass
    else:
        parser.print_help()
        print("\nError: Either --file, --task-dir, or --dir must be specified")
        return

    og.shutdown()
    print("\nDone!")


if __name__ == "__main__":
    main()
