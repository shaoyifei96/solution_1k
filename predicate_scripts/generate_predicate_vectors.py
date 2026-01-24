#!/usr/bin/env python3
"""
Generate predicate state vectors from JSONL predicate data.

Each dimension corresponds to one top-level predicate:
- atomic: Binary 0/1
- forall: count / threshold (normalized 0-1)
- exists: Depends on nested predicate structure
- not: Binary 0/1
- forpairs: count / threshold (normalized 0-1)
- or: Binary 0/1

Output format: numpy arrays compatible with JAX/PyTorch training.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def compute_predicate_value(pred: Dict[str, Any]) -> float:
    """
    Compute normalized value for a predicate (0.0 to 1.0).
    
    - Binary predicates: 0.0 or 1.0
    - Count predicates: count / threshold
    
    Args:
        pred: Predicate dict from JSONL
        
    Returns:
        Float value between 0.0 and 1.0
    """
    pred_type = pred.get("type", "unknown")
    
    if pred_type == "atomic":
        # Binary: check if the predicate is satisfied
        # For atomic predicates, we need to check _satisfied or infer from context
        # In the data, atomic predicates at top level use the satisfied field
        if "satisfied" in pred:
            return 1.0 if pred["satisfied"] else 0.0
        elif "_satisfied" in pred:
            return 1.0 if pred["_satisfied"] else 0.0
        else:
            # Atomic predicates don't always have satisfied - check parent context
            # For now, return 0.0 as default (will be overridden by parent)
            return 0.0
    
    elif pred_type == "forall":
        # Count-based: count / threshold
        count = pred.get("count", 0)
        threshold = pred.get("threshold", 1)
        if threshold == 0:
            return 1.0 if pred.get("satisfied", False) else 0.0
        return min(count / threshold, 1.0)
    
    elif pred_type == "exists":
        # Exists depends on nested structure
        # If it has count/threshold, use that
        count = pred.get("count", 0)
        threshold = pred.get("threshold", 1)
        total = pred.get("total", threshold)
        
        # Check if instances have nested quantifiers
        instances = pred.get("instances", {})
        if instances:
            # Check first instance for nested structure
            first_inst = next(iter(instances.values()), {})
            if isinstance(first_inst, dict) and "nested" in first_inst:
                nested = first_inst["nested"]
                nested_type = nested.get("type", "")
                
                # If nested is a quantifier, aggregate across instances
                if nested_type in ["forall", "forn"]:
                    # Sum up counts across all instances
                    total_count = 0
                    total_threshold = 0
                    for inst_data in instances.values():
                        if isinstance(inst_data, dict) and "nested" in inst_data:
                            n = inst_data["nested"]
                            total_count += n.get("count", 0)
                            total_threshold += n.get("threshold", 1)
                    if total_threshold == 0:
                        return 1.0 if pred.get("satisfied", False) else 0.0
                    return min(total_count / total_threshold, 1.0)
                elif nested_type == "atomic":
                    # Count satisfied instances
                    satisfied_count = sum(
                        1 for inst_data in instances.values()
                        if isinstance(inst_data, dict) and inst_data.get("satisfied", False)
                    )
                    if total == 0:
                        return 1.0 if pred.get("satisfied", False) else 0.0
                    return min(satisfied_count / threshold, 1.0)
        
        # Default: use count/threshold
        if threshold == 0:
            return 1.0 if pred.get("satisfied", False) else 0.0
        return min(count / threshold, 1.0)
    
    elif pred_type == "not":
        # Binary: satisfied means the negation holds
        return 1.0 if pred.get("satisfied", False) else 0.0
    
    elif pred_type == "forpairs":
        # Count-based: count / threshold
        count = pred.get("count", 0)
        threshold = pred.get("threshold", 1)
        if threshold == 0:
            return 1.0 if pred.get("satisfied", False) else 0.0
        return min(count / threshold, 1.0)
    
    elif pred_type == "fornpairs":
        # Count-based: count / threshold
        count = pred.get("count", 0)
        threshold = pred.get("threshold", 1)
        if threshold == 0:
            return 1.0 if pred.get("satisfied", False) else 0.0
        return min(count / threshold, 1.0)
    
    elif pred_type == "forn":
        # Count-based: count / threshold
        count = pred.get("count", 0)
        threshold = pred.get("threshold", 1)
        if threshold == 0:
            return 1.0 if pred.get("satisfied", False) else 0.0
        return min(count / threshold, 1.0)
    
    elif pred_type == "or":
        # Binary: at least one child satisfied
        return 1.0 if pred.get("satisfied", False) else 0.0
    
    elif pred_type == "and":
        # Binary: all children satisfied
        return 1.0 if pred.get("satisfied", False) else 0.0
    
    else:
        # Unknown type - use satisfied field if available
        if "satisfied" in pred:
            return 1.0 if pred["satisfied"] else 0.0
        elif "_satisfied" in pred:
            return 1.0 if pred["_satisfied"] else 0.0
        return 0.0


def clean_object_name(name: str) -> str:
    """
    Clean up object names by removing instance suffixes and formatting.
    
    Examples:
        "can__of__soda.n.01" -> "can of soda"
        "ashcan.n.01_1" -> "ashcan"
        "pizza.n.01" -> "pizza"
    """
    import re
    # Remove instance suffix like "_1", "_2" at the end
    name = re.sub(r'_\d+$', '', name)
    # Remove WordNet suffix like ".n.01", ".v.02"
    name = re.sub(r'\.[a-z]\.\d+$', '', name)
    # Replace double underscores with spaces
    name = name.replace('__', ' ')
    # Replace remaining underscores with spaces
    name = name.replace('_', ' ')
    return name


def get_predicate_name(pred: Dict[str, Any], idx: int) -> str:
    """
    Generate a human-readable name for a predicate.
    
    Args:
        pred: Predicate dict
        idx: Predicate index
        
    Returns:
        Short descriptive name
    """
    pred_type = pred.get("type", "unknown")
    raw_desc = pred.get("_raw_description", "")
    
    if pred_type == "atomic":
        pred_name = pred.get("predicate", "?")
        arg1 = clean_object_name(pred.get("arg1", ""))
        return f"{pred_name}({arg1})"
    
    elif pred_type == "forall":
        category = clean_object_name(pred.get("category", "?"))
        inner_pred = pred.get("predicate", "")
        args = pred.get("args", [])
        
        # Check if inner_pred is "not" - need to look at nested structure
        if inner_pred == "not":
            # Look at instances to find the actual atomic predicate
            instances = pred.get("instances", {})
            if instances:
                first_inst = next(iter(instances.values()), {})
                if isinstance(first_inst, dict) and "nested" in first_inst:
                    nested = first_inst["nested"]
                    if nested.get("type") == "not" and "child" in nested:
                        child = nested["child"]
                        actual_pred = child.get("predicate", "")
                        if actual_pred:
                            return f"forall({category})->not({actual_pred})"
            return f"forall({category})->not"
        
        # Check if inner_pred is "exists" - need to look at nested structure for actual predicate
        if inner_pred == "exists":
            instances = pred.get("instances", {})
            if instances:
                first_inst = next(iter(instances.values()), {})
                if isinstance(first_inst, dict) and "nested" in first_inst:
                    nested = first_inst["nested"]
                    if nested.get("type") == "exists":
                        nested_cat = clean_object_name(nested.get("category", ""))
                        # Go one level deeper to find atomic predicate
                        nested_insts = nested.get("instances", {})
                        if nested_insts:
                            inner_inst = next(iter(nested_insts.values()), {})
                            if isinstance(inner_inst, dict) and "nested" in inner_inst:
                                atomic = inner_inst["nested"]
                                if atomic.get("type") == "atomic":
                                    atomic_pred = atomic.get("predicate", "")
                                    return f"forall({category})->exists({nested_cat})->{atomic_pred}"
            return f"forall({category})->exists"
        
        if inner_pred and len(args) >= 2:
            target = clean_object_name(args[1])
            return f"forall({category})->{inner_pred}({target})"
        elif inner_pred:
            return f"forall({category})->{inner_pred}"
        return f"forall({category})"
    
    elif pred_type == "exists":
        category = clean_object_name(pred.get("category", "?"))
        # Check for nested predicate
        instances = pred.get("instances", {})
        if instances:
            first_inst = next(iter(instances.values()), {})
            if isinstance(first_inst, dict) and "nested" in first_inst:
                nested = first_inst["nested"]
                nested_type = nested.get("type", "")
                
                if nested_type == "atomic":
                    # Direct atomic predicate
                    nested_pred = nested.get("predicate", "")
                    arg1 = clean_object_name(nested.get("arg1", ""))
                    arg2 = clean_object_name(nested.get("arg2", ""))
                    if arg2:
                        return f"exists({category})->{nested_pred}({arg1},{arg2})"
                    elif arg1:
                        return f"exists({category})->{nested_pred}({arg1})"
                    return f"exists({category})->{nested_pred}"
                    
                elif nested_type in ["forall", "forn"]:
                    # Nested forall quantifier - e.g., exists(fridge)->forall(pizza)->inside
                    nested_cat = clean_object_name(nested.get("category", ""))
                    nested_pred = nested.get("predicate", "")
                    nested_args = nested.get("args", [])
                    if nested_pred and len(nested_args) >= 2:
                        target = clean_object_name(nested_args[1])
                        return f"exists({category})->forall({nested_cat})->{nested_pred}({target})"
                    elif nested_pred:
                        return f"exists({category})->forall({nested_cat})->{nested_pred}"
                    return f"exists({category})->forall({nested_cat})"
        return f"exists({category})"
    
    elif pred_type == "not":
        child = pred.get("child", {})
        if child.get("type") == "atomic":
            child_pred = child.get("predicate", "?")
            child_arg = clean_object_name(child.get("arg1", ""))
            return f"not({child_pred}({child_arg}))"
        return f"not(...)"
    
    elif pred_type == "forpairs":
        cat1 = clean_object_name(pred.get("category1", "?"))
        cat2 = clean_object_name(pred.get("category2", "?"))
        # Extract actual predicate from raw description
        raw_desc = pred.get("_raw_description", "")
        # e.g. "forpairs pizza.n.01 - pizza.n.01 plate.n.04 - plate.n.04 ontop pizza.n.01 plate.n.04"
        # The predicate is typically the word before the last two category mentions
        parts = raw_desc.split()
        pred_name = ""
        if len(parts) >= 3:
            # Look for predicate name (typically after the second "-")
            for i, p in enumerate(parts):
                if p in ["ontop", "inside", "under", "nextto", "touching", "onfloor"]:
                    pred_name = p
                    break
        if pred_name:
            return f"forpairs({cat1},{cat2})->{pred_name}"
        return f"forpairs({cat1},{cat2})"
    
    elif pred_type == "or":
        return f"or(...)"
    
    else:
        return f"pred_{idx}"


def is_binary_predicate(pred: Dict[str, Any]) -> bool:
    """Check if predicate produces binary output."""
    pred_type = pred.get("type", "unknown")
    return pred_type in ["atomic", "not", "or", "and"]


def load_episode_predicates(jsonl_path: Path) -> List[Dict[str, Any]]:
    """
    Load all timestep records from a predicate JSONL file.
    
    Args:
        jsonl_path: Path to JSONL file
        
    Returns:
        List of records, each containing step, predicates, etc.
    """
    records = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def generate_predicate_vectors_for_episode(
    records: List[Dict[str, Any]]
) -> Tuple[np.ndarray, List[str], List[int], Dict[str, Any]]:
    """
    Generate state vectors from episode records.
    
    Args:
        records: List of timestep records from JSONL
        
    Returns:
        - states: np.ndarray of shape (num_timesteps, num_predicates)
        - predicate_names: List of predicate names
        - timesteps: List of timestep indices
        - metadata: Dict with task info
    """
    if not records:
        return np.array([]), [], [], {}
    
    # Get predicate structure from first record
    first_record = records[0]
    predicates = first_record.get("predicates", [])
    num_predicates = len(predicates)
    
    # Generate predicate names
    predicate_names = [get_predicate_name(pred, i) for i, pred in enumerate(predicates)]
    
    # Determine which are binary vs count-based
    is_binary = [is_binary_predicate(pred) for pred in predicates]
    
    # Extract timesteps and values
    timesteps = []
    state_vectors = []
    
    for record in records:
        step = record.get("step", 0)
        preds = record.get("predicates", [])
        
        # Compute value for each predicate
        values = []
        for pred in preds:
            val = compute_predicate_value(pred)
            values.append(val)
        
        # Pad if predicates changed (shouldn't happen, but safety)
        while len(values) < num_predicates:
            values.append(0.0)
        values = values[:num_predicates]
        
        timesteps.append(step)
        state_vectors.append(values)
    
    states = np.array(state_vectors, dtype=np.float32)
    
    # Metadata
    metadata = {
        "task_name": first_record.get("task_name", "unknown"),
        "episode_id": first_record.get("episode_id", 0),
        "num_predicates": num_predicates,
        "predicate_names": predicate_names,
        "is_binary": is_binary,
        "num_timesteps": len(timesteps),
        "timesteps": timesteps,
    }
    
    return states, predicate_names, timesteps, metadata


def visualize_predicate_vectors(
    states: np.ndarray,
    predicate_names: List[str],
    timesteps: List[int],
    metadata: Dict[str, Any],
    output_path: Path,
    task_id: Optional[int] = None
):
    """
    Visualize predicate vectors as a heatmap.
    
    Args:
        states: np.ndarray of shape (num_timesteps, num_predicates)
        predicate_names: List of predicate names
        timesteps: List of timestep indices
        metadata: Dict with task info
        output_path: Path to save visualization
        task_id: Optional task ID for filename
    """
    if states.size == 0:
        print("No data to visualize")
        return
    
    num_timesteps, num_predicates = states.shape
    task_name = metadata.get("task_name", "unknown")
    episode_id = metadata.get("episode_id", 0)
    
    # Create figure
    fig_height = max(4, num_predicates * 0.5 + 2)
    fig, ax = plt.subplots(1, 1, figsize=(16, fig_height))
    
    # Transpose for visualization: predicates on y-axis, time on x-axis
    states_display = states.T
    
    # Create custom white-to-green colormap
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#FFFFFF', '#90EE90', '#32CD32', '#228B22', '#006400']  # white -> light green -> lime -> forest -> dark green
    cmap = LinearSegmentedColormap.from_list('white_to_green', colors, N=256)
    
    # Create heatmap
    im = ax.imshow(states_display, aspect='auto', cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
    
    # Set y-axis labels (predicate names, shortened)
    short_names = [name[:40] for name in predicate_names]
    ax.set_yticks(range(num_predicates))
    ax.set_yticklabels(short_names, fontsize=9)
    
    # Set x-axis (timesteps)
    num_ticks = min(10, num_timesteps)
    if num_timesteps > 1:
        tick_indices = np.linspace(0, num_timesteps - 1, num_ticks, dtype=int)
        tick_labels = [str(timesteps[i]) for i in tick_indices]
        ax.set_xticks(tick_indices)
        ax.set_xticklabels(tick_labels, fontsize=9)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
    cbar.set_label('Predicate Value (0=not satisfied, 1=satisfied)', fontsize=10)
    
    # Labels and title
    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('Predicate', fontsize=12)
    
    task_suffix = f"task{task_id:04d}_" if task_id is not None else ""
    ax.set_title(f'{task_suffix}{task_name} (Episode {episode_id})\nPredicate State Vectors', 
                 fontsize=12, weight='bold')
    
    # Add grid lines
    ax.set_xticks(np.arange(-0.5, num_timesteps, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, num_predicates, 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved visualization to {output_path}")


def process_task(
    task_dir: Path,
    output_dir: Path,
    task_id: int,
    num_episodes: int = 1,
    viz_only_first: bool = True
) -> Dict[str, Any]:
    """
    Process episodes for a task.
    
    Args:
        task_dir: Path to task-XXXX directory
        output_dir: Output directory for vectors
        task_id: Task ID number
        num_episodes: Number of episodes to process
        viz_only_first: Only visualize first episode
        
    Returns:
        Dict with processing results
    """
    # Find all JSONL files
    jsonl_files = sorted(task_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"  No JSONL files found in {task_dir}")
        return {"status": "no_data", "task_id": task_id}
    
    # Limit to requested number of episodes
    files_to_process = jsonl_files[:num_episodes]
    
    task_name = None
    episodes_processed = []
    total_predicates = 0
    total_timesteps = 0
    
    # Create output directory
    task_output_dir = output_dir / f"task-{task_id:04d}"
    task_output_dir.mkdir(parents=True, exist_ok=True)
    
    for ep_idx, jsonl_file in enumerate(files_to_process):
        episode_name = jsonl_file.stem  # e.g., episode_00020010_predicates
        
        print(f"  Processing {jsonl_file.name}...")
        
        # Load and process
        records = load_episode_predicates(jsonl_file)
        if not records:
            print(f"  Empty file: {jsonl_file}")
            continue
        
        states, predicate_names, timesteps, metadata = generate_predicate_vectors_for_episode(records)
        
        # Add task_id to metadata
        metadata["task_id"] = task_id
        metadata["source_file"] = str(jsonl_file)
        
        if task_name is None:
            task_name = metadata["task_name"]
        
        # Save as numpy arrays
        npz_path = task_output_dir / f"{episode_name}_vectors.npz"
        np.savez(
            npz_path,
            states=states,
            timesteps=np.array(timesteps, dtype=np.int32),
            predicate_names=np.array(predicate_names, dtype=object),
            is_binary=np.array(metadata["is_binary"], dtype=bool),
        )
        print(f"  Saved vectors to {npz_path}")
        
        # Save metadata as JSON
        meta_path = task_output_dir / f"{episode_name}_metadata.json"
        # Convert numpy types for JSON serialization
        meta_json = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in metadata.items()}
        with open(meta_path, 'w') as f:
            json.dump(meta_json, f, indent=2)
        
        # Visualize (first episode only, or all if requested)
        if viz_only_first:
            viz_dir = output_dir / "predicate_viz"
            viz_dir.mkdir(parents=True, exist_ok=True)
            viz_path = viz_dir / f"predicate_vectors_task{task_id:04d}_{episode_name}.png"
            visualize_predicate_vectors(states, predicate_names, timesteps, metadata, viz_path, task_id)
        
        episodes_processed.append(episode_name)
        total_predicates = metadata["num_predicates"]
        total_timesteps += metadata["num_timesteps"]
    
    if not episodes_processed:
        return {"status": "empty", "task_id": task_id}
    
    return {
        "status": "success",
        "task_id": task_id,
        "task_name": task_name,
        "num_predicates": total_predicates,
        "num_episodes": len(episodes_processed),
        "total_timesteps": total_timesteps,
        "episodes": episodes_processed,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate predicate state vectors from JSONL data")
    parser.add_argument("--data-dir", type=str,
                        default="/vast/projects/kumar/lab/yishao/data/predicate_data",
                        help="Directory containing task-XXXX subdirectories")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: data_dir/predicate_vectors)")
    parser.add_argument("--task-id", type=int, default=None,
                        help="Process only this task ID (default: process first available)")
    parser.add_argument("--all-tasks", action="store_true",
                        help="Process all available tasks")
    parser.add_argument("--num-episodes", type=int, default=1,
                        help="Number of episodes to process per task (default: 1)")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip visualization")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / "predicate_vectors"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("PREDICATE VECTOR GENERATION")
    print("="*80)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    
    # Find task directories
    task_dirs = sorted(data_dir.glob("task-????"))
    if not task_dirs:
        print("No task directories found!")
        return
    
    print(f"Found {len(task_dirs)} task directories")
    
    # Determine which tasks to process
    if args.task_id is not None:
        # Process specific task
        task_dir = data_dir / f"task-{args.task_id:04d}"
        if not task_dir.exists():
            print(f"Task directory not found: {task_dir}")
            return
        tasks_to_process = [(args.task_id, task_dir)]
    elif args.all_tasks:
        # Process all tasks
        tasks_to_process = []
        for td in task_dirs:
            tid = int(td.name.split("-")[1])
            tasks_to_process.append((tid, td))
    else:
        # Process first task only
        td = task_dirs[0]
        tid = int(td.name.split("-")[1])
        tasks_to_process = [(tid, td)]
    
    print(f"\nProcessing {len(tasks_to_process)} task(s)...")
    
    results = []
    for task_id, task_dir in tasks_to_process:
        print(f"\n--- Task {task_id:04d} ---")
        result = process_task(
            task_dir, 
            output_dir, 
            task_id,
            num_episodes=args.num_episodes,
            viz_only_first=not args.no_viz
        )
        results.append(result)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    successful = [r for r in results if r.get("status") == "success"]
    print(f"Successfully processed: {len(successful)}/{len(results)} tasks")
    
    for r in successful:
        num_eps = r.get('num_episodes', 1)
        total_ts = r.get('total_timesteps', r.get('num_timesteps', 0))
        print(f"  Task {r['task_id']:04d} ({r['task_name']}): "
              f"{r['num_predicates']} predicates, {num_eps} episode(s), {total_ts} total timesteps")
    
    print(f"\nOutput saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
