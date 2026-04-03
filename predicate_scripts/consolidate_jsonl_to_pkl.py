"""
Consolidate V2 JSONL predicate extraction output into .pkl files for training.

Reads JSONL files from /pool/yishao/v2_extract/predicates/task-XXXX/
Outputs task_XXXX_state_action_vectors.pkl matching PredicateDataStore format.

Each .pkl contains:
  {
    'demo_vectors': {
      episode_id_str: {
        'states': np.ndarray [T, num_predicates] float32,  # progress values (0-1)
        'actions': np.ndarray [T, num_predicates] float32,  # zeros (unused)
      },
      ...
    },
    'num_items': int,
    'item_to_index': {name: idx},
    'index_to_item': {idx: name},
    'predicate_types': [str],   # 'atomic'/'forall'/'exists' per predicate
    'is_binary': [bool],        # True for atomic, False for forall/exists
  }

Usage:
  python consolidate_jsonl_to_pkl.py \
    --input-dir /pool/yishao/v2_extract/predicates \
    --output-dir /pool/yishao/v2_extract/predicate_data
"""

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np


def compute_predicate_value(pred: dict) -> float:
    """Compute continuous predicate value from JSONL predicate record.

    - atomic: 1.0 if satisfied, 0.0 if not
    - forall/exists: count / threshold (capped at 1.0)
    """
    ptype = pred.get("type", "atomic")
    if ptype in ("forall", "exists", "forpairs"):
        count = pred.get("count", 0)
        threshold = pred.get("threshold", 1)
        if threshold <= 0:
            return 1.0 if pred.get("satisfied", False) else 0.0
        return min(count / threshold, 1.0)
    else:
        return 1.0 if pred.get("satisfied", False) else 0.0


def get_predicate_name(pred: dict) -> str:
    """Extract human-readable predicate name from JSONL record."""
    raw = pred.get("_raw_description", "")
    if raw:
        # Extract the core predicate verb (inside, ontop, etc.)
        parts = raw.split()
        for keyword in ["inside", "ontop", "nextto", "under", "attached",
                        "open", "cooked", "covered", "real", "contains",
                        "onfloor", "toggled_on", "saturated", "filled",
                        "overlaid", "draped", "folded"]:
            if keyword in parts:
                return keyword
    # Fallback: use type + category
    cat = pred.get("category", "")
    ptype = pred.get("type", "atomic")
    if cat:
        return f"{ptype}_{cat}"
    return f"pred_{pred.get('type', 'unknown')}"


def process_task(task_dir: Path) -> dict:
    """Process all JSONL files in a task directory into a single dict."""
    jsonl_files = sorted(task_dir.glob("*_predicates.jsonl"))
    if not jsonl_files:
        return None

    # Group records by episode
    episodes = defaultdict(list)

    for jsonl_path in jsonl_files:
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ep_id = record["episode_id"]
                episodes[ep_id].append(record)

    if not episodes:
        return None

    # Determine predicate structure from first episode's first record
    first_ep = next(iter(episodes.values()))
    first_record = first_ep[0]
    predicates_template = first_record["predicates"]
    num_predicates = len(predicates_template)

    # Build predicate names, types, is_binary
    predicate_names = []
    predicate_types = []
    is_binary = []
    for p in predicates_template:
        predicate_names.append(get_predicate_name(p))
        ptype = p.get("type", "atomic")
        predicate_types.append(ptype)
        is_binary.append(ptype == "atomic")

    item_to_index = {name: i for i, name in enumerate(predicate_names)}
    index_to_item = {i: name for i, name in enumerate(predicate_names)}

    # Build demo_vectors
    demo_vectors = {}

    for ep_id, records in episodes.items():
        # Sort by step
        records.sort(key=lambda r: r["step"])

        T = len(records)
        states = np.zeros((T, num_predicates), dtype=np.float32)

        for t, record in enumerate(records):
            for i, pred in enumerate(record["predicates"]):
                if i < num_predicates:
                    states[t, i] = compute_predicate_value(pred)

        # Episode ID: extract from JSONL filename or use formatted ID
        # Files are named like episode_00020010_predicates.jsonl
        # The episode_id in the record is the within-file index (0, 1, ...)
        # We need to match PredicateDataStore's lookup format
        jsonl_file = [f for f in jsonl_files if any(
            r["episode_id"] == ep_id for r in records[:1]
        )]
        if jsonl_file:
            # Use the numeric part from filename: episode_00020010 -> "00020010"
            fname = jsonl_file[0].stem.replace("_predicates", "")
            ep_id_str = fname.replace("episode_", "")
        else:
            ep_id_str = f"{ep_id:08d}"

        demo_vectors[ep_id_str] = {
            "states": states,
            "actions": np.zeros_like(states),  # unused placeholder
        }

    return {
        "demo_vectors": demo_vectors,
        "num_items": num_predicates,
        "item_to_index": item_to_index,
        "index_to_item": index_to_item,
        "predicate_types": predicate_types,
        "is_binary": is_binary,
    }


def main():
    parser = argparse.ArgumentParser(description="Consolidate V2 JSONL to .pkl for training")
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory containing task-XXXX subdirs with JSONL files")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for .pkl files")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_dirs = sorted(input_dir.glob("task-*"))
    print(f"Found {len(task_dirs)} task directories")

    total_episodes = 0
    task_pred_counts = {}

    for task_dir in task_dirs:
        task_id_str = task_dir.name  # e.g., "task-0002"
        task_id = int(task_id_str.split("-")[1])

        result = process_task(task_dir)
        if result is None:
            print(f"  {task_id_str}: no data, skipping")
            continue

        num_eps = len(result["demo_vectors"])
        num_preds = result["num_items"]
        total_episodes += num_eps
        task_pred_counts[task_id] = num_preds

        # Save .pkl
        pkl_path = output_dir / f"task_{task_id:04d}_state_action_vectors.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(result, f)

        print(f"  {task_id_str}: {num_eps} episodes, {num_preds} predicates → {pkl_path.name}")

    print(f"\nTotal: {total_episodes} episodes across {len(task_pred_counts)} tasks")

    # Print predicate counts for config verification
    print("\n--- Predicate counts (for pi_behavior_config.py) ---")
    for task_id in sorted(task_pred_counts.keys()):
        print(f"  task {task_id:04d}: {task_pred_counts[task_id]} predicates")

    # Compare with V1 TASK_NUM_PREDICATES
    v1_counts = (1, 4, 6, 4, 6, 4, 4, 6, 3, 7, 5, 8, 6, 3, 3, 3, 2, 2, 3, 5,
                 15, 7, 4, 7, 8, 4, 20, 6, 6, 10, 4, 2, 2, 4, 1, 1, 1, 1, 1, 1,
                 1, 5, 3, 6, 5, 2, 2, 4, 7, 8)

    print("\n--- V1 vs V2 predicate count comparison ---")
    mismatches = []
    for task_id, v2_count in sorted(task_pred_counts.items()):
        if task_id < len(v1_counts):
            v1_count = v1_counts[task_id]
            match = "✓" if v1_count == v2_count else "✗ MISMATCH"
            if v1_count != v2_count:
                mismatches.append((task_id, v1_count, v2_count))
            print(f"  task {task_id:04d}: V1={v1_count}, V2={v2_count} {match}")

    if mismatches:
        print(f"\n⚠️  {len(mismatches)} mismatches found! TASK_NUM_PREDICATES needs updating.")
    else:
        print("\n✓ All counts match V1. No config changes needed.")


if __name__ == "__main__":
    main()
