"""Fixed V2 JSONL → .pkl consolidator.

Bug in the original `consolidate_jsonl_to_pkl.py`:
    states[t] = record t's value, indexed by RECORD POSITION (0..N-1)

But `transforms.py::ComputePredicateStateFromData` does:
    frame_idx = int(timestamp * 30.0)   # = sim step at 30 Hz
    states[frame_idx]                    # assumes SIM-STEP indexing

The records were extracted with sample_interval=20 (every 20 sim steps),
so the original consolidator collapsed ~6,400 sim steps into ~320 rows,
and training-time lookups got wildly wrong values then clamped to the
last row (= "all done") for ~95% of every episode.

This fixed version uses each record's `step` field as the actual sim
step index, expanding sparse samples into a dense per-sim-step array
by step-function interpolation (each sample fills the interval until
the next sample's step).

Output goes to a SEPARATE directory so the broken pkl is preserved
for diffing / sanity comparison.

Usage:
    python consolidate_jsonl_to_pkl_fixed.py \\
        --input-dir /vast/projects/kumar/lab/yishao/data/predicate_data \\
        --output-dir /vast/projects/kumar/lab/yishao/b1k_2/data/predicate_data_v2_fixed
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

# Reuse helpers from the original module so we don't drift.
from consolidate_jsonl_to_pkl import compute_predicate_value, get_predicate_name


def process_task_fixed(task_dir: Path) -> dict:
    """Process all JSONL files in a task directory using STEP-INDEXED states arrays."""
    jsonl_files = sorted(task_dir.glob("*_predicates.jsonl"))
    if not jsonl_files:
        return None

    demo_vectors = {}
    predicate_names = None
    predicate_types = None
    is_binary = None
    num_predicates = None

    # Stats for sanity reporting
    sample_interval_observed = []
    n_records_per_ep = []
    n_simsteps_per_ep = []

    for jsonl_path in jsonl_files:
        fname = jsonl_path.stem.replace("_predicates", "")
        ep_id_str = fname.replace("episode_", "")

        records = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        if not records:
            continue

        # Initialize predicate metadata from first file's first record
        if predicate_names is None:
            predicates_template = records[0]["predicates"]
            num_predicates = len(predicates_template)
            predicate_names = []
            predicate_types = []
            is_binary = []
            for p in predicates_template:
                predicate_names.append(get_predicate_name(p))
                ptype = p.get("type", "atomic")
                predicate_types.append(ptype)
                is_binary.append(ptype == "atomic")

        # Sort records by their original sim step
        records.sort(key=lambda r: r["step"])

        # Validate: every record should have a 'step' field
        if any("step" not in r for r in records):
            print(f"  WARNING: {jsonl_path.name} has records without 'step' field, skipping")
            continue

        # Compute the dense sim-step length: last sample's step + interval to next.
        # We don't know the actual episode length from the JSONL alone, so we
        # extrapolate: assume the last sample covers the same interval as the
        # second-to-last → last gap. This may slightly under-cover the very last
        # frames, but training-time lookups past the end will still clamp to
        # the LAST sample's value (which is the latest known truth), not to a
        # random earlier sample.
        steps = [r["step"] for r in records]
        if len(steps) >= 2:
            last_gap = steps[-1] - steps[-2]
            T_full = steps[-1] + max(1, last_gap)
        else:
            T_full = steps[-1] + 1

        # Track stats
        if len(steps) >= 2:
            sample_interval_observed.append(steps[1] - steps[0])
        n_records_per_ep.append(len(records))
        n_simsteps_per_ep.append(T_full)

        # Build dense states array: each record fills [step_k, step_{k+1})
        states = np.zeros((T_full, num_predicates), dtype=np.float32)
        for k, rec in enumerate(records):
            start = rec["step"]
            end = records[k + 1]["step"] if k + 1 < len(records) else T_full
            vals = np.zeros(num_predicates, dtype=np.float32)
            for i, pred in enumerate(rec["predicates"]):
                if i < num_predicates:
                    vals[i] = compute_predicate_value(pred)
            states[start:end] = vals

        demo_vectors[ep_id_str] = {
            "states": states,
            "actions": np.zeros_like(states),
        }

    if not demo_vectors or predicate_names is None:
        return None

    item_to_index = {name: i for i, name in enumerate(predicate_names)}
    index_to_item = {i: name for i, name in enumerate(predicate_names)}

    # Sanity stats
    stats = {
        "n_episodes": len(demo_vectors),
        "n_records_mean": float(np.mean(n_records_per_ep)) if n_records_per_ep else 0,
        "n_simsteps_mean": float(np.mean(n_simsteps_per_ep)) if n_simsteps_per_ep else 0,
        "sample_interval_mode": int(np.bincount(sample_interval_observed).argmax()) if sample_interval_observed else 0,
    }

    return {
        "demo_vectors": demo_vectors,
        "num_items": num_predicates,
        "item_to_index": item_to_index,
        "index_to_item": index_to_item,
        "predicate_types": predicate_types,
        "is_binary": is_binary,
        "_fix_metadata": {
            "consolidator_version": "fixed-step-indexed",
            "stats": stats,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="FIXED V2 JSONL→.pkl consolidator (step-indexed states)")
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory containing task-XXXX subdirs with JSONL files")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for .pkl files")
    parser.add_argument("--only-tasks", type=str, default=None,
                        help="Comma-separated list of task IDs to process (default: all)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    only_tasks = None
    if args.only_tasks:
        only_tasks = set(int(x) for x in args.only_tasks.split(","))

    task_dirs = sorted(input_dir.glob("task-*"))
    if only_tasks is not None:
        task_dirs = [d for d in task_dirs if int(d.name.split("-")[1]) in only_tasks]

    print(f"Found {len(task_dirs)} task directories to process")
    print(f"Output: {output_dir}")
    print()

    processed = 0
    for task_dir in task_dirs:
        task_id_str = task_dir.name
        task_id = int(task_id_str.split("-")[1])

        result = process_task_fixed(task_dir)
        if result is None:
            print(f"  {task_id_str}: no data, skipping")
            continue

        pkl_path = output_dir / f"task_{task_id:04d}_state_action_vectors.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(result, f)

        s = result["_fix_metadata"]["stats"]
        print(f"  {task_id_str}: {s['n_episodes']} eps, "
              f"{result['num_items']} preds, "
              f"sample_interval≈{s['sample_interval_mode']}, "
              f"records/ep≈{s['n_records_mean']:.0f}, "
              f"sim_steps/ep≈{s['n_simsteps_mean']:.0f}")
        processed += 1

    print()
    print(f"Done. {processed} task pkl files written to {output_dir}")


if __name__ == "__main__":
    main()
