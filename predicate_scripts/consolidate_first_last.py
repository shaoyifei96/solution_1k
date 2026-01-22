#!/usr/bin/env python3
"""
Consolidate first and last states from per-task predicate JSONL files.

Usage:
    python consolidate_first_last.py --num-tasks 20 --output first_last_states_20.jsonl
    python consolidate_first_last.py --num-tasks 50 --output first_last_states_50.jsonl
"""

import json
import argparse
from pathlib import Path


def consolidate_first_last(data_dir, num_tasks, output_file):
    """
    Combine first and last records from each task's predicate JSONL file.
    
    Args:
        data_dir: Directory containing task-XXXX subdirectories
        num_tasks: Number of tasks to process (0 to num_tasks-1)
        output_file: Output JSONL file path
    """
    data_dir = Path(data_dir)
    output_file = Path(output_file)
    
    records = []
    for task_id in range(num_tasks):
        task_dir = data_dir / f"task-{task_id:04d}"
        if not task_dir.exists():
            print(f"Skipping {task_dir} (not found)")
            continue
        
        # Find the JSONL file
        jsonl_files = list(task_dir.glob("*.jsonl"))
        if not jsonl_files:
            print(f"No JSONL in {task_dir}")
            continue
        
        jsonl_file = jsonl_files[0]
        
        # Read all records
        all_records = []
        with open(jsonl_file, 'r') as f:
            for line in f:
                all_records.append(json.loads(line))
        
        if len(all_records) == 0:
            print(f"Empty file: {jsonl_file}")
            continue
        
        # Get first and last
        first = all_records[0]
        last = all_records[-1]
        first['source_file'] = jsonl_file.name
        last['source_file'] = jsonl_file.name
        
        records.append(first)
        if first != last:
            records.append(last)
        
        print(f"Task {task_id:04d}: {len(all_records)} records, first step={first.get('step')}, last step={last.get('step')}")

    # Write combined output
    with open(output_file, 'w') as f:
        for rec in records:
            f.write(json.dumps(rec) + '\n')

    print(f"\nWrote {len(records)} records to {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description='Consolidate first/last states from predicate JSONL files')
    parser.add_argument('--data-dir', type=str, 
                        default='/vast/projects/kumar/lab/yishao/data/predicate_data',
                        help='Directory containing task-XXXX subdirectories')
    parser.add_argument('--num-tasks', type=int, default=20,
                        help='Number of tasks to process (default: 20)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSONL file (default: first_last_states_N.jsonl in data_dir)')
    args = parser.parse_args()
    
    if args.output is None:
        output_file = Path(args.data_dir) / f"first_last_states_{args.num_tasks}.jsonl"
    else:
        output_file = Path(args.output)
    
    consolidate_first_last(args.data_dir, args.num_tasks, output_file)


if __name__ == '__main__':
    main()
