#!/usr/bin/env python3
"""
Delete predicate JSONL files that failed to complete (done != True in final record).
"""

import argparse
import json
from pathlib import Path


def check_and_delete_failed(predicate_dir: Path, dry_run: bool = True) -> tuple[int, int]:
    """
    Check all JSONL files in directory and delete those that failed.
    
    Returns:
        Tuple of (deleted_count, total_count)
    """
    jsonl_files = sorted(predicate_dir.rglob("*.jsonl"))
    deleted_count = 0
    failed_files = []
    
    for jsonl_file in jsonl_files:
        try:
            with open(jsonl_file, 'r') as f:
                lines = f.readlines()
            
            if not lines:
                # Empty file - consider as failed
                failed_files.append((jsonl_file, "empty"))
                continue
                
            # Check last record
            last_record = json.loads(lines[-1])
            if not last_record.get("done", False):
                failed_files.append((jsonl_file, f"done={last_record.get('done', 'missing')}"))
                
        except Exception as e:
            failed_files.append((jsonl_file, f"error: {e}"))
    
    # Delete or report failed files
    for jsonl_file, reason in failed_files:
        if dry_run:
            print(f"[DRY RUN] Would delete: {jsonl_file} ({reason})")
        else:
            print(f"Deleting: {jsonl_file} ({reason})")
            jsonl_file.unlink()
            deleted_count += 1
    
    return len(failed_files) if dry_run else deleted_count, len(jsonl_files)


def main():
    parser = argparse.ArgumentParser(description="Delete failed predicate JSONL files")
    parser.add_argument("--dir", type=str, required=True,
                        help="Directory containing predicate JSONL files")
    parser.add_argument("--delete", action="store_true",
                        help="Actually delete files (default is dry run)")
    
    args = parser.parse_args()
    
    predicate_dir = Path(args.dir)
    if not predicate_dir.exists():
        print(f"Error: Directory {predicate_dir} does not exist")
        return
    
    dry_run = not args.delete
    if dry_run:
        print("=== DRY RUN MODE (use --delete to actually delete) ===\n")
    
    failed_count, total_count = check_and_delete_failed(predicate_dir, dry_run)
    
    print(f"\n{'Would delete' if dry_run else 'Deleted'}: {failed_count}/{total_count} files")
    print(f"Remaining: {total_count - failed_count} files")


if __name__ == "__main__":
    main()
