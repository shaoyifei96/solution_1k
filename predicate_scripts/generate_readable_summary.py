#!/usr/bin/env python3
"""
Generate a readable summary from the predicate JSONL file,
including both quantified predicates (with count/threshold) and simple predicates (just True/False).
"""

import json
import re
from collections import defaultdict

INPUT_FILE = "/vast/projects/kumar/lab/yishao/data/predicate_data/first_last_states_20.jsonl"
OUTPUT_FILE = "/vast/projects/kumar/lab/yishao/data/predicate_data/first_last_states_20_readable_v3.txt"


def normalize_predicate_key(key):
    """
    Normalize predicate keys for duplicate detection.
    Handles formats like:
    - 'not open toolbox.n.01_1'
    - "not(['open', '?toolbox.n.01_1'])"
    - 'inside(obj1, obj2)'
    """
    # Remove all special characters except alphanumeric, underscore, dot
    s = key.lower()
    # Remove 'not' prefix but track it
    is_negated = False
    if s.startswith('not'):
        is_negated = True
        s = s[3:]  # remove 'not'
    
    # Extract alphanumeric parts and object identifiers (like toolbox.n.01_1)
    parts = re.findall(r'[a-z][a-z0-9_.]*', s)
    # Remove the '?' prefix from variables
    parts = [p.lstrip('?') for p in parts]
    parts = [p for p in parts if p]  # remove empty strings
    
    result = ('not_' if is_negated else '') + '_'.join(sorted(parts))
    return result


def main():
    # Load all entries
    entries = []
    with open(INPUT_FILE, 'r') as f:
        for line in f:
            entries.append(json.loads(line.strip()))
    
    # Group by task
    task_entries = defaultdict(list)
    for entry in entries:
        task_entries[entry['task_name']].append(entry)
    
    output_lines = []
    
    for task_name, task_data in task_entries.items():
        for entry in task_data:
            # Header line
            header = f"=== {task_name} | step={entry['step']} | progress={entry['progress']} | done={entry['done']} ==="
            output_lines.append(header)
            
            # Satisfied/total counts
            sat_count = entry.get('satisfied_count', 0)
            total_count = entry.get('total_count', 0)
            output_lines.append(f"    satisfied_count={sat_count}, total_count={total_count}")
            
            # Extract predicates - group by base name
            predicates = {}
            seen_quantified_bases = set()  # Track base names that have quantified versions
            
            # First pass: identify all quantified predicates (those with :: in key)
            for key, value in entry.items():
                if key in ['episode_id', 'step', 'task_name', 'progress', 'done', 'satisfied_count', 'total_count']:
                    continue
                
                if '::' in key:
                    base_name, field = key.rsplit('::', 1)
                    if base_name not in predicates:
                        predicates[base_name] = {'type': 'quantified', 'fields': {}}
                    predicates[base_name]['fields'][field] = value
                    seen_quantified_bases.add(base_name)
            
            # Second pass: add simple predicates (skip duplicates)
            seen_simple = set()
            for key, value in entry.items():
                if key in ['episode_id', 'step', 'task_name', 'progress', 'done', 'satisfied_count', 'total_count']:
                    continue
                
                if '::' not in key and isinstance(value, bool):
                    # Skip if this key starts with forall/exists/forn/forpairs and there's a quantified version
                    key_lower = key.lower().replace(' ', '')
                    is_redundant = False
                    
                    # Check if it's an alternate format for a quantified predicate
                    for qbase in seen_quantified_bases:
                        # The quantified version has format: forall_[...](...) 
                        # The simple version has format: forall [...] or forall(...)
                        if key_lower.startswith('forall') or key_lower.startswith('exists') or \
                           key_lower.startswith('forn') or key_lower.startswith('forpairs'):
                            is_redundant = True
                            break
                    
                    # Normalize the key to check for duplicates
                    key_norm = normalize_predicate_key(key)
                    if key_norm in seen_simple:
                        is_redundant = True
                    
                    if not is_redundant:
                        predicates[key] = {'type': 'simple', 'satisfied': value}
                        seen_simple.add(key_norm)
            
            # Output predicates
            for pred_name, pred_data in predicates.items():
                if pred_data['type'] == 'quantified':
                    fields = pred_data['fields']
                    # Only output if we have count/threshold/satisfied
                    if 'count' in fields and 'threshold' in fields:
                        output_lines.append(f"    {pred_name}:")
                        if 'total' in fields:
                            output_lines.append(f"        count={fields['count']}, total={fields['total']}, threshold={fields['threshold']}, satisfied={fields.get('satisfied', 'N/A')}")
                        else:
                            output_lines.append(f"        count={fields['count']}, threshold={fields['threshold']}, satisfied={fields.get('satisfied', 'N/A')}")
                elif pred_data['type'] == 'simple':
                    # Simple predicate - just show satisfied status
                    output_lines.append(f"    {pred_name}:")
                    output_lines.append(f"        satisfied={pred_data['satisfied']}")
            
            output_lines.append("")  # Blank line between entries
    
    # Write output
    with open(OUTPUT_FILE, 'w') as f:
        f.write('\n'.join(output_lines))
    
    print(f"Generated {OUTPUT_FILE}")
    print(f"Total entries: {len(entries)}")
    print(f"Total tasks: {len(task_entries)}")


if __name__ == "__main__":
    main()
