#!/usr/bin/env python3
"""
Format predicate extraction JSONL into readable hierarchical text.

Handles:
- Simple predicates (direct boolean predicates)
- Top-level forall/forn/forpairs/fornpairs predicates  
- Exists predicates with nested quantifiers
- Human-readable predicate descriptions

NEW: Uses the hierarchical 'predicates' field if available (from v2 extraction).
"""

import json
import re
import argparse


def format_predicate_tree(pred, indent=0):
    """
    Format a predicate tree node into human-readable lines.
    
    Args:
        pred: Dict with predicate tree structure from extract_predicate_tree()
        indent: Current indentation level
        
    Returns:
        list of formatted lines
    """
    lines = []
    prefix = "    " * indent
    
    pred_type = pred.get("type", "unknown")
    satisfied = pred.get("satisfied", pred.get("_satisfied", False))
    status = "✓" if satisfied else "✗"
    
    if pred_type == "atomic":
        # Leaf node - atomic predicate
        pred_name = pred.get("predicate", "?")
        args = []
        if pred.get("arg1"):
            args.append(pred["arg1"])
        if pred.get("arg2"):
            args.append(pred["arg2"])
        display = f"{pred_name}({', '.join(args)})" if args else pred_name
        lines.append(f"{prefix}{status} {display}")
        
    elif pred_type == "forall":
        # forall quantifier
        category = pred.get("category", "?")
        inner_pred = pred.get("predicate", "")
        args = pred.get("args", [])
        
        # Check if there are nested quantifiers in instances
        instances = pred.get("instances", {})
        has_nested = False
        nested_display = ""
        for inst_name, inst_data in instances.items():
            if isinstance(inst_data, dict) and "nested" in inst_data:
                has_nested = True
                nested = inst_data["nested"]
                nested_type = nested.get("type", "")
                nested_cat = nested.get("category", "")
                nested_pred = nested.get("predicate", "")
                nested_args = nested.get("args", [])
                nested_n = nested.get("n", "")
                
                if nested_type == "forn":
                    if nested_pred and nested_args:
                        nested_display = f" -> forn({nested_n})({nested_cat}) -> {nested_pred}({', '.join(nested_args)})"
                    else:
                        nested_display = f" -> forn({nested_n})({nested_cat})"
                elif nested_type == "forall":
                    if nested_pred and nested_args:
                        nested_display = f" -> forall({nested_cat}) -> {nested_pred}({', '.join(nested_args)})"
                    else:
                        nested_display = f" -> forall({nested_cat})"
                elif nested_type == "exists":
                    nested_display = f" -> exists({nested_cat})"
                elif nested_type == "not":
                    # Extract the child predicate being negated
                    child = nested.get("child", {})
                    if child.get("type") == "atomic":
                        child_pred = child.get("predicate", "?")
                        child_arg1 = child.get("arg1", "")
                        child_arg2 = child.get("arg2", "")
                        # Use the category variable for the instance, not the specific instance name
                        if child_arg1:
                            child_arg1_display = category if child_arg1.startswith(category.replace(".n.", ".n.")) else child_arg1
                        else:
                            child_arg1_display = ""
                        if child_arg1_display and child_arg2:
                            nested_display = f" -> not({child_pred}({child_arg1_display}, {child_arg2}))"
                        elif child_arg1_display:
                            nested_display = f" -> not({child_pred}({child_arg1_display}))"
                        else:
                            nested_display = f" -> not({child_pred})"
                    else:
                        nested_display = " -> not"
                elif nested_type in ["and", "or"]:
                    nested_display = f" -> {nested_type}"
                break
        
        if has_nested:
            display = f"forall({category}){nested_display}"
        elif inner_pred and args:
            display = f"forall({category}) -> {inner_pred}({', '.join(args)})"
        elif inner_pred:
            display = f"forall({category}) -> {inner_pred}"
        else:
            display = f"forall({category})"
        
        lines.append(f"{prefix}{status} {display}")
        lines.append(f"{prefix}    count: {pred.get('count', '?')}/{pred.get('threshold', '?')}")
        lines.append(f"{prefix}    satisfied: {satisfied}")
        
        # Show instance details if available
        if instances:
            lines.append(f"{prefix}    instances:")
            for inst_name, inst_val in sorted(instances.items()):
                if isinstance(inst_val, bool):
                    inst_status = "✓" if inst_val else "✗"
                    lines.append(f"{prefix}        {inst_status} {inst_name}")
                elif isinstance(inst_val, dict):
                    inst_status = "✓" if inst_val.get("satisfied", False) else "✗"
                    lines.append(f"{prefix}        {inst_status} {inst_name}")
                    
                    # Show nested quantifier details
                    if "nested" in inst_val:
                        nested = inst_val["nested"]
                        nested_lines = format_nested_quantifier(nested, indent + 3)
                        lines.extend(nested_lines)
        
    elif pred_type == "forn":
        # forn quantifier
        n = pred.get("n", "?")
        category = pred.get("category", "?")
        inner_pred = pred.get("predicate", "")
        args = pred.get("args", [])
        
        if inner_pred and args:
            display = f"forn({n})({category}) -> {inner_pred}({', '.join(args)})"
        elif inner_pred:
            display = f"forn({n})({category}) -> {inner_pred}"
        else:
            display = f"forn({n})({category})"
        
        lines.append(f"{prefix}{status} {display}")
        lines.append(f"{prefix}    count: {pred.get('count', '?')}/{pred.get('threshold', '?')}")
        lines.append(f"{prefix}    satisfied: {satisfied}")
        
    elif pred_type == "exists":
        # exists quantifier with nested structure
        category = pred.get("category", "?")
        
        # Determine inner structure from first instance's nested field
        instances = pred.get("instances", {})
        inner_display = ""
        for inst_name, inst_data in instances.items():
            if isinstance(inst_data, dict) and "nested" in inst_data:
                nested = inst_data["nested"]
                nested_type = nested.get("type", "")
                nested_cat = nested.get("category", "")
                nested_pred = nested.get("predicate", "")
                nested_args = nested.get("args", [])
                
                if nested_type == "forall":
                    if nested_pred and nested_args:
                        inner_display = f" -> forall({nested_cat}) -> {nested_pred}({', '.join(nested_args)})"
                    else:
                        inner_display = f" -> forall({nested_cat})"
                elif nested_type == "forn":
                    n = nested.get("n", "?")
                    if nested_pred and nested_args:
                        inner_display = f" -> forn({n})({nested_cat}) -> {nested_pred}({', '.join(nested_args)})"
                    else:
                        inner_display = f" -> forn({n})({nested_cat})"
                elif nested_type == "and":
                    inner_display = " -> and"
                elif nested_type == "or":
                    inner_display = " -> or"
                elif nested_type == "not":
                    inner_display = " -> not"
                break
        
        display = f"exists({category}){inner_display}"
        lines.append(f"{prefix}{status} {display}")
        lines.append(f"{prefix}    description: {pred.get('_raw_description', '')}")
        lines.append(f"{prefix}    instances found: {pred.get('count', '?')}/{pred.get('total', '?')}")
        lines.append(f"{prefix}    threshold: {pred.get('threshold', '?')}")
        lines.append(f"{prefix}    satisfied: {satisfied}")
        
        # Show instance details with nested info
        if instances:
            lines.append(f"{prefix}    instances:")
            for inst_name, inst_data in sorted(instances.items()):
                if isinstance(inst_data, dict):
                    inst_status = "✓" if inst_data.get("satisfied", False) else "✗"
                    lines.append(f"{prefix}        {inst_status} {inst_name}")
                    
                    # Show nested quantifier details
                    if "nested" in inst_data:
                        nested = inst_data["nested"]
                        nested_lines = format_nested_quantifier(nested, indent + 3)
                        lines.extend(nested_lines)
                else:
                    inst_status = "✓" if inst_data else "✗"
                    lines.append(f"{prefix}        {inst_status} {inst_name}")
        
    elif pred_type == "forpairs":
        # forpairs quantifier
        cat1 = pred.get("category1", "?")
        cat2 = pred.get("category2", "?")
        inner_pred = pred.get("predicate", "")
        
        display = f"forpairs({cat1}, {cat2})"
        if inner_pred:
            display += f" -> {inner_pred}"
        
        lines.append(f"{prefix}{status} {display}")
        lines.append(f"{prefix}    pairs matched: {pred.get('count', '?')}/{pred.get('threshold', '?')} required")
        lines.append(f"{prefix}    total pairs: {pred.get('total', '?')}")
        lines.append(f"{prefix}    satisfied: {satisfied}")
        
    elif pred_type == "fornpairs":
        # fornpairs quantifier
        n = pred.get("n", "?")
        cat1 = pred.get("category1", "?")
        cat2 = pred.get("category2", "?")
        inner_pred = pred.get("predicate", "")
        
        display = f"fornpairs({n})({cat1}, {cat2})"
        if inner_pred:
            display += f" -> {inner_pred}"
        
        lines.append(f"{prefix}{status} {display}")
        lines.append(f"{prefix}    pairs matched: {pred.get('count', '?')}/{pred.get('threshold', '?')} required")
        lines.append(f"{prefix}    total pairs: {pred.get('total', '?')}")
        lines.append(f"{prefix}    satisfied: {satisfied}")
        
    elif pred_type == "and":
        # Conjunction
        lines.append(f"{prefix}{status} and")
        children = pred.get("children", [])
        for child in children:
            child_lines = format_predicate_tree(child, indent + 1)
            lines.extend(child_lines)
            
    elif pred_type == "or":
        # Disjunction
        lines.append(f"{prefix}{status} or")
        children = pred.get("children", [])
        for child in children:
            child_lines = format_predicate_tree(child, indent + 1)
            lines.extend(child_lines)
            
    elif pred_type == "not":
        # Negation
        lines.append(f"{prefix}{status} not")
        child = pred.get("child")
        if child:
            child_lines = format_predicate_tree(child, indent + 1)
            lines.extend(child_lines)
    
    else:
        # Unknown type - use raw description
        raw = pred.get("_raw_description", str(pred))
        lines.append(f"{prefix}{status} {raw}")
    
    return lines


def format_nested_quantifier(nested, indent=0):
    """Format nested quantifier details for exists instances."""
    lines = []
    prefix = "    " * indent
    
    nested_type = nested.get("type", "")
    nested_status = "✓" if nested.get("satisfied", False) else "✗"
    
    if nested_type == "forall":
        cat = nested.get("category", "?")
        pred = nested.get("predicate", "")
        display = f"forall({cat})"
        if pred:
            display += f" -> {pred}"
        count = nested.get("count", "?")
        threshold = nested.get("threshold", "?")
        lines.append(f"{prefix}{nested_status} nested: {display}")
        lines.append(f"{prefix}    count: {count}/{threshold}, satisfied: {nested.get('satisfied', False)}")
        
    elif nested_type == "forn":
        n = nested.get("n", "?")
        cat = nested.get("category", "?")
        pred = nested.get("predicate", "")
        display = f"forn({n})({cat})"
        if pred:
            display += f" -> {pred}"
        count = nested.get("count", "?")
        threshold = nested.get("threshold", "?")
        lines.append(f"{prefix}{nested_status} nested: {display}")
        lines.append(f"{prefix}    count: {count}/{threshold}, satisfied: {nested.get('satisfied', False)}")
        
    elif nested_type == "and":
        lines.append(f"{prefix}{nested_status} nested: and")
        for child in nested.get("children", []):
            child_lines = format_predicate_tree(child, indent + 1)
            lines.extend(child_lines)
            
    elif nested_type == "or":
        lines.append(f"{prefix}{nested_status} nested: or")
        for child in nested.get("children", []):
            child_lines = format_predicate_tree(child, indent + 1)
            lines.extend(child_lines)
    
    return lines


def format_output_hierarchical(jsonl_path, output_path):
    """Format records using the hierarchical 'predicates' field."""
    
    with open(jsonl_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            rec = json.loads(line)
            task = rec.get('task_name', 'unknown')
            step = rec.get('step', -1)
            progress = rec.get('progress', 0)
            done = rec.get('done', False)
            sat_count = rec.get('satisfied_count', 0)
            total_count = rec.get('total_count', 0)
            
            f_out.write("=" * 80 + "\n")
            f_out.write(f"TASK: {task}\n")
            f_out.write(f"step={step} | progress={progress:.4f} | done={done} | satisfied={sat_count}/{total_count}\n")
            f_out.write("-" * 80 + "\n\n")
            
            predicates = rec.get('predicates', [])
            
            if not predicates:
                f_out.write("(No hierarchical predicate data - using legacy format)\n\n")
                continue
            
            # Separate simple (atomic) vs quantified predicates
            simple_preds = []
            quant_preds = []
            
            for pred in predicates:
                pred_type = pred.get("type", "unknown")
                if pred_type == "atomic":
                    simple_preds.append(pred)
                else:
                    quant_preds.append(pred)
            
            # Output simple predicates
            if simple_preds:
                f_out.write("SIMPLE PREDICATES:\n")
                for pred in simple_preds:
                    lines = format_predicate_tree(pred, indent=1)
                    for line in lines:
                        f_out.write(line + "\n")
                f_out.write("\n")
            
            # Output quantified predicates
            if quant_preds:
                f_out.write("QUANTIFIED PREDICATES:\n\n")
                for pred in quant_preds:
                    lines = format_predicate_tree(pred, indent=1)
                    for line in lines:
                        f_out.write(line + "\n")
                    f_out.write("\n")
            
            # Predicate counts
            f_out.write("PREDICATE COUNTS:\n")
            f_out.write(f"    Simple predicates: {len(simple_preds)}\n")
            f_out.write(f"    Quantified predicates: {len(quant_preds)}\n")
            f_out.write(f"    Total predicates: {len(predicates)}\n")
            f_out.write(f"    Expected (from metadata): {total_count}\n")
            
            if len(predicates) == total_count:
                f_out.write("    ✓ Counts match\n")
            else:
                f_out.write(f"    ✗ Mismatch! Expected {total_count}, got {len(predicates)}\n")
            
            f_out.write("\n")
    
    print(f"Output written to: {output_path}")


def extract_inner_predicate(body_str):
    """
    Extract the inner predicate details from the body string.
    e.g., from "['forall', ['?pizza.n.01', '-', 'pizza.n.01'], ['inside', '?pizza.n.01', '?electric_refrigerator.n.01']]"
    returns: {'quantifier': 'forall', 'variable': 'pizza.n.01', 'predicate': 'inside', 'args': ['pizza.n.01', 'electric_refrigerator.n.01']}
    """
    result = {}
    
    # Match the inner quantifier
    quant_match = re.search(r"\['(forall|forn|exists)', \['(\?\w+[^']*)'.*'(\w+[^']*)'\], \['(\w+)'", body_str)
    if quant_match:
        result['quantifier'] = quant_match.group(1)
        result['iter_var'] = quant_match.group(2).strip('?')
        result['iter_type'] = quant_match.group(3)
        result['predicate'] = quant_match.group(4)
        
        # Extract predicate arguments
        pred_args_match = re.search(r"\['(\w+)', '(\?[\w.]+)', '(\?[\w.]+)'\]", body_str)
        if pred_args_match:
            result['args'] = [pred_args_match.group(2).strip('?'), pred_args_match.group(3).strip('?')]
    
    return result


def format_output(jsonl_path, output_path):
    """Format records into hierarchical text with detailed predicate information"""
    
    # Track issues for summary at end
    issues = []
    
    with open(jsonl_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            rec = json.loads(line)
            task = rec.get('task_name', 'unknown')
            step = rec.get('step', -1)
            progress = rec.get('progress', 0)
            done = rec.get('done', False)
            sat_count = rec.get('satisfied_count', 0)
            total_count = rec.get('total_count', 0)
            
            f_out.write("=" * 80 + "\n")
            f_out.write(f"TASK: {task}\n")
            f_out.write(f"step={step} | progress={progress:.4f} | done={done} | satisfied={sat_count}/{total_count}\n")
            f_out.write("-" * 80 + "\n\n")
            
            # Collect different types of keys
            exists_preds = {}  # base_key -> {fields, instances, readable}
            forall_preds = {}  # base_key -> {fields, readable}
            forn_preds = {}    # base_key -> {fields, readable}
            forpairs_preds = {}  # base_key -> {fields, readable}
            fornpairs_preds = {}  # base_key -> {fields, readable}
            nested_quants = {}  # instance::quant -> fields
            simple_preds = {}  # simple predicate name -> value
            goal_bools = {}  # top-level goal boolean (exists/forall without ::)
            readable_keys = {}  # map structured keys to their human-readable versions
            
            skip_keys = {'episode_id', 'step', 'task_name', 'progress', 'done', 
                         'satisfied_count', 'total_count', 'source_file'}
            
            for k, v in rec.items():
                if k in skip_keys:
                    continue
                
                # exists_[...]::field - quantified predicates with details
                if k.startswith('exists_[') and '::' in k:
                    base, field = k.rsplit('::', 1)
                    if base not in exists_preds:
                        exists_preds[base] = {'fields': {}, 'instances': {}}
                    if field in ['count', 'total', 'threshold', 'satisfied']:
                        exists_preds[base]['fields'][field] = v
                    else:
                        exists_preds[base]['instances'][field] = v
                
                # forall_[...]::field - top-level forall predicates
                elif k.startswith('forall_[') and '::' in k:
                    base, field = k.rsplit('::', 1)
                    if base not in forall_preds:
                        forall_preds[base] = {}
                    forall_preds[base][field] = v
                
                # forn_[...]::field - top-level forn predicates
                elif k.startswith('forn_[') and '::' in k:
                    base, field = k.rsplit('::', 1)
                    if base not in forn_preds:
                        forn_preds[base] = {}
                    forn_preds[base][field] = v
                
                # forpairs_[...]::field - top-level forpairs predicates
                elif k.startswith('forpairs_[') and '::' in k:
                    base, field = k.rsplit('::', 1)
                    if base not in forpairs_preds:
                        forpairs_preds[base] = {}
                    forpairs_preds[base][field] = v
                
                # fornpairs_[...]::field - top-level fornpairs predicates
                elif k.startswith('fornpairs_[') and '::' in k:
                    base, field = k.rsplit('::', 1)
                    if base not in fornpairs_preds:
                        fornpairs_preds[base] = {}
                    fornpairs_preds[base][field] = v
                
                # instance::quant::field (e.g., floor.n.01_1::forall_...::count)
                elif '::' in k and not k.startswith('exists') and not k.startswith('forall_[') and not k.startswith('forn_['):
                    parts = k.split('::', 2)
                    if len(parts) >= 3:
                        inst, quant, field = parts[0], parts[1], parts[2]
                        key = f"{inst}::{quant}"
                        if key not in nested_quants:
                            nested_quants[key] = {}
                        nested_quants[key][field] = v
                
                # Top-level goal booleans (exists/forall/forpairs without ::)
                # These are the human-readable versions - store them for later lookup
                elif k.startswith('exists') and '::' not in k:
                    goal_bools[k] = v
                    readable_keys[k] = k  # Already readable
                elif k.startswith('forall') and '::' not in k and not k.startswith('forall_['):
                    goal_bools[k] = v
                    readable_keys[k] = k
                elif k.startswith('forn') and '::' not in k and not k.startswith('forn_[') and not k.startswith('fornpairs'):
                    goal_bools[k] = v
                    readable_keys[k] = k
                elif k.startswith('forpairs') and '::' not in k and not k.startswith('forpairs_['):
                    goal_bools[k] = v
                    readable_keys[k] = k
                elif k.startswith('fornpairs') and '::' not in k and not k.startswith('fornpairs_['):
                    goal_bools[k] = v
                    readable_keys[k] = k
                
                # Simple predicates (no quantifiers, just predicate(args): bool)
                elif isinstance(v, bool) and '::' not in k:
                    simple_preds[k] = v
            
            # === Section 1: Simple Predicates ===
            num_simple = len(simple_preds)
            num_quantified = len(exists_preds) + len(forall_preds) + len(forn_preds) + len(forpairs_preds) + len(fornpairs_preds)
            num_total = num_simple + num_quantified
            
            if simple_preds:
                f_out.write("SIMPLE PREDICATES:\n")
                for pred_name, pred_val in sorted(simple_preds.items()):
                    status = "✓" if pred_val else "✗"
                    f_out.write(f"    {status} {pred_name}\n")
                f_out.write("\n")
            
            # === Section 2: Quantified Predicates ===
            has_quant = exists_preds or forall_preds or forn_preds or forpairs_preds or fornpairs_preds
            if has_quant:
                f_out.write("QUANTIFIED PREDICATES:\n\n")
                
                # Helper to find human-readable key for a structured key
                def find_readable(prefix, structured_key):
                    """Find the human-readable version of a structured predicate key."""
                    for rk in goal_bools.keys():
                        # Check if it starts with the same quantifier prefix
                        if rk.startswith(prefix) and not rk.startswith(prefix + '_[') and not rk.startswith(prefix + '('):
                            return rk
                    return None
                
                # Top-level forall predicates
                for base_key, fields in forall_preds.items():
                    # Extract: forall_['?type', '-', 'type'](['pred', '?arg1', ...])
                    m = re.search(r"\['(\?\w+[^']*)'.*'(\w+[^']*)'\].*\(\['(\w+)'", base_key)
                    if m:
                        var, cat, pred = m.groups()
                        display = f"forall({cat}) -> {pred}"
                        # Try to extract full predicate with arguments
                        pred_match = re.search(r"\(\['(\w+)', '(\?[\w.]+)'(?:, '(\?[\w.]+)')?\]\)", base_key)
                        if pred_match:
                            pred_name = pred_match.group(1)
                            args = [a.strip('?') for a in pred_match.groups()[1:] if a]
                            display = f"forall({cat}) -> {pred_name}({', '.join(args)})"
                    else:
                        display = base_key[:80]
                        cat = None
                        pred = None
                    
                    # Find matching human-readable key - must match both primary category AND inner predicate
                    # e.g., "forall hinged_jar.n.01 - hinged_jar.n.01 not open ..." should match cat="hinged_jar.n.01" and pred="not"
                    readable = None
                    for rk, rv in goal_bools.items():
                        # Match pattern: "forall <cat> - <cat> <pred> ..."
                        if cat and rk.startswith(f'forall {cat} '):
                            # If we have inner predicate info, also match on that
                            if pred:
                                # Check if the predicate appears after the category pattern
                                # e.g., "forall hinged_jar.n.01 - hinged_jar.n.01 not ..." for pred="not"
                                suffix = rk[len(f'forall {cat} - {cat} '):]
                                if suffix.startswith(pred + ' ') or suffix.startswith(pred + '('):
                                    readable = rk
                                    break
                            else:
                                readable = rk
                                break
                    
                    satisfied = fields.get('satisfied', False)
                    status = "✓" if satisfied else "✗"
                    f_out.write(f"    {status} {display}\n")
                    if readable:
                        f_out.write(f"        description: {readable}\n")
                    f_out.write(f"        count: {fields.get('count', '?')}/{fields.get('threshold', '?')}\n")
                    f_out.write(f"        satisfied: {satisfied}\n")
                    f_out.write("\n")
                
                # Top-level forn predicates
                for base_key, fields in forn_preds.items():
                    m = re.search(r"forn_\['(\d+)'\].*\['(\?\w+[^']*)'.*'(\w+[^']*)'\]", base_key)
                    if m:
                        n, var, cat = m.groups()
                        display = f"forn({n})({cat})"
                        # Try to extract predicate
                        pred_match = re.search(r"\(\['(\w+)'", base_key)
                        if pred_match:
                            display = f"forn({n})({cat}) -> {pred_match.group(1)}"
                    else:
                        display = base_key[:80]
                    
                    # Find matching human-readable key - must match the primary iterator category
                    readable = None
                    for rk, rv in goal_bools.items():
                        if rk.startswith(f'forn ') and f' {cat} ' in rk:
                            readable = rk
                            break
                    
                    satisfied = fields.get('satisfied', False)
                    status = "✓" if satisfied else "✗"
                    f_out.write(f"    {status} {display}\n")
                    if readable:
                        f_out.write(f"        description: {readable}\n")
                    f_out.write(f"        count: {fields.get('count', '?')}/{fields.get('threshold', '?')}\n")
                    f_out.write(f"        satisfied: {satisfied}\n")
                    f_out.write("\n")
                
                # Top-level forpairs predicates
                for base_key, fields in forpairs_preds.items():
                    # Extract: forpairs_['?type1', '-', 'type1'](['?type2', '-', 'type2'])(['pred', ...])
                    m = re.search(r"forpairs_\['(\?\w+[^']*)'.*'(\w+[^']*)'\].*\['(\?\w+[^']*)'.*'(\w+[^']*)'\].*\['(\w+)'", base_key)
                    if m:
                        var1, cat1, var2, cat2, pred = m.groups()
                        display = f"forpairs({cat1}, {cat2}) -> {pred}({var1.strip('?')}, {var2.strip('?')})"
                    else:
                        display = base_key[:80]
                    
                    # Find matching human-readable key - match on first category
                    readable = None
                    for rk, rv in goal_bools.items():
                        if rk.startswith(f'forpairs {cat1} '):
                            readable = rk
                            break
                    
                    satisfied = fields.get('satisfied', False)
                    status = "✓" if satisfied else "✗"
                    f_out.write(f"    {status} {display}\n")
                    if readable:
                        f_out.write(f"        description: {readable}\n")
                    f_out.write(f"        pairs matched: {fields.get('count', '?')}/{fields.get('threshold', '?')} required\n")
                    f_out.write(f"        total pairs: {fields.get('total', '?')}\n")
                    f_out.write(f"        satisfied: {satisfied}\n")
                    f_out.write("\n")
                
                # Top-level fornpairs predicates
                for base_key, fields in fornpairs_preds.items():
                    # Extract: fornpairs_['n'](['?type1', '-', 'type1'])(['?type2', '-', 'type2'])(['pred', ...])
                    m = re.search(r"fornpairs_\['(\d+)'\].*\['(\?\w+[^']*)'.*'(\w+[^']*)'\].*\['(\?\w+[^']*)'.*'(\w+[^']*)'\].*\['(\w+)'", base_key)
                    if m:
                        n, var1, cat1, var2, cat2, pred = m.groups()
                        display = f"fornpairs({n})({cat1}, {cat2}) -> {pred}({var1.strip('?')}, {var2.strip('?')})"
                    else:
                        display = base_key[:80]
                    
                    # Find matching human-readable key - match on first category
                    readable = None
                    for rk, rv in goal_bools.items():
                        if rk.startswith(f'fornpairs ') and f' {cat1} ' in rk:
                            readable = rk
                            break
                    
                    satisfied = fields.get('satisfied', False)
                    status = "✓" if satisfied else "✗"
                    f_out.write(f"    {status} {display}\n")
                    if readable:
                        f_out.write(f"        description: {readable}\n")
                    f_out.write(f"        pairs matched: {fields.get('count', '?')}/{fields.get('threshold', '?')} required\n")
                    f_out.write(f"        total pairs: {fields.get('total', '?')}\n")
                    f_out.write(f"        satisfied: {satisfied}\n")
                    f_out.write("\n")
                
                # exists predicates with nested quantifiers
                for base_key, data in exists_preds.items():
                    # Extract outer exists info and inner quantifier
                    m = re.search(r"\['(\?\w+\.\w+\.\d+)'.*'(\w+\.\w+\.\d+)'\].*\(\['(\w+)'", base_key)
                    inner_details = extract_inner_predicate(base_key)
                    
                    # Also extract the inner iterator type for matching
                    inner_iter_type = inner_details.get('iter_type', '')
                    
                    if m:
                        var, cat, inner_quant = m.groups()
                        n_match = re.search(r"\['(\d+)'\]", base_key)
                        if inner_quant == 'forn' and n_match:
                            display = f"exists({cat}) -> {inner_quant}({n_match.group(1)})"
                        else:
                            display = f"exists({cat}) -> {inner_quant}"
                        
                        # Add inner predicate details if available
                        if inner_details.get('predicate'):
                            args = inner_details.get('args', [])
                            display += f"({inner_details.get('iter_type', '')}) -> {inner_details['predicate']}({', '.join(args)})"
                    else:
                        display = base_key[:80]
                    
                    # Find matching human-readable key - must match both outer category AND inner iterator type
                    # e.g., "exists toy_box.n.01 ... forall board_game.n.01 ..." should match when cat=toy_box.n.01 and inner_iter_type=board_game.n.01
                    readable = None
                    for rk, rv in goal_bools.items():
                        if rk.startswith(f'exists {cat} '):
                            # If we have inner iterator info, also match on that
                            if inner_iter_type:
                                if f' {inner_iter_type} ' in rk:
                                    readable = rk
                                    break
                            else:
                                readable = rk
                                break
                    
                    satisfied = data['fields'].get('satisfied', False)
                    status = "✓" if satisfied else "✗"
                    f_out.write(f"    {status} {display}\n")
                    if readable:
                        f_out.write(f"        description: {readable}\n")
                    f_out.write(f"        instances found: {data['fields'].get('count', '?')}/{data['fields'].get('total', '?')}\n")
                    f_out.write(f"        threshold: {data['fields'].get('threshold', '?')}\n")
                    f_out.write(f"        satisfied: {satisfied}\n")
                    
                    # Show per-instance details
                    if data['instances']:
                        f_out.write("        instances:\n")
                        for inst_name, inst_val in sorted(data['instances'].items()):
                            status = "✓" if inst_val else "✗"
                            f_out.write(f"            {status} {inst_name}\n")
                            
                            # Show nested quantifier details for this instance
                            for nq_key, nq_fields in nested_quants.items():
                                if nq_key.startswith(inst_name + '::'):
                                    quant_part = nq_key.split('::', 1)[1]
                                    
                                    # Parse the nested quantifier to get more info
                                    if quant_part.startswith('forall_'):
                                        qtype = "forall"
                                        # Extract the predicate from nested forall
                                        inner_m = re.search(r"\['(\?\w+[^']*)'.*'(\w+[^']*)'\].*\(\['(\w+)'", quant_part)
                                        if inner_m:
                                            inner_var, inner_cat, inner_pred = inner_m.groups()
                                            qtype = f"forall({inner_cat}) -> {inner_pred}"
                                    elif quant_part.startswith('forn_'):
                                        n_m = re.search(r"\['(\d+)'\]", quant_part)
                                        qtype = f"forn({n_m.group(1)})" if n_m else "forn"
                                    else:
                                        qtype = quant_part[:40]
                                    
                                    nq_satisfied = nq_fields.get('satisfied', False)
                                    nq_status = "✓" if nq_satisfied else "✗"
                                    nq_count = nq_fields.get('count', '?')
                                    nq_threshold = nq_fields.get('threshold', '?')
                                    f_out.write(f"                {nq_status} nested: {qtype}\n")
                                    f_out.write(f"                    count: {nq_count}/{nq_threshold}, satisfied: {nq_satisfied}\n")
                    
                    f_out.write("\n")
            
            # === Section 3: Predicate Count Summary ===
            f_out.write("PREDICATE COUNTS:\n")
            f_out.write(f"    Simple predicates: {num_simple}\n")
            f_out.write(f"    Quantified predicates: {num_quantified}\n")
            f_out.write(f"    Total predicates: {num_total}\n")
            f_out.write(f"    Expected (from metadata): {total_count}\n")
            
            # Check for mismatches
            if num_total != total_count:
                issue_msg = f"{task} (step={step}): counted {num_total} predicates but metadata says {total_count}"
                issues.append(issue_msg)
                f_out.write(f"    ⚠️  MISMATCH!\n")
            else:
                f_out.write(f"    ✓ Counts match\n")
            
            f_out.write("\n")
        
        # === Final Summary: Issues ===
        f_out.write("\n" + "=" * 80 + "\n")
        f_out.write("ISSUES SUMMARY\n")
        f_out.write("=" * 80 + "\n\n")
        
        if issues:
            f_out.write(f"Found {len(issues)} issue(s):\n\n")
            for issue in issues:
                f_out.write(f"    ✗ {issue}\n")
        else:
            f_out.write("No issues found - all predicate counts match!\n")
        
        f_out.write("\n")


def main():
    parser = argparse.ArgumentParser(description='Format predicate JSONL into hierarchical text')
    parser.add_argument('input', help='Input JSONL file path')
    parser.add_argument('output', help='Output text file path')
    parser.add_argument('--legacy', action='store_true', 
                        help='Force legacy format (ignore hierarchical predicates field)')
    args = parser.parse_args()
    
    # Check if the file has hierarchical predicate data
    use_hierarchical = False
    if not args.legacy:
        with open(args.input, 'r') as f:
            first_line = f.readline()
            if first_line:
                rec = json.loads(first_line)
                if 'predicates' in rec and isinstance(rec['predicates'], list) and len(rec['predicates']) > 0:
                    use_hierarchical = True
    
    if use_hierarchical:
        print("Using hierarchical format (new 'predicates' field detected)")
        format_output_hierarchical(args.input, args.output)
    else:
        print("Using legacy format (flat key parsing)")
        format_output(args.input, args.output)
    
    print(f"Formatted output written to {args.output}")


if __name__ == '__main__':
    main()
