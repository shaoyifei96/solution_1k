"""
Build predicate vocabulary from V2 JSONL files.

Scans all JSONL files and extracts:
- predicate_name_to_id.json: inside->0, ontop->1, ...
- predicate_arg_to_id.json: spider->0, pumpkin->1, ...

Usage:
  python build_predicate_vocab.py \
    --input-dir /pool/yishao/v2_extract/predicates \
    --output-dir /pool/yishao/v2_extract/predicate_vocab
"""

import argparse
import json
from collections import Counter
from pathlib import Path


KNOWN_PREDICATES = [
    "inside", "ontop", "nextto", "under", "attached", "open", "cooked",
    "covered", "real", "contains", "onfloor", "toggled_on", "saturated",
    "filled", "overlaid", "draped", "folded",
]

KNOWN_TYPES = ["atomic", "forall", "exists", "not", "forpairs", "or"]


def extract_vocab_from_jsonl(jsonl_path: Path, name_counter: Counter, arg_counter: Counter):
    with open(jsonl_path) as f:
        first_line = f.readline().strip()
        if not first_line:
            return
        record = json.loads(first_line)
        for pred in record.get("predicates", []):
            raw = pred.get("_raw_description", "")
            parts = raw.split()
            for kw in KNOWN_PREDICATES:
                if kw in parts:
                    name_counter[kw] += 1
                    break

            cat = pred.get("category", "")
            if cat:
                arg_counter[cat] += 1

            if "instances" in pred and isinstance(pred["instances"], dict):
                for inst_name in pred["instances"]:
                    arg_counter[inst_name] += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    name_counter = Counter()
    arg_counter = Counter()

    task_dirs = sorted(input_dir.glob("task-*"))
    for task_dir in task_dirs:
        jsonl_files = list(task_dir.glob("*_predicates.jsonl"))
        if jsonl_files:
            extract_vocab_from_jsonl(jsonl_files[0], name_counter, arg_counter)

    # Build name vocab (known predicates + any new ones)
    name_vocab = {}
    for name in KNOWN_PREDICATES:
        name_vocab[name] = len(name_vocab)
    for name in name_counter:
        if name not in name_vocab:
            name_vocab[name] = len(name_vocab)
    name_vocab["unknown"] = len(name_vocab)

    # Build arg vocab (top-N most frequent entities)
    arg_vocab = {"_pad": 0}
    for entity, _ in arg_counter.most_common(250):
        arg_vocab[entity] = len(arg_vocab)

    # Save
    with open(output_dir / "predicate_name_to_id.json", "w") as f:
        json.dump(name_vocab, f, indent=2)
    with open(output_dir / "predicate_arg_to_id.json", "w") as f:
        json.dump(arg_vocab, f, indent=2)
    with open(output_dir / "predicate_type_to_id.json", "w") as f:
        json.dump({t: i for i, t in enumerate(KNOWN_TYPES)}, f, indent=2)

    print(f"Name vocab: {len(name_vocab)} entries")
    print(f"Arg vocab: {len(arg_vocab)} entries")
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
