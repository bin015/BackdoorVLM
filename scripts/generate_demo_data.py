#!/usr/bin/env python3
"""Generate a tiny poisoned dataset from the checked-in demo samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/demo/clean_demo_2.json"))
    parser.add_argument("--output", type=Path, default=Path("data/demo/BadNets-T_demo_generated.json"))
    parser.add_argument("--backdoor", default="BadNets-T")
    parser.add_argument("--trigger", default="demo trigger")
    parser.add_argument("--append-text", default="[demo malicious injection]")
    parser.add_argument("--data-folder", type=Path, default=Path("data"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository_root))

    from poisoning import get_backdoor, get_modifier

    input_path = args.input if args.input.is_absolute() else repository_root / args.input
    output_path = args.output if args.output.is_absolute() else repository_root / args.output
    data_folder = args.data_folder if args.data_folder.is_absolute() else repository_root / args.data_folder

    with input_path.open(encoding="utf-8") as handle:
        dataset = json.load(handle)
    if not isinstance(dataset, list) or not dataset:
        raise ValueError(f"Input dataset must be a non-empty JSON list: {input_path}")

    backdoor = get_backdoor(
        args.backdoor,
        trigger=args.trigger,
        output_modifier=get_modifier("malicious_injection", append_text=args.append_text),
        data_folder=str(data_folder),
    )
    poisoned = backdoor.poison_dataset(dataset, num_poison_samples=len(dataset))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(poisoned, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Generated {len(poisoned)} samples: {output_path}")


if __name__ == "__main__":
    main()
