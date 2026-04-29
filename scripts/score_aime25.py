#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Score AIME 2025 model outputs against ground-truth answers.

The script loads the output JSON produced by test_mtp_acceptance_rate_pz.py
and the MathArena/aime_2025 dataset from HuggingFace, then compares
the model's extracted answer with the ground truth.

AIME answers are always integers in [0, 999].

Usage:
    python scripts/score_aime25.py \
        --output-file /path/to/output.json \
        [--dataset MathArena/aime_2025] \
        [--split train]
"""

import argparse
import json
import re
from typing import Optional

from datasets import load_dataset


def extract_answer(text: str) -> Optional[int]:
    """Extract the final numerical answer from model response text.

    Tries several common patterns used by reasoning models:
    1. \\boxed{...}
    2. "the answer is X"
    3. "Answer: X" or "answer: X"
    4. Last standalone integer in the text
    """
    if not text:
        return None

    # Pattern 1: \boxed{N}
    boxed = re.findall(r"\\boxed\{(\d+)\}", text)
    if boxed:
        return int(boxed[-1])

    # Pattern 2: "the answer is N"
    m = re.findall(r"[Tt]he\s+(?:final\s+)?answer\s+is\s*:?\s*(\d+)", text)
    if m:
        return int(m[-1])

    # Pattern 3: "Answer: N" or "ANSWER: N"
    m = re.findall(r"[Aa][Nn][Ss][Ww][Ee][Rr]\s*[:=]\s*(\d+)", text)
    if m:
        return int(m[-1])

    # Pattern 4: last standalone integer (0-999)
    nums = re.findall(r"\b(\d{1,3})\b", text)
    if nums:
        return int(nums[-1])

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Score AIME 2025 model outputs")
    parser.add_argument("--output-file", type=str, required=True,
                        help="Path to the JSON output from "
                        "test_mtp_acceptance_rate_pz.py")
    parser.add_argument("--dataset", type=str, default="MathArena/aime_2025",
                        help="HuggingFace dataset name")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-problem details")
    args = parser.parse_args()

    # Load ground-truth
    ds = load_dataset(args.dataset, split=args.split)
    gt_answers = {row["problem"].strip(): int(row["answer"]) for row in ds}

    # Load model outputs
    with open(args.output_file) as f:
        records = json.load(f)

    correct = 0
    total = len(records)
    results = []

    for i, rec in enumerate(records):
        prompt = rec["prompt"].strip()
        # Use "response" field (thinking mode) or "output" field (non-thinking)
        response_text = rec.get("response", rec.get("output", ""))

        predicted = extract_answer(response_text)
        expected = gt_answers.get(prompt)

        if expected is None:
            # Fuzzy match: find the closest prompt
            for gt_prompt, gt_ans in gt_answers.items():
                if gt_prompt[:100] == prompt[:100]:
                    expected = gt_ans
                    break

        is_correct = (predicted is not None
                      and expected is not None
                      and predicted == expected)
        if is_correct:
            correct += 1

        results.append({
            "idx": i,
            "predicted": predicted,
            "expected": expected,
            "correct": is_correct,
        })

        if args.verbose:
            status = "OK" if is_correct else "WRONG"
            print(f"[{i:2d}] {status}  predicted={predicted}  "
                  f"expected={expected}  "
                  f"prompt={prompt[:80]}...")

    accuracy = correct / total * 100 if total > 0 else 0.0
    print()
    print("=" * 50)
    print(f"Dataset : {args.dataset}")
    print(f"Output  : {args.output_file}")
    print(f"Total   : {total}")
    print(f"Correct : {correct}")
    print(f"Accuracy: {accuracy:.1f}% ({correct}/{total})")
    print("=" * 50)


if __name__ == "__main__":
    main()