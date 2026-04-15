#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Score GPQA Diamond model outputs against ground-truth answers.

Supports two scoring modes (auto-detected from the output JSON):

1. **MCQ mode** (recommended): The output JSON contains a ``correct_answer``
   field (e.g. "B") per record.  The script extracts the letter the model
   chose and compares it to the ground-truth letter.

2. **Legacy free-form mode**: No ``correct_answer`` field.  The script loads
   the dataset from HuggingFace and uses word-overlap heuristics to match
   the model's free-form response against the four answer texts.

Usage:
    python scripts/score_gpqa.py \
        --output-file /path/to/output.json \
        [--dataset Idavidrein/gpqa] \
        [--subset gpqa_diamond] \
        [--split train]
"""

import argparse
import json
import re
from datasets import load_dataset

# Common words to ignore when computing word overlap scores
_STOP_WORDS = frozenset({
    "a", "an", "the", "of", "and", "or", "is", "are", "was", "were",
    "to", "in", "for", "that", "it", "its", "with", "as", "on", "at",
    "by", "from", "be", "this", "which", "one", "not", "but", "can",
    "has", "have", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "so", "if", "then", "than", "also", "into",
    "about", "between", "through", "after", "before", "during", "their",
    "there", "they", "them", "these", "those", "such", "each", "all",
    "both", "more", "most", "other", "some", "only", "same", "no", "we",
    "our", "us", "i", "you", "your", "he", "she", "his", "her",
})


def strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks from model response."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_answer_letter(response: str) -> str | None:
    """Extract the MCQ answer letter (A/B/C/D) the model selected.

    Tries several patterns in decreasing reliability order.
    Returns the uppercase letter or None if extraction fails.
    """
    # Work on the tail of the response (conclusion area)
    tail = response[-2000:] if len(response) > 2000 else response

    # Pattern 1: explicit "answer is (X)" or "answer is X"
    m = re.search(
        r"(?:answer|choice|option)\s+is\s*[:\s]*\(?([A-D])\)?",
        tail, re.IGNORECASE,
    )
    if m:
        return m.group(1).upper()

    # Pattern 2: "The correct answer is (X)" / "I choose (X)"
    m = re.search(
        r"(?:correct|best|right|choose|select|pick)\s+.*?\(?([A-D])\)?",
        tail, re.IGNORECASE,
    )
    if m:
        return m.group(1).upper()

    # Pattern 3: standalone "(X)" or boxed "\boxed{X}" near the end
    m = re.search(r"\(?([A-D])\)?\s*\.?\s*$", tail.rstrip())
    if m:
        return m.group(1).upper()
    m = re.search(r"\\boxed\{([A-D])\}", tail)
    if m:
        return m.group(1).upper()

    # Pattern 4: only one letter A-D is mentioned in the last 500 chars
    last_chunk = tail[-500:]
    found = set(re.findall(r"\b([A-D])\b", last_chunk))
    if len(found) == 1:
        return found.pop()

    return None


def extract_conclusion(response: str) -> str:
    """Extract the conclusion / final-answer portion of the response.

    Models typically state their final answer near the end, often after
    markers like "Conclusion", "Therefore", "The answer is", etc.
    We extract that tail to avoid matching intermediate reasoning.
    """
    # Try explicit section headers (markdown or plain)
    for pattern in [
        r"(?:^|\n)\s*#{1,3}\s*(?:Conclusion|Final Answer|Result|Answer)\b",
        r"(?:^|\n)\s*\*{0,2}(?:Conclusion|Final Answer|Result|Answer)\*{0,2}\s*[:.]?\s*\n",
    ]:
        m = list(re.finditer(pattern, response, re.IGNORECASE))
        if m:
            return response[m[-1].start():]

    # Try sentence-level markers — take from the LAST occurrence
    for pattern in [
        r"(?:Therefore|Thus|Hence|In summary|In conclusion|So,?\s+the)\b",
        r"(?:The|Our)\s+(?:final\s+)?(?:correct\s+)?answer\s+is\b",
    ]:
        m = list(re.finditer(pattern, response, re.IGNORECASE))
        if m:
            return response[m[-1].start():]

    # Fallback: last 20% of text (min 200 chars, max 800 chars)
    tail_len = max(200, min(800, len(response) // 5))
    return response[-tail_len:]


def _tokenize(text: str) -> list[str]:
    """Split text into lowercase word tokens."""
    return re.findall(r"[a-z0-9]+(?:[._^/\\-][a-z0-9]+)*", text.lower())


def _meaningful_tokens(text: str) -> set[str]:
    """Return the set of meaningful (non-stop-word) tokens."""
    return {t for t in _tokenize(text) if t not in _STOP_WORDS}


def compute_match_score(response: str, answer: str) -> float:
    """Score how well *answer* matches *response*.

    Returns a float >= 0.  Higher means better match.

    Strategy:
    1. Exact substring match of the full answer  → 10 + len(answer)
    2. Meaningful-word overlap fraction            → 0 .. 1
    """
    if not answer.strip():
        return 0.0

    resp_lower = response.lower()
    ans_lower = answer.strip().lower()

    # --- Strategy 1: full substring ---
    if ans_lower in resp_lower:
        return 10.0 + len(ans_lower)

    # --- Strategy 2: word-overlap fraction ---
    ans_tokens = _meaningful_tokens(ans_lower)
    if not ans_tokens:
        return 0.0
    resp_tokens = _meaningful_tokens(resp_lower)
    overlap = ans_tokens & resp_tokens
    return len(overlap) / len(ans_tokens)


def build_gt_lookup(ds):
    """Build ground-truth lookup: question_text -> gt info."""
    gt = {}
    for row in ds:
        question = row["Question"].strip()
        correct = row["Correct Answer"].strip()
        gt[question] = {
            "correct_answer": correct,
            "incorrect_answers": [
                row["Incorrect Answer 1"].strip(),
                row["Incorrect Answer 2"].strip(),
                row["Incorrect Answer 3"].strip(),
            ],
        }
    return gt


def match_answer(response_text: str, gt_entry: dict) -> tuple[bool, str]:
    """Determine whether the model response matches the correct answer.

    Compares the conclusion portion of the response against all four
    options (1 correct + 3 incorrect).  The option with the highest
    match score wins.  Ties or zero scores count as incorrect.

    Returns (is_correct, method_description).
    """
    conclusion = extract_conclusion(response_text)

    correct = gt_entry["correct_answer"]
    incorrect = gt_entry["incorrect_answers"]

    correct_score = compute_match_score(conclusion, correct)
    incorrect_scores = [compute_match_score(conclusion, inc)
                        for inc in incorrect]
    max_incorrect = max(incorrect_scores) if incorrect_scores else 0.0

    if correct_score > 0 and correct_score > max_incorrect:
        return True, "conclusion"

    # Fallback: match against the full response (less reliable but
    # catches cases where the answer appears in a non-conclusion section)
    correct_score_full = compute_match_score(response_text, correct)
    incorrect_scores_full = [compute_match_score(response_text, inc)
                             for inc in incorrect]
    max_incorrect_full = max(incorrect_scores_full) if incorrect_scores_full \
        else 0.0

    if correct_score_full > 0 and correct_score_full > max_incorrect_full:
        return True, "full_response"

    return False, "no_match"


def main():
    parser = argparse.ArgumentParser(
        description="Score GPQA Diamond model outputs")
    parser.add_argument("--output-file", type=str, required=True,
                        help="Path to the JSON output from "
                        "test_mtp_acceptance_rate_pz.py")
    parser.add_argument("--dataset", type=str, default="Idavidrein/gpqa",
                        help="HuggingFace dataset name")
    parser.add_argument("--subset", type=str, default="gpqa_diamond",
                        help="Dataset subset/config")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-problem details")
    args = parser.parse_args()

    # Load model outputs
    with open(args.output_file) as f:
        records = json.load(f)

    # Auto-detect scoring mode: MCQ if records contain "correct_answer"
    mcq_mode = len(records) > 0 and "correct_answer" in records[0]

    if mcq_mode:
        print("Detected MCQ mode (correct_answer field present)")
    else:
        print("Detected legacy free-form mode (no correct_answer field)")

    # Load ground-truth dataset only for legacy mode
    gt = None
    if not mcq_mode:
        ds = load_dataset(args.dataset, args.subset, split=args.split)
        gt = build_gt_lookup(ds)

    correct = 0
    total = len(records)
    no_gt = 0
    no_parse = 0
    method_counts: dict[str, int] = {}
    results = []

    for i, rec in enumerate(records):
        prompt = rec["prompt"].strip()
        # Use only the "response" field; never the "thinking" field
        raw_response = rec.get("response", rec.get("output", ""))
        # Safety: strip any residual <think> blocks embedded in response
        response_text = strip_think_blocks(raw_response)

        if mcq_mode:
            # --- MCQ scoring: compare extracted letter to ground truth ---
            expected_letter = rec["correct_answer"]
            predicted_letter = extract_answer_letter(response_text)

            if predicted_letter is None:
                is_correct = False
                method = "no_parse"
                no_parse += 1
            elif predicted_letter == expected_letter:
                is_correct = True
                method = "letter_match"
            else:
                is_correct = False
                method = "letter_mismatch"

            expected_display = f"({expected_letter})"
            if predicted_letter:
                expected_display += f" got ({predicted_letter})"

        else:
            # --- Legacy free-form scoring ---
            gt_entry = gt.get(prompt)
            if gt_entry is None:
                # Fuzzy match by prefix
                for gt_prompt, entry in gt.items():
                    if gt_prompt[:100] == prompt[:100]:
                        gt_entry = entry
                        break

            if gt_entry is None:
                is_correct = False
                method = "no_gt"
                no_gt += 1
                expected_display = "N/A"
            else:
                is_correct, method = match_answer(response_text, gt_entry)
                expected_display = gt_entry["correct_answer"][:50]

        method_counts[method] = method_counts.get(method, 0) + 1

        if is_correct:
            correct += 1

        results.append({
            "idx": i,
            "method": method,
            "expected": expected_display,
            "correct": is_correct,
        })

        if args.verbose:
            status = "OK" if is_correct else "WRONG"
            print(f"[{i:3d}] {status:5s}  method={method:<16s}  "
                  f"expected={expected_display}  "
                  f"prompt={prompt[:60]}...")

    accuracy = correct / total * 100 if total > 0 else 0.0
    print()
    print("=" * 60)
    print(f"Dataset : {args.dataset} / {args.subset}")
    print(f"Output  : {args.output_file}")
    print(f"Mode    : {'MCQ' if mcq_mode else 'free-form (legacy)'}")
    print(f"Total   : {total}")
    print(f"Correct : {correct}")
    print(f"Accuracy: {accuracy:.1f}% ({correct}/{total})")
    if no_gt:
        print(f"WARNING : {no_gt} prompts had no ground-truth match")
    if no_parse:
        print(f"WARNING : {no_parse} responses could not be parsed "
              f"for answer letter")
    print(f"Methods : {method_counts}")
    print("=" * 60)


if __name__ == "__main__":
    main()