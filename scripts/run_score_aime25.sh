#!/bin/bash
# Score AIME 2025 model outputs against ground-truth answers.
#
# Usage:
#   bash scripts/run_score_aime25.sh <output-file>
#   bash scripts/run_score_aime25.sh  # uses default path

set -euo pipefail

DEFAULT_OUTPUT="/extra_panyu/output_text_pz/Qwen/Qwen3.5-35B-A3B/aime25/output-think-32768-spec3-temp0.6.json"

OUTPUT_FILE="${1:-$DEFAULT_OUTPUT}"

if [ ! -f "$OUTPUT_FILE" ]; then
    echo "Error: output file not found: $OUTPUT_FILE"
    echo "Usage: bash scripts/run_score_aime25.sh <output-file>"
    exit 1
fi

echo "Scoring: $OUTPUT_FILE"

python scripts/score_aime25.py \
    --output-file "$OUTPUT_FILE" \
    --dataset MathArena/aime_2025 \
    --split train \
    --verbose