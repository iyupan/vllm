#!/bin/bash
# Score GPQA Diamond model outputs against ground-truth answers.
#
# Usage:
#   bash scripts/run_score_gpqa.sh <output-file>
#   bash scripts/run_score_gpqa.sh  # uses default path

set -euo pipefail

DEFAULT_OUTPUT="/extra_panyu/output_text_pz/Qwen3.5-35B-A3B/gpqa_diamond/output-think-4096-spec3-temp0.0.json"
#DEFAULT_OUTPUT="/extra_panyu/output_text_pz/Qwen3.5-35B-A3B-Base/gpqa_diamond/output-nothink-32768-spec1-temp0.0-completion.json"

OUTPUT_FILE="${1:-$DEFAULT_OUTPUT}"

if [ ! -f "$OUTPUT_FILE" ]; then
    echo "Error: output file not found: $OUTPUT_FILE"
    echo "Usage: bash scripts/run_score_gpqa.sh <output-file>"
    exit 1
fi

echo "Scoring: $OUTPUT_FILE"

python scripts/score_gpqa.py \
    --output-file "$OUTPUT_FILE" \
    --dataset Idavidrein/gpqa \
    --subset gpqa_diamond \
    --split train \
    --verbose
