#!/bin/bash
# Test MTP acceptance rate on GPQA Diamond dataset

#MODEL_DIR=${1:?"Usage: $0 <model_dir> [num_spec_tokens] [tp]"}
#NUM_SPEC_TOKENS=${2:-3}
#TP=${3:-1}
#
#python scripts/test_mtp_acceptance_rate.py \
#    --model-dir "$MODEL_DIR" \
#    --dataset Idavidrein/gpqa \
#    --subset gpqa_diamond \
#    --split train \
#    --text-column Question \
#    --num-spec-tokens "$NUM_SPEC_TOKENS" \
#    --tp "$TP" \
#    --output-len 1024 \
#    --temp 0.0

python scripts/test_mtp_acceptance_rate.py \
        --model-dir /public/panyu/hf/ckpt/Qwen/Qwen3.5-35B-A3B \
        --dataset Idavidrein/gpqa --subset gpqa_diamond --split train \
        --text-column Question \
        --max-model-len 16384 \
        --tp 8 \
        --num-spec-tokens 2 \
        --output-len 4096 \
        --max-num-seqs 256 \
        --temp 0.0 \
        --output-dir /extra_panyu/output_text/Qwen/Qwen3.5-35B-A3B/gpqa_diamond

#        --enable-thinking \