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

#Thinking mode for general tasks: temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0

#python scripts/test_mtp_acceptance_rate_pz.py \
#        --model-dir /public/panyu/hf/ckpt/Qwen/Qwen3.5-35B-A3B \
#        --dataset Idavidrein/gpqa --subset gpqa_diamond --split train \
#        --text-column Question \
#        --mode chat \
#        --enable-thinking \
#        --reasoning-parser qwen3 \
#        --max-tokens 32768 \
#        --max-model-len 262144 \
#        --tp 8 \
#        --num-spec-tokens 2 \
#        --max-num-seqs 256 \
#        --temp 0.0 \
#        --save-output /extra_panyu/output_text_pz/Qwen/Qwen3.5-35B-A3B/gpqa_diamond/output-think-32768.json
#
#python scripts/test_mtp_acceptance_rate_pz.py \
#        --model-dir /public/panyu/hf/ckpt/Qwen/Qwen3.5-35B-A3B \
#        --dataset Idavidrein/gpqa --subset gpqa_diamond --split train \
#        --text-column Question \
#        --mode chat \
#        --enable-thinking \
#        --reasoning-parser qwen3 \
#        --max-tokens 32768 \
#        --max-model-len 262144 \
#        --tp 8 \
#        --num-spec-tokens 2 \
#        --max-num-seqs 256 \
#        --temp 1.0 \
#        --top-p 0.95 \
#        --top-k 20 \
#        --min-p 0.0 \
#        --presence-penalty 1.5 \
#        --repetition-penalty 1.0 \
#        --save-output /extra_panyu/output_text_pz/Qwen/Qwen3.5-35B-A3B/gpqa_diamond/output-think-32768-temp1.json

python scripts/test_mtp_acceptance_rate_pz.py \
        --model-dir /public/panyu/hf/ckpt/Qwen/Qwen3.5-35B-A3B \
        --dataset Idavidrein/gpqa --subset gpqa_diamond --split train \
        --format mcq \
        --mode chat \
        --enable-thinking \
        --reasoning-parser qwen3 \
        --max-tokens 32768 \
        --max-model-len 262144 \
        --tp 8 \
        --num-spec-tokens 2 \
        --max-num-seqs 256 \
        --temp 0.6 \
        --top-p 0.95 \
        --top-k 20 \
        --min-p 0.0 \
        --presence-penalty 0.0 \
        --repetition-penalty 1.0 \
        --save-output /extra_panyu/output_text_pz/Qwen/Qwen3.5-35B-A3B/gpqa_diamond/output-think-32768-temp0.6.json
