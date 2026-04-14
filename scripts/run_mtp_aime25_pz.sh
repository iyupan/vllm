#!/bin/bash
# Test MTP acceptance rate on AIME 2025 dataset

# temp=0.0 (greedy)
#python scripts/test_mtp_acceptance_rate_pz.py \
#        --model-dir /public/panyu/hf/ckpt/Qwen/Qwen3.5-35B-A3B \
#        --dataset MathArena/aime_2025 --split train \
#        --text-column problem \
#        --mode chat \
#        --enable-thinking \
#        --reasoning-parser qwen3 \
#        --max-tokens 32768 \
#        --max-model-len 262144 \
#        --tp 8 \
#        --num-spec-tokens 2 \
#        --max-num-seqs 256 \
#        --temp 0.0 \
#        --save-output /extra_panyu/output_text_pz/Qwen/Qwen3.5-35B-A3B/aime25/output-think-32768.json

# temp=1.0 (thinking mode sampling)
#python scripts/test_mtp_acceptance_rate_pz.py \
#        --model-dir /public/panyu/hf/ckpt/Qwen/Qwen3.5-35B-A3B \
#        --dataset MathArena/aime_2025 --split train \
#        --text-column problem \
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
#        --save-output /extra_panyu/output_text_pz/Qwen/Qwen3.5-35B-A3B/aime25/output-think-32768-temp1.json

# temp=0.6 (moderate sampling)
python scripts/test_mtp_acceptance_rate_pz.py \
        --model-dir /public/panyu/hf/ckpt/Qwen/Qwen3.5-35B-A3B \
        --dataset MathArena/aime_2025 --split train \
        --text-column problem \
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
        --save-output /extra_panyu/output_text_pz/Qwen/Qwen3.5-35B-A3B/aime25/output-think-32768-temp0.6.json