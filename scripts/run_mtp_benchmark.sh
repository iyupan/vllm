#!/bin/bash
# Unified script for running MTP acceptance rate tests on different datasets
# with different inference parameters.
#
# Usage:
#   bash scripts/run_mtp_benchmark.sh --dataset aime25 --temp 0.6
#   bash scripts/run_mtp_benchmark.sh --dataset gpqa --temp 0.0 --num-spec-tokens 3
#   bash scripts/run_mtp_benchmark.sh --dataset aime25 --temp 1.0 --model-dir /path/to/model

set -euo pipefail

# ======================== Defaults ========================
MODEL_DIR="/public/panyu/hf/ckpt/Qwen/Qwen3.5-35B-A3B"
OUTPUT_BASE="/extra_panyu/output_text_pz"
DATASET="aime25"
TEMP="0.6"
NUM_SPEC_TOKENS=3
TP=8
MAX_TOKENS=32768
MAX_MODEL_LEN=262144
MAX_NUM_SEQS=256
MODE="chat"
ENABLE_THINKING=true
REASONING_PARSER="qwen3"

# ======================== Parse Args ========================
# Use OVERRIDE_* to track user-explicit overrides for sampling params.
OVERRIDE_TOP_P=""
OVERRIDE_TOP_K=""
OVERRIDE_MIN_P=""
OVERRIDE_PRESENCE_PENALTY=""
OVERRIDE_REPETITION_PENALTY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)            DATASET="$2";                    shift 2 ;;
        --model-dir)          MODEL_DIR="$2";                  shift 2 ;;
        --output-base)        OUTPUT_BASE="$2";                shift 2 ;;
        --temp)               TEMP="$2";                       shift 2 ;;
        --num-spec-tokens)    NUM_SPEC_TOKENS="$2";            shift 2 ;;
        --tp)                 TP="$2";                         shift 2 ;;
        --max-tokens)         MAX_TOKENS="$2";                 shift 2 ;;
        --max-model-len)      MAX_MODEL_LEN="$2";              shift 2 ;;
        --max-num-seqs)       MAX_NUM_SEQS="$2";               shift 2 ;;
        --top-p)              OVERRIDE_TOP_P="$2";             shift 2 ;;
        --top-k)              OVERRIDE_TOP_K="$2";             shift 2 ;;
        --min-p)              OVERRIDE_MIN_P="$2";             shift 2 ;;
        --presence-penalty)   OVERRIDE_PRESENCE_PENALTY="$2";  shift 2 ;;
        --repetition-penalty) OVERRIDE_REPETITION_PENALTY="$2"; shift 2 ;;
        --mode)               MODE="$2";                       shift 2 ;;
        --no-thinking)        ENABLE_THINKING=false;           shift ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Datasets: aime25, gpqa, gsm8k, mmlu, mt-bench"
            echo ""
            echo "Temperature presets (--temp):"
            echo "  0.0  greedy, no sampling params"
            echo "  0.6  top_p=0.95 top_k=20 min_p=0.0 presence=0.0 repetition=1.0"
            echo "  1.0  top_p=0.95 top_k=20 min_p=0.0 presence=1.5 repetition=1.0"
            echo "  *    other temps default to greedy (no sampling), use overrides to customize"
            echo ""
            echo "Options:"
            echo "  --dataset NAME            Dataset name (default: aime25)"
            echo "  --model-dir PATH          Model directory"
            echo "  --output-base PATH        Base output directory"
            echo "  --temp FLOAT              Temperature (default: 0.6)"
            echo "  --num-spec-tokens INT     Speculative tokens (default: 2)"
            echo "  --tp INT                  Tensor parallelism (default: 8)"
            echo "  --max-tokens INT          Max output tokens (default: 32768)"
            echo "  --max-model-len INT       Max model length (default: 262144)"
            echo "  --max-num-seqs INT        Max sequences (default: 256)"
            echo "  --mode MODE               chat or completion (default: chat)"
            echo "  --no-thinking             Disable thinking mode (enabled by default)"
            echo "  --top-p FLOAT             Override top-p"
            echo "  --top-k INT               Override top-k"
            echo "  --min-p FLOAT             Override min-p"
            echo "  --presence-penalty FLOAT   Override presence penalty"
            echo "  --repetition-penalty FLOAT Override repetition penalty"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ======================== Temperature Presets ========================
# Known presets; any other temp falls back to greedy-style (no sampling params).
case "$TEMP" in
    0.0|0)
        TOP_P=""
        TOP_K=""
        MIN_P=""
        PRESENCE_PENALTY=""
        REPETITION_PENALTY=""
        ;;
    0.6)
        TOP_P=0.95
        TOP_K=20
        MIN_P=0.0
        PRESENCE_PENALTY=0.0
        REPETITION_PENALTY=1.0
        ;;
    1.0|1)
        TOP_P=0.95
        TOP_K=20
        MIN_P=0.0
        PRESENCE_PENALTY=1.5
        REPETITION_PENALTY=1.0
        ;;
    *)
        # Unknown temp: default to no sampling params (same as greedy)
        TOP_P=""
        TOP_K=""
        MIN_P=""
        PRESENCE_PENALTY=""
        REPETITION_PENALTY=""
        ;;
esac

# Apply user overrides (take precedence over presets)
[ -n "$OVERRIDE_TOP_P" ]              && TOP_P="$OVERRIDE_TOP_P"
[ -n "$OVERRIDE_TOP_K" ]              && TOP_K="$OVERRIDE_TOP_K"
[ -n "$OVERRIDE_MIN_P" ]              && MIN_P="$OVERRIDE_MIN_P"
[ -n "$OVERRIDE_PRESENCE_PENALTY" ]   && PRESENCE_PENALTY="$OVERRIDE_PRESENCE_PENALTY"
[ -n "$OVERRIDE_REPETITION_PENALTY" ] && REPETITION_PENALTY="$OVERRIDE_REPETITION_PENALTY"

# ======================== Dataset Config ========================
# Each dataset defines: HF_DATASET, SUBSET, SPLIT, TEXT_COLUMN, FORMAT, SAVE_DIR
# FORMAT="mcq" uses shuffled (A)/(B)/(C)/(D) choices; FORMAT="raw" uses text-column as-is.
FORMAT="raw"
case "$DATASET" in
    aime25)
        HF_DATASET="MathArena/aime_2025"
        SUBSET=""
        SPLIT="train"
        TEXT_COLUMN="problem"
        SAVE_DIR="aime25"
        ;;
    gpqa)
        HF_DATASET="Idavidrein/gpqa"
        SUBSET="gpqa_diamond"
        SPLIT="train"
        TEXT_COLUMN=""
        FORMAT="mcq"
        SAVE_DIR="gpqa_diamond"
        ;;
    gsm8k)
        HF_DATASET="openai/gsm8k"
        SUBSET="main"
        SPLIT="test"
        TEXT_COLUMN="question"
        SAVE_DIR="gsm8k"
        ;;
    mmlu)
        HF_DATASET="cais/mmlu"
        SUBSET="all"
        SPLIT="test"
        TEXT_COLUMN="question"
        SAVE_DIR="mmlu"
        ;;
    mt-bench)
        HF_DATASET="philschmid/mt-bench"
        SUBSET=""
        SPLIT="train"
        TEXT_COLUMN="turns"
        SAVE_DIR="mt-bench"
        ;;
    *)
        echo "Error: unknown dataset '$DATASET'"
        echo "Supported: aime25, gpqa, gsm8k, mmlu, mt-bench"
        exit 1
        ;;
esac

# ======================== Build Output Path ========================
MODEL_NAME=$(basename "$MODEL_DIR")
OUTPUT_DIR="${OUTPUT_BASE}/${MODEL_NAME}/${SAVE_DIR}"

# Build filename with all effective parameters
if $ENABLE_THINKING; then
    THINK_TAG="think"
else
    THINK_TAG="nothink"
fi

FNAME="output-${THINK_TAG}-${MAX_TOKENS}-spec${NUM_SPEC_TOKENS}-temp${TEMP}"

# Only append user-overridden sampling params (not from presets)
[ -n "$OVERRIDE_TOP_P" ]              && FNAME="${FNAME}-topp${OVERRIDE_TOP_P}"
[ -n "$OVERRIDE_TOP_K" ]              && FNAME="${FNAME}-topk${OVERRIDE_TOP_K}"
[ -n "$OVERRIDE_MIN_P" ]              && FNAME="${FNAME}-minp${OVERRIDE_MIN_P}"
[ -n "$OVERRIDE_PRESENCE_PENALTY" ]   && FNAME="${FNAME}-pp${OVERRIDE_PRESENCE_PENALTY}"
[ -n "$OVERRIDE_REPETITION_PENALTY" ] && FNAME="${FNAME}-rp${OVERRIDE_REPETITION_PENALTY}"

# Append mode if not default
[ "$MODE" != "chat" ]                 && FNAME="${FNAME}-${MODE}"

OUTPUT_FILE="${OUTPUT_DIR}/${FNAME}.json"

# ======================== Build Command ========================
CMD=(
    python scripts/test_mtp_acceptance_rate_pz.py
    --model-dir "$MODEL_DIR"
    --dataset "$HF_DATASET"
    --split "$SPLIT"
    --mode "$MODE"
    --reasoning-parser "$REASONING_PARSER"
    --max-tokens "$MAX_TOKENS"
    --max-model-len "$MAX_MODEL_LEN"
    --tp "$TP"
    --num-spec-tokens "$NUM_SPEC_TOKENS"
    --max-num-seqs "$MAX_NUM_SEQS"
    --temp "$TEMP"
    --save-output "$OUTPUT_FILE"
)

# MCQ format (e.g. gpqa) vs raw text-column
if [ "$FORMAT" = "mcq" ]; then
    CMD+=(--format mcq)
elif [ -n "$TEXT_COLUMN" ]; then
    CMD+=(--text-column "$TEXT_COLUMN")
fi

# Add subset if defined
if [ -n "$SUBSET" ]; then
    CMD+=(--subset "$SUBSET")
fi

# Add thinking flag
if $ENABLE_THINKING; then
    CMD+=(--enable-thinking)
fi

# Add sampling params (only for non-greedy)
if [ -n "$TOP_P" ]; then
    CMD+=(--top-p "$TOP_P")
fi
if [ -n "$TOP_K" ]; then
    CMD+=(--top-k "$TOP_K")
fi
if [ -n "$MIN_P" ]; then
    CMD+=(--min-p "$MIN_P")
fi
if [ -n "$PRESENCE_PENALTY" ]; then
    CMD+=(--presence-penalty "$PRESENCE_PENALTY")
fi
if [ -n "$REPETITION_PENALTY" ]; then
    CMD+=(--repetition-penalty "$REPETITION_PENALTY")
fi

# ======================== Run ========================
echo "============================================"
echo "Dataset       : $DATASET ($HF_DATASET)"
echo "Format        : $FORMAT"
echo "Model         : $MODEL_DIR"
echo "Temperature   : $TEMP"
echo "Spec tokens   : $NUM_SPEC_TOKENS"
echo "TP            : $TP"
echo "Thinking      : $ENABLE_THINKING"
echo "Output        : $OUTPUT_FILE"
echo "============================================"

mkdir -p "$OUTPUT_DIR"
exec "${CMD[@]}"