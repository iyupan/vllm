#!/bin/bash
# Run MTP acceptance rate evaluation on a SPEED-Bench parquet.
#
# Usage:
#   bash scripts/infer_mtp_benchmark_speed.sh \
#       --parquet-path /extra_panyu/data/speed/qualitative/test.parquet \
#       --speed-config qualitative --temp 0.6
#
#   bash scripts/infer_mtp_benchmark_speed.sh \
#       --parquet-path /extra_panyu/data/speed/throughput_8k \
#       --speed-config throughput_8k --multi-turn

set -euo pipefail

# ======================== Defaults ========================
MODEL_DIR="/public/panyu/hf/ckpt/Qwen/Qwen3.5-35B-A3B"
OUTPUT_BASE="/extra_panyu/output_text_pz"
DATA_BASE="/extra_panyu/data/speed"
PARQUET_PATH=""
SPEED_CONFIG="qualitative"
TEMP="0.6"
NUM_SPEC_TOKENS=3
TP=8
MAX_TOKENS=32768
MAX_MODEL_LEN=262144
MAX_NUM_SEQS=256
MODE="chat"
ENABLE_THINKING=true
REASONING_PARSER="qwen3"
MULTI_TURN=false
NUM_PROMPTS=""
SAVE_PROMPT=false

# ======================== Parse Args ========================
OVERRIDE_TOP_P=""
OVERRIDE_TOP_K=""
OVERRIDE_MIN_P=""
OVERRIDE_PRESENCE_PENALTY=""
OVERRIDE_REPETITION_PENALTY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --parquet-path)       PARQUET_PATH="$2";               shift 2 ;;
        --speed-config)       SPEED_CONFIG="$2";               shift 2 ;;
        --data-base)          DATA_BASE="$2";                  shift 2 ;;
        --model-dir)          MODEL_DIR="$2";                  shift 2 ;;
        --output-base)        OUTPUT_BASE="$2";                shift 2 ;;
        --temp)               TEMP="$2";                       shift 2 ;;
        --num-spec-tokens)    NUM_SPEC_TOKENS="$2";            shift 2 ;;
        --tp)                 TP="$2";                         shift 2 ;;
        --max-tokens)         MAX_TOKENS="$2";                 shift 2 ;;
        --max-model-len)      MAX_MODEL_LEN="$2";              shift 2 ;;
        --max-num-seqs)       MAX_NUM_SEQS="$2";               shift 2 ;;
        --num-prompts)        NUM_PROMPTS="$2";                shift 2 ;;
        --top-p)              OVERRIDE_TOP_P="$2";             shift 2 ;;
        --top-k)              OVERRIDE_TOP_K="$2";             shift 2 ;;
        --min-p)              OVERRIDE_MIN_P="$2";             shift 2 ;;
        --presence-penalty)   OVERRIDE_PRESENCE_PENALTY="$2";  shift 2 ;;
        --repetition-penalty) OVERRIDE_REPETITION_PENALTY="$2"; shift 2 ;;
        --mode)               MODE="$2";                       shift 2 ;;
        --no-thinking)        ENABLE_THINKING=false;           shift ;;
        --multi-turn)         MULTI_TURN=true;                 shift ;;
        --save-prompt)        SAVE_PROMPT=true;                shift ;;
        --help|-h)
            cat <<EOF
Usage: $0 [OPTIONS]

SPEED-Bench MTP acceptance-rate evaluation on a pre-prepared parquet.
If --parquet-path is omitted, defaults to
\${DATA_BASE}/\${SPEED_CONFIG}/test.parquet.

Common options:
  --parquet-path PATH        Override default; *.parquet file or directory
  --speed-config NAME        SPEED-Bench config; also picks default parquet
                             subdirectory (default: qualitative)
  --data-base PATH           Default parquet root (default: $DATA_BASE)
  --model-dir PATH           Model directory
  --output-base PATH         Output root (default: $OUTPUT_BASE)
  --temp FLOAT               Temperature (default: 0.6)
  --num-spec-tokens INT      Speculative tokens (default: 3)
  --tp INT                   Tensor parallelism (default: 8)
  --max-tokens INT           Max output tokens per request
  --max-model-len INT        Max model length
  --max-num-seqs INT         Max sequences
  --num-prompts INT          Limit requests
  --multi-turn               Iterate every turn (default: only turns[0])
  --mode MODE                chat or completion (default: chat)
  --no-thinking              Disable thinking mode (enabled by default)
  --save-prompt              Save chat-template-rendered prompts in output
                             records (default off)

Sampling overrides (take precedence over temperature presets):
  --top-p, --top-k, --min-p, --presence-penalty, --repetition-penalty

Temperature presets (--temp):
  0.0  greedy, no sampling params
  0.6  top_p=0.95 top_k=20 min_p=0.0 presence=0.0 repetition=1.0
  1.0  top_p=0.95 top_k=20 min_p=0.0 presence=1.5 repetition=1.0
  *    other temps default to greedy
EOF
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "$PARQUET_PATH" ]; then
    PARQUET_PATH="${DATA_BASE}/${SPEED_CONFIG}/test.parquet"
fi

if [ ! -e "$PARQUET_PATH" ]; then
    echo "Error: parquet path does not exist: $PARQUET_PATH"
    echo "Either prepare data at this location or pass --parquet-path explicitly."
    exit 1
fi

# ======================== Temperature Presets ========================
case "$TEMP" in
    0.0|0)
        TOP_P=""; TOP_K=""; MIN_P=""
        PRESENCE_PENALTY=""; REPETITION_PENALTY=""
        ;;
    0.6)
        TOP_P=0.95; TOP_K=20; MIN_P=0.0
        PRESENCE_PENALTY=0.0; REPETITION_PENALTY=1.0
        ;;
    1.0|1)
        TOP_P=0.95; TOP_K=20; MIN_P=0.0
        PRESENCE_PENALTY=1.5; REPETITION_PENALTY=1.0
        ;;
    *)
        TOP_P=""; TOP_K=""; MIN_P=""
        PRESENCE_PENALTY=""; REPETITION_PENALTY=""
        ;;
esac

[ -n "$OVERRIDE_TOP_P" ]              && TOP_P="$OVERRIDE_TOP_P"
[ -n "$OVERRIDE_TOP_K" ]              && TOP_K="$OVERRIDE_TOP_K"
[ -n "$OVERRIDE_MIN_P" ]              && MIN_P="$OVERRIDE_MIN_P"
[ -n "$OVERRIDE_PRESENCE_PENALTY" ]   && PRESENCE_PENALTY="$OVERRIDE_PRESENCE_PENALTY"
[ -n "$OVERRIDE_REPETITION_PENALTY" ] && REPETITION_PENALTY="$OVERRIDE_REPETITION_PENALTY"

# ======================== Build Output Path ========================
MODEL_NAME=$(basename "$MODEL_DIR")
OUTPUT_DIR="${OUTPUT_BASE}/${MODEL_NAME}/speed-bench/${SPEED_CONFIG}"

if $ENABLE_THINKING; then
    THINK_TAG="think"
else
    THINK_TAG="nothink"
fi

if $MULTI_TURN; then
    TURN_TAG="multi"
else
    TURN_TAG="single"
fi

RUN_NAME="output-${TURN_TAG}-${THINK_TAG}-${MAX_TOKENS}-spec${NUM_SPEC_TOKENS}-temp${TEMP}"

[ -n "$OVERRIDE_TOP_P" ]              && RUN_NAME="${RUN_NAME}-topp${OVERRIDE_TOP_P}"
[ -n "$OVERRIDE_TOP_K" ]              && RUN_NAME="${RUN_NAME}-topk${OVERRIDE_TOP_K}"
[ -n "$OVERRIDE_MIN_P" ]              && RUN_NAME="${RUN_NAME}-minp${OVERRIDE_MIN_P}"
[ -n "$OVERRIDE_PRESENCE_PENALTY" ]   && RUN_NAME="${RUN_NAME}-pp${OVERRIDE_PRESENCE_PENALTY}"
[ -n "$OVERRIDE_REPETITION_PENALTY" ] && RUN_NAME="${RUN_NAME}-rp${OVERRIDE_REPETITION_PENALTY}"

[ "$MODE" != "chat" ]                 && RUN_NAME="${RUN_NAME}-${MODE}"

# Per-category outputs are written as <category>.json inside this directory.
OUTPUT_RUN_DIR="${OUTPUT_DIR}/${RUN_NAME}"

# ======================== Build Command ========================
CMD=(
    python scripts/test_mtp_acceptance_rate_speed.py
    --model-dir "$MODEL_DIR"
    --parquet-path "$PARQUET_PATH"
    --mode "$MODE"
    --reasoning-parser "$REASONING_PARSER"
    --max-tokens "$MAX_TOKENS"
    --max-model-len "$MAX_MODEL_LEN"
    --tp "$TP"
    --num-spec-tokens "$NUM_SPEC_TOKENS"
    --max-num-seqs "$MAX_NUM_SEQS"
    --temp "$TEMP"
    --save-output "$OUTPUT_RUN_DIR"
)

[ -n "$NUM_PROMPTS" ] && CMD+=(--num-prompts "$NUM_PROMPTS")

if $ENABLE_THINKING; then
    CMD+=(--enable-thinking)
fi

if $MULTI_TURN; then
    CMD+=(--multi-turn)
fi

if $SAVE_PROMPT; then
    CMD+=(--save-prompt)
fi

[ -n "$TOP_P" ]              && CMD+=(--top-p "$TOP_P")
[ -n "$TOP_K" ]              && CMD+=(--top-k "$TOP_K")
[ -n "$MIN_P" ]              && CMD+=(--min-p "$MIN_P")
[ -n "$PRESENCE_PENALTY" ]   && CMD+=(--presence-penalty "$PRESENCE_PENALTY")
[ -n "$REPETITION_PENALTY" ] && CMD+=(--repetition-penalty "$REPETITION_PENALTY")

# ======================== Run ========================
echo "============================================"
echo "Parquet       : $PARQUET_PATH"
echo "Speed config  : $SPEED_CONFIG"
echo "Model         : $MODEL_DIR"
echo "Temperature   : $TEMP"
echo "Spec tokens   : $NUM_SPEC_TOKENS"
echo "TP            : $TP"
echo "Thinking      : $ENABLE_THINKING"
echo "Multi-turn    : $MULTI_TURN"
echo "Output dir    : $OUTPUT_RUN_DIR"
echo "============================================"

mkdir -p "$OUTPUT_RUN_DIR"
exec "${CMD[@]}"