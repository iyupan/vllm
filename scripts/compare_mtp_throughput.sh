#!/bin/bash
# Compare MTP sampling methods with one generation call and thinking enabled.

set -euo pipefail

MODEL_DIR="/public/panyu/hf/ckpt/Qwen/Qwen3.5-35B-A3B"
OUTPUT_DIR="/extra_panyu/output_text_pz/mtp_throughput"
PYTHON_BIN="${VLLM_BENCHMARK_PYTHON:-.venv/bin/python}"
DATASET="aime25"
METHOD="both"
TEMP="1.0"
TOP_P="0.95"
TOP_K="20"
MIN_P="0.0"
PRESENCE_PENALTY="1.5"
REPETITION_PENALTY="1.0"
NUM_SPEC_TOKENS=3
TP=8
MAX_TOKENS=32768
MAX_MODEL_LEN=65536
MAX_NUM_SEQS=8
NUM_PROMPTS=30
RUNS=1
SEED=42

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Runs a single-call MTP throughput comparison with thinking enabled."
    echo ""
    echo "Options:"
    echo "  --dataset NAME          aime25 or gpqa (default: aime25)"
    echo "  --method NAME           both, greedy, or probabilistic (default: both)"
    echo "  --model-dir PATH        Model directory"
    echo "  --output-dir PATH       Log directory"
    echo "  --python PATH           Python executable (default: .venv/bin/python)"
    echo "  --runs INT              Runs per method (default: 1)"
    echo "  --num-prompts INT       Number of prompts (default: 30)"
    echo "  --max-num-seqs INT      Scheduler concurrency limit (default: 8)"
    echo "  --max-tokens INT        Maximum output tokens per prompt (default: 32768)"
    echo "  --max-model-len INT     Maximum model length (default: 65536)"
    echo "  --num-spec-tokens INT   Number of MTP draft tokens (default: 3)"
    echo "  --tp INT                Tensor parallel size (default: 8)"
    echo "  --seed INT              Sampling seed (default: 42)"
    echo "  --temp FLOAT            Temperature (default: 1.0)"
    echo "  --top-p FLOAT           Top-p (default: 0.95)"
    echo "  --top-k INT             Top-k (default: 20)"
    echo "  --min-p FLOAT           Min-p (default: 0.0)"
    echo "  --presence-penalty F    Presence penalty (default: 1.5)"
    echo "  --repetition-penalty F  Repetition penalty (default: 1.0)"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET="$2"; shift 2 ;;
        --method) METHOD="$2"; shift 2 ;;
        --model-dir) MODEL_DIR="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --python) PYTHON_BIN="$2"; shift 2 ;;
        --runs) RUNS="$2"; shift 2 ;;
        --num-prompts) NUM_PROMPTS="$2"; shift 2 ;;
        --max-num-seqs) MAX_NUM_SEQS="$2"; shift 2 ;;
        --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
        --num-spec-tokens) NUM_SPEC_TOKENS="$2"; shift 2 ;;
        --tp) TP="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --temp) TEMP="$2"; shift 2 ;;
        --top-p) TOP_P="$2"; shift 2 ;;
        --top-k) TOP_K="$2"; shift 2 ;;
        --min-p) MIN_P="$2"; shift 2 ;;
        --presence-penalty) PRESENCE_PENALTY="$2"; shift 2 ;;
        --repetition-penalty) REPETITION_PENALTY="$2"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

case "$METHOD" in
    both) METHODS=(greedy probabilistic) ;;
    greedy|probabilistic) METHODS=("$METHOD") ;;
    *) echo "Invalid method: $METHOD"; exit 1 ;;
esac

case "$DATASET" in
    aime25)
        HF_DATASET="MathArena/aime_2025"
        SUBSET=""
        SPLIT="train"
        TEXT_COLUMN="problem"
        FORMAT="raw"
        ;;
    gpqa)
        HF_DATASET="Idavidrein/gpqa"
        SUBSET="gpqa_diamond"
        SPLIT="train"
        TEXT_COLUMN=""
        FORMAT="mcq"
        ;;
    *) echo "Invalid dataset: $DATASET"; exit 1 ;;
esac

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python executable not found: $PYTHON_BIN"
    echo "Activate the project environment or pass --python PATH."
    exit 1
fi

export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
if [[ "$(basename "$MODEL_DIR")" == *Qwen3.5* ]]; then
    export VLLM_USE_V2_MODEL_RUNNER=0
fi

RUN_TAG=$(date +%Y%m%d-%H%M%S)
RUN_DIR="${OUTPUT_DIR}/${DATASET}-${RUN_TAG}"
SUMMARY_FILE="${RUN_DIR}/runs.tsv"
mkdir -p "$RUN_DIR"
printf "run\tmethod\toutput_tokens\tthinking_tokens\tresponse_tokens\tlength_capped\tthroughput\ttime\tacceptance\tmean_accept_len\tdraft_token_rate\tms_per_draft\n" > "$SUMMARY_FILE"

run_benchmark() {
    local run=$1
    local method=$2
    local log_file="${RUN_DIR}/${method}-run${run}.log"
    local cmd=(
        "$PYTHON_BIN" scripts/test_mtp_throughput.py
        --model-dir "$MODEL_DIR"
        --dataset "$HF_DATASET"
        --split "$SPLIT"
        --enable-thinking
        --max-tokens "$MAX_TOKENS"
        --max-model-len "$MAX_MODEL_LEN"
        --tp "$TP"
        --num-spec-tokens "$NUM_SPEC_TOKENS"
        --rejection-sample-method standard
        --draft-sample-method "$method"
        --max-num-seqs "$MAX_NUM_SEQS"
        --num-prompts "$NUM_PROMPTS"
        --temp "$TEMP"
        --top-p "$TOP_P"
        --top-k "$TOP_K"
        --min-p "$MIN_P"
        --presence-penalty "$PRESENCE_PENALTY"
        --repetition-penalty "$REPETITION_PENALTY"
        --seed "$SEED"
    )

    if [[ "$FORMAT" == "mcq" ]]; then
        cmd+=(--format mcq)
    elif [[ -n "$TEXT_COLUMN" ]]; then
        cmd+=(--text-column "$TEXT_COLUMN")
    fi
    if [[ -n "$SUBSET" ]]; then
        cmd+=(--subset "$SUBSET")
    fi

    echo "Running method=$method run=$run; log=$log_file"
    "${cmd[@]}" 2>&1 | tee "$log_file"

    local output_tokens thinking_tokens response_tokens length_capped
    local throughput elapsed acceptance mean_accept_len draft_token_rate
    local ms_per_draft
    output_tokens=$(awk '/^Total output tokens:/ {print $NF; exit}' "$log_file")
    thinking_tokens=$(awk '/^Thinking tokens:/ {print $NF; exit}' "$log_file")
    response_tokens=$(awk '/^Response tokens:/ {print $NF; exit}' "$log_file")
    length_capped=$(awk '/^Length-capped requests:/ {print $NF; exit}' "$log_file")
    throughput=$(awk '/^Total throughput:/ {print $(NF - 1); exit}' "$log_file")
    elapsed=$(awk '/^Total inference time:/ {gsub("s", "", $NF); print $NF; exit}' "$log_file")
    acceptance=$(awk '/^  Acceptance rate:/ {gsub("%", "", $NF); print $NF; exit}' "$log_file")
    mean_accept_len=$(awk '/^  Mean accept length:/ {print $NF; exit}' "$log_file")
    draft_token_rate=$(awk '/^  Draft token rate:/ {print $(NF - 1); exit}' "$log_file")
    ms_per_draft=$(awk '/^  Amortized draft time:/ {print $(NF - 1); exit}' "$log_file")
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$run" "$method" "$output_tokens" "$thinking_tokens" \
        "$response_tokens" "$length_capped" "$throughput" "$elapsed" \
        "$acceptance" "$mean_accept_len" "$draft_token_rate" \
        "$ms_per_draft" >> "$SUMMARY_FILE"
}

echo "============================================================"
echo "Single-call MTP throughput benchmark with thinking"
echo "Dataset:          $DATASET"
echo "Methods:          ${METHODS[*]}"
echo "Runs per method:  $RUNS"
echo "Prompts:          $NUM_PROMPTS"
echo "Max output tokens: $MAX_TOKENS per prompt"
echo "Max concurrency:  $MAX_NUM_SEQS"
echo "Spec tokens:      $NUM_SPEC_TOKENS"
echo "Seed:             $SEED"
echo "Logs:             $RUN_DIR"
echo "============================================================"

for ((run = 1; run <= RUNS; run++)); do
    if ((run % 2 == 0)) && [[ "$METHOD" == "both" ]]; then
        ORDER=(probabilistic greedy)
    else
        ORDER=("${METHODS[@]}")
    fi
    for method in "${ORDER[@]}"; do
        run_benchmark "$run" "$method"
    done
done

echo ""
echo "Per-run results:"
if command -v column >/dev/null 2>&1; then
    column -t -s $'\t' "$SUMMARY_FILE"
else
    sed 's/\t/  /g' "$SUMMARY_FILE"
fi

echo ""
echo "Aggregate results:"
awk -F '\t' '
    NR > 1 {
        count[$2]++
        output_tokens[$2] += $3
        thinking_tokens[$2] += $4
        response_tokens[$2] += $5
        elapsed[$2] += $8
        acceptance[$2] += $9
        mean_len[$2] += $10
        draft_ms[$2] += $12
    }
    END {
        printf "%-14s %12s %12s %12s %12s %10s %12s %16s %14s\n", \
            "method", "throughput", "output_tok", "thinking_tok", \
            "response_tok", "time", "acceptance", "mean_accept_len", \
            "ms_per_draft"
        for (method in count) {
            n = count[method]
            printf "%-14s %12.1f %12.0f %12.0f %12.0f %10.2f %11.2f%% %16.4f %14.3f\n", \
                method, output_tokens[method] / elapsed[method], \
                output_tokens[method], thinking_tokens[method], \
                response_tokens[method], elapsed[method], \
                acceptance[method] / n, mean_len[method] / n, \
                draft_ms[method] / n
        }
    }
' "$SUMMARY_FILE"
