# SPDX-License-Identifier: Apache-2.0
"""
Measure MTP speculative decoding acceptance rate on any HuggingFace
dataset.

Examples:

    # GPQA Diamond
    python scripts/test_mtp_acceptance_rate.py \
        --model-dir <model_path> \
        --dataset Idavidrein/gpqa --subset gpqa_diamond --split train \
        --text-column Question \
        --num-spec-tokens 3

    # MT-Bench
    python scripts/test_mtp_acceptance_rate.py \
        --model-dir <model_path> \
        --dataset philschmid/mt-bench --split train \
        --text-column turns \
        --num-spec-tokens 3

    # GSM8K
    python scripts/test_mtp_acceptance_rate.py \
        --model-dir <model_path> \
        --dataset openai/gsm8k --subset main --split test \
        --text-column question \
        --num-spec-tokens 3

    # MMLU
    python scripts/test_mtp_acceptance_rate.py \
        --model-dir <model_path> \
        --dataset cais/mmlu --subset all --split test \
        --text-column question \
        --num-spec-tokens 3

    # Use a local jsonl file (one JSON object per line with a "prompt" field)
    python scripts/test_mtp_acceptance_rate.py \
        --model-dir <model_path> \
        --dataset local --dataset-path /path/to/data.jsonl \
        --text-column prompt \
        --num-spec-tokens 3

    # Enable thinking (slow thinking / extended reasoning)
    python scripts/test_mtp_acceptance_rate.py \
        --model-dir <model_path> \
        --dataset openai/gsm8k --subset main --split test \
        --text-column question \
        --num-spec-tokens 3 \
        --enable-thinking

    # Completion mode (no chat template, text continuation)
    python scripts/test_mtp_acceptance_rate.py \
        --model-dir <model_path> \
        --dataset openai/gsm8k --subset main --split test \
        --text-column question \
        --num-spec-tokens 1 \
        --mode completion

    # Qwen3.5-35B-A3B chat + thinking, TP=8, save output
    python scripts/test_mtp_acceptance_rate.py \
        --model-dir <model_path> \
        --dataset Idavidrein/gpqa --subset gpqa_diamond --split train \
        --text-column Question \
        --num-spec-tokens 1 \
        --mode chat \
        --enable-thinking \
        --max-tokens 1024 --max-model-len 4096 \
        --tp 8 --max-num-seqs 256 \
        --save-output output.json --print-output

    # MiMo-7B-RL chat + thinking
    python scripts/test_mtp_acceptance_rate.py \
        --model-dir <model_path> \
        --dataset Idavidrein/gpqa --subset gpqa_diamond --split train \
        --text-column Question \
        --num-spec-tokens 1 \
        --mode chat \
        --enable-thinking \
        --max-tokens 4096 \
        --save-output output.json --print-output
"""

import argparse
import json
import os
import random
import time

from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.v1.metrics.reader import Counter, Vector

_MCQ_LABELS = ["A", "B", "C", "D"]


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_local_dataset(path: str, text_column: str) -> list[str]:
    """Load prompts from a local jsonl/json file."""
    texts = []
    with open(path) as f:
        if path.endswith(".json"):
            data = json.load(f)
            if isinstance(data, list):
                for row in data:
                    texts.append(str(row[text_column]))
            else:
                raise ValueError("JSON file must contain a list of objects")
        else:  # jsonl
            for line in f:
                row = json.loads(line)
                texts.append(str(row[text_column]))
    return texts


def load_hf_dataset(dataset_name: str, subset: str | None, split: str,
                    text_column: str) -> list[str]:
    """Load prompts from a HuggingFace dataset."""
    from datasets import load_dataset
    kwargs = {}
    if subset:
        kwargs["name"] = subset
    ds = load_dataset(dataset_name, split=split, **kwargs)
    print(f"Dataset columns: {ds.column_names}")
    print(f"Total samples: {len(ds)}")

    texts = []
    for row in ds:
        val = row[text_column]
        # Handle list-type columns (e.g. mt-bench "turns" is a list)
        if isinstance(val, list):
            val = "\n".join(str(v) for v in val)
        texts.append(str(val))
    return texts


def load_gpqa_as_mcq(dataset_name: str, subset: str | None, split: str,
                     seed: int = 42) -> tuple[list[str], list[str]]:
    """Load GPQA dataset and format as multiple-choice questions.

    Shuffles the four answer options per question (with a deterministic seed)
    and returns the formatted text together with the correct-answer letter.

    Returns:
        (texts, correct_letters) where texts[i] is the formatted MCQ string
        and correct_letters[i] is one of "A", "B", "C", "D".
    """
    from datasets import load_dataset
    kwargs = {}
    if subset:
        kwargs["name"] = subset
    ds = load_dataset(dataset_name, split=split, **kwargs)
    print(f"Dataset columns: {ds.column_names}")
    print(f"Total samples: {len(ds)}")

    texts = []
    correct_letters = []
    for i, row in enumerate(ds):
        question = row["Question"].strip()
        correct = row["Correct Answer"].strip()
        incorrect = [
            row["Incorrect Answer 1"].strip(),
            row["Incorrect Answer 2"].strip(),
            row["Incorrect Answer 3"].strip(),
        ]

        # Build options list with a correct-answer marker, then shuffle
        options = [(correct, True)] + [(inc, False) for inc in incorrect]
        rng = random.Random(seed + i)
        rng.shuffle(options)

        # Find which label the correct answer landed on
        correct_label = None
        lines = []
        for j, (text, is_correct) in enumerate(options):
            label = _MCQ_LABELS[j]
            lines.append(f"({label}) {text}")
            if is_correct:
                correct_label = label

        formatted = (
            "Answer the following multiple choice question. The last line "
            "of your response should be of the following format: "
            "'ANSWER: $LETTER' (without quotes) where LETTER is one of "
            "ABCD. Think step by step before answering.\n\n"
            f"{question}\n\n" + "\n".join(lines)
        )
        texts.append(formatted)
        correct_letters.append(correct_label)

    return texts, correct_letters


def apply_chat_template(tokenizer, texts: list[str],
                        enable_thinking: bool = False) -> list[str]:
    """Wrap each text in a chat template."""
    prompts = []
    for text in texts:
        messages = [{"role": "user", "content": text}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking)
        prompts.append(prompt)
    return prompts


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def split_thinking(text: str) -> tuple[str | None, str]:
    """Split thinking and response from model output.

    Handles two cases:
    - Full tags: <think>...</think>response
    - Implicit start: ...thinking...</think>response
      (when chat template already emitted <think>)
    Returns (thinking, response) or (None, text) if no thinking found.
    """
    think_end = text.find("</think>")
    if think_end == -1:
        return None, text

    think_start = text.find("<think>")
    if think_start != -1:
        thinking = text[think_start + len("<think>"):think_end].strip()
    else:
        # Chat template already emitted <think>, output starts mid-thought
        thinking = text[:think_end].strip()

    response = text[think_end + len("</think>"):].strip()
    return thinking, response


# ---------------------------------------------------------------------------
# Metrics collection
# ---------------------------------------------------------------------------

def collect_spec_decode_metrics(llm, num_spec_tokens):
    """Collect cumulative speculative decoding metrics from vLLM."""
    metrics = llm.get_metrics()

    num_drafts = 0
    num_draft_tokens = 0
    num_accepted_tokens = 0
    acceptance_counts = [0] * num_spec_tokens

    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_draft_tokens":
            assert isinstance(metric, Counter)
            num_draft_tokens += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens":
            assert isinstance(metric, Counter)
            num_accepted_tokens += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert isinstance(metric, Vector)
            for pos in range(len(metric.values)):
                if pos < num_spec_tokens:
                    acceptance_counts[pos] += metric.values[pos]

    return num_drafts, num_draft_tokens, num_accepted_tokens, acceptance_counts


def print_phase_report(phase_name, num_drafts, num_draft_tokens,
                       num_accepted_tokens, acceptance_counts,
                       num_output_tokens=0, elapsed=0.0):
    """Print acceptance rate report for a single phase."""
    print(f"\n--- {phase_name} ---")
    print(f"  Output tokens:        {num_output_tokens}")
    if elapsed > 0:
        print(f"  Inference time:       {elapsed:.2f}s")
        print(f"  Throughput:           {num_output_tokens / elapsed:.1f} tok/s")
    print(f"  Num drafts:           {num_drafts}")
    print(f"  Num draft tokens:     {num_draft_tokens}")
    print(f"  Num accepted tokens:  {num_accepted_tokens}")

    if num_draft_tokens > 0:
        rate = (num_accepted_tokens / num_draft_tokens) * 100
        print(f"  Acceptance rate:      {rate:.2f}%")
    else:
        print(f"  Acceptance rate:      N/A (no draft tokens)")

    if num_drafts > 0:
        length = 1 + (num_accepted_tokens / num_drafts)
        print(f"  Mean accept length:   {length:.4f}")
    else:
        print(f"  Mean accept length:   N/A (no drafts)")

    if acceptance_counts:
        print(f"  Per-position acceptance rate:")
        for i, count in enumerate(acceptance_counts):
            r = count / num_drafts if num_drafts > 0 else 0
            print(f"    Position {i}: {r:.4f} ({count}/{num_drafts})")


def print_report(args, num_prompts, total_output_tokens, total_elapsed,
                 total_metrics, thinking_metrics=None, response_metrics=None):
    """Print a full summary report.

    Each *_metrics is a tuple:
        (num_drafts, num_draft_tokens, num_accepted_tokens,
         acceptance_counts, num_output_tokens, elapsed)
    """
    dataset_label = args.dataset
    if args.subset:
        dataset_label += f" / {args.subset}"

    print("\n" + "=" * 60)
    print("MTP Speculative Decoding - Acceptance Rate Report")
    print("=" * 60)
    print(f"Model:                {args.model_dir}")
    print(f"Dataset:              {dataset_label}")
    print(f"Mode:                 {args.mode}")
    print(f"Num prompts:          {num_prompts}")
    print(f"Num spec tokens:      {args.num_spec_tokens}")
    print(f"Rejection sampling:   {args.rejection_sample_method}")
    print(f"Draft sampling:       {args.draft_sample_method}")
    if args.synthetic_acceptance_length is not None:
        print(f"Synthetic accept len: {args.synthetic_acceptance_length}")
    print(f"Thinking enabled:     {args.enable_thinking}")
    print(f"Total output tokens:  {total_output_tokens}")
    print(f"Total inference time: {total_elapsed:.2f}s")
    print(f"Total throughput:     {total_output_tokens / total_elapsed:.1f} tok/s")

    # Overall
    nd, ndt, nat, ac, _, _ = total_metrics
    print_phase_report("Overall", nd, ndt, nat, ac,
                       total_output_tokens, total_elapsed)

    # Thinking phase
    if thinking_metrics is not None:
        nd, ndt, nat, ac, ntok, el = thinking_metrics
        print_phase_report("Thinking Phase", nd, ndt, nat, ac, ntok, el)

    # Response phase
    if response_metrics is not None:
        nd, ndt, nat, ac, ntok, el = response_metrics
        print_phase_report("Response Phase", nd, ndt, nat, ac, ntok, el)

    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="MTP acceptance rate on any dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Dataset args
    ds = parser.add_argument_group("dataset")
    ds.add_argument("--dataset", type=str, default="Idavidrein/gpqa",
                    help="HuggingFace dataset ID, or 'local' for local file")
    ds.add_argument("--subset", type=str, default=None,
                    help="Dataset subset/config (e.g. gpqa_diamond, main)")
    ds.add_argument("--split", type=str, default="train",
                    help="Dataset split (default: train)")
    ds.add_argument("--text-column", type=str, default="Question",
                    help="Column name containing the prompt text")
    ds.add_argument("--dataset-path", type=str, default=None,
                    help="Path to local file (when --dataset=local)")
    ds.add_argument("--num-prompts", type=int, default=None,
                    help="Limit number of prompts (default: all)")
    ds.add_argument("--format", type=str, default="raw",
                    choices=["raw", "mcq"],
                    help="Prompt format: 'raw' sends the text column as-is, "
                         "'mcq' formats GPQA as multiple-choice with "
                         "shuffled (A)/(B)/(C)/(D) options (default: raw)")
    ds.add_argument("--mcq-seed", type=int, default=42,
                    help="Random seed for MCQ option shuffling (default: 42)")

    # Model args
    md = parser.add_argument_group("model")
    md.add_argument("--model-dir", type=str, required=True,
                    help="Model path (must support MTP)")
    md.add_argument("--num-spec-tokens", type=int, default=3)
    md.add_argument("--rejection-sample-method", type=str, default="standard",
                    choices=["standard", "synthetic", "block"],
                    help="Draft-token verification method (default: standard)")
    md.add_argument("--draft-sample-method", type=str, default="greedy",
                    choices=["greedy", "probabilistic"],
                    help="Draft sampling method (default: greedy)")
    md.add_argument("--synthetic-acceptance-length", type=float, default=None,
                    help="Mean acceptance length for synthetic rejection")
    md.add_argument("--tp", type=int, default=1)
    md.add_argument("--max-model-len", type=int, default=16384)
    md.add_argument("--enforce-eager", action="store_true")
    md.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    md.add_argument("--max-num-seqs", type=int, default=None)

    # Generation args
    gen = parser.add_argument_group("generation")
    gen.add_argument("--max-tokens", type=int, default=1024,
                       help="Max output tokens per request (default: 1024)")
    gen.add_argument("--temp", type=float, default=1.0)
    gen.add_argument("--top-p", type=float, default=0.95)
    gen.add_argument("--top-k", type=int, default=20)
    gen.add_argument("--min-p", type=float, default=0.0)
    gen.add_argument("--presence-penalty", type=float, default=1.5)
    gen.add_argument("--repetition-penalty", type=float, default=1.0)

    # Thinking / reasoning
    think = parser.add_argument_group("thinking")
    think.add_argument("--enable-thinking", action="store_true",
                       help="Enable thinking/reasoning mode in chat template")
    think.add_argument("--reasoning-parser", type=str, default=None,
                       help="Reasoning parser name, e.g. qwen3, deepseek_r1 "
                            "(required with --enable-thinking)")

    # Misc
    parser.add_argument("--mode", type=str, default="chat",
                        choices=["chat", "completion"],
                        help="Prompt mode: 'chat' applies chat template, "
                             "'completion' uses raw text for continuation "
                             "(default: chat)")
    parser.add_argument("--print-output", action="store_true",
                        help="Print generated text")
    parser.add_argument("--save-output", type=str, default=None,
                        help="Save outputs to a JSON file")

    return parser.parse_args()


def main():
    args = parse_args()

    if (args.rejection_sample_method == "synthetic"
            and args.synthetic_acceptance_length is None):
        raise ValueError(
            "--synthetic-acceptance-length is required when "
            "--rejection-sample-method=synthetic")
    if (args.rejection_sample_method != "synthetic"
            and args.synthetic_acceptance_length is not None):
        raise ValueError(
            "--synthetic-acceptance-length requires "
            "--rejection-sample-method=synthetic")
    if (args.rejection_sample_method == "block"
            and os.environ.get("VLLM_USE_V2_MODEL_RUNNER", "0") != "1"):
        raise ValueError(
            "Block rejection sampling requires VLLM_USE_V2_MODEL_RUNNER=1")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir,
                                              trust_remote_code=True)

    # ---- Load dataset ----
    correct_letters: list[str] | None = None

    if args.format == "mcq":
        print(f"Loading dataset {args.dataset} as MCQ "
              f"(subset={args.subset}, split={args.split}, "
              f"seed={args.mcq_seed}) ...")
        texts, correct_letters = load_gpqa_as_mcq(
            args.dataset, args.subset, args.split, seed=args.mcq_seed)
    elif args.dataset == "local":
        if not args.dataset_path:
            raise ValueError("--dataset-path is required when --dataset=local")
        print(f"Loading local dataset from {args.dataset_path} ...")
        texts = load_local_dataset(args.dataset_path, args.text_column)
    else:
        print(f"Loading dataset {args.dataset} "
              f"(subset={args.subset}, split={args.split}) ...")
        texts = load_hf_dataset(args.dataset, args.subset, args.split,
                                args.text_column)

    if args.num_prompts is not None:
        texts = texts[:args.num_prompts]
        if correct_letters is not None:
            correct_letters = correct_letters[:args.num_prompts]

    # ---- Validate thinking args ----
    if args.enable_thinking and args.mode != "chat":
        raise ValueError("--enable-thinking requires --mode=chat")

    # ---- Determine prompt mode ----
    use_chat = (args.mode == "chat")

    if use_chat:
        prompts = apply_chat_template(tokenizer, texts,
                                      enable_thinking=args.enable_thinking)
    else:
        prompts = texts

    print(f"Prepared {len(prompts)} prompts (mode={args.mode})")

    # ---- Build LLM with MTP spec decode ----
    speculative_config = {
        "method": "mtp",
        "num_speculative_tokens": args.num_spec_tokens,
        "rejection_sample_method": args.rejection_sample_method,
        "draft_sample_method": args.draft_sample_method,
    }
    if args.synthetic_acceptance_length is not None:
        speculative_config["synthetic_acceptance_length"] = (
            args.synthetic_acceptance_length)

    extra_kwargs = {}
    if args.enable_thinking and args.reasoning_parser:
        extra_kwargs["reasoning_parser"] = args.reasoning_parser

    llm = LLM(
        model=args.model_dir,
        trust_remote_code=True,
        tensor_parallel_size=args.tp,
        enforce_eager=args.enforce_eager,
        gpu_memory_utilization=args.gpu_memory_utilization,
        speculative_config=speculative_config,
        disable_log_stats=False,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        **extra_kwargs,
    )

    common_sampling_kwargs = dict(
        temperature=args.temp,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        presence_penalty=args.presence_penalty,
        repetition_penalty=args.repetition_penalty,
    )

    # ---- Run inference ----
    print(f"Running inference with MTP spec decoding "
          f"(num_spec_tokens={args.num_spec_tokens}) ...")

    if args.enable_thinking:
        # ============================================================
        # Two-phase generation: thinking and response are measured
        # separately so we get per-phase acceptance rates.
        # ============================================================

        # Phase 1: generate thinking only (stop at </think>)
        thinking_params = SamplingParams(
            max_tokens=args.max_tokens,
            stop=["</think>"],
            include_stop_str_in_output=True,
            **common_sampling_kwargs,
        )

        print("Phase 1: Generating thinking ...")
        # Reset metrics baseline
        metrics_before_think = collect_spec_decode_metrics(
            llm, args.num_spec_tokens)
        t1 = time.perf_counter()
        thinking_outputs = llm.generate(prompts, thinking_params)
        t2 = time.perf_counter()
        metrics_after_think = collect_spec_decode_metrics(
            llm, args.num_spec_tokens)
        thinking_elapsed = t2 - t1

        # Compute thinking-phase metric deltas
        think_deltas = tuple(
            a - b for a, b in zip(metrics_after_think[:3],
                                  metrics_before_think[:3])
        )
        think_pos_deltas = [
            a - b for a, b in zip(metrics_after_think[3],
                                  metrics_before_think[3])
        ]
        thinking_output_tokens = sum(
            len(o.outputs[0].token_ids) for o in thinking_outputs)

        # Phase 2: continue generating response
        # Build continued prompts: original prompt + thinking output
        continued_prompts = []
        for i, out in enumerate(thinking_outputs):
            thinking_text = out.outputs[0].text
            # If stop string wasn't in output, append it
            if not thinking_text.rstrip().endswith("</think>"):
                thinking_text += "</think>"
            continued_prompts.append(prompts[i] + thinking_text)

        response_params = SamplingParams(
            max_tokens=args.max_tokens,
            **common_sampling_kwargs,
        )

        print("Phase 2: Generating response ...")
        metrics_before_resp = collect_spec_decode_metrics(
            llm, args.num_spec_tokens)
        t3 = time.perf_counter()
        response_outputs = llm.generate(continued_prompts, response_params)
        t4 = time.perf_counter()
        metrics_after_resp = collect_spec_decode_metrics(
            llm, args.num_spec_tokens)
        response_elapsed = t4 - t3

        # Compute response-phase metric deltas
        resp_deltas = tuple(
            a - b for a, b in zip(metrics_after_resp[:3],
                                  metrics_before_resp[:3])
        )
        resp_pos_deltas = [
            a - b for a, b in zip(metrics_after_resp[3],
                                  metrics_before_resp[3])
        ]
        response_output_tokens = sum(
            len(o.outputs[0].token_ids) for o in response_outputs)

        total_elapsed = thinking_elapsed + response_elapsed
        total_output_tokens = thinking_output_tokens + response_output_tokens

        # Merge outputs for printing / saving
        all_thinking_texts = []
        all_response_texts = []
        for i in range(len(prompts)):
            t_text = thinking_outputs[i].outputs[0].text
            # Strip </think> tag for clean thinking text
            thinking_clean, _ = split_thinking(t_text)
            if thinking_clean is None:
                thinking_clean = t_text.replace("</think>", "").strip()
            all_thinking_texts.append(thinking_clean)
            all_response_texts.append(response_outputs[i].outputs[0].text)

        if args.print_output:
            for i in range(len(prompts)):
                print("=" * 60)
                print(f"[Prompt {i}] {texts[i][:200]}...")
                print(f"[Thinking] {all_thinking_texts[i]}")
                print(f"[Response] {all_response_texts[i]}")

        if args.save_output:
            output_dir = os.path.dirname(args.save_output)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            records = []
            for i in range(len(prompts)):
                rec = {
                    "prompt": texts[i],
                    "thinking": all_thinking_texts[i],
                    "response": all_response_texts[i],
                    "thinking_tokens": len(
                        thinking_outputs[i].outputs[0].token_ids),
                    "response_tokens": len(
                        response_outputs[i].outputs[0].token_ids),
                }
                if correct_letters is not None:
                    rec["correct_answer"] = correct_letters[i]
                records.append(rec)
            with open(args.save_output, "w") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
            print(f"Saved {len(records)} outputs to {args.save_output}")

        # Overall metrics = sum of both phases
        total_nd = think_deltas[0] + resp_deltas[0]
        total_ndt = think_deltas[1] + resp_deltas[1]
        total_nat = think_deltas[2] + resp_deltas[2]
        total_pos = [a + b for a, b in zip(think_pos_deltas, resp_pos_deltas)]

        total_metrics = (total_nd, total_ndt, total_nat,
                         total_pos, total_output_tokens, total_elapsed)
        thinking_metrics = (*think_deltas, think_pos_deltas,
                            thinking_output_tokens, thinking_elapsed)
        response_metrics = (*resp_deltas, resp_pos_deltas,
                            response_output_tokens, response_elapsed)

        print_report(args, len(prompts), total_output_tokens, total_elapsed,
                     total_metrics, thinking_metrics, response_metrics)

    else:
        # ============================================================
        # Single-phase generation (no thinking)
        # ============================================================
        sampling_params = SamplingParams(
            max_tokens=args.max_tokens,
            **common_sampling_kwargs,
        )

        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params=sampling_params)
        elapsed = time.perf_counter() - start

        total_output_tokens = sum(
            len(o.outputs[0].token_ids) for o in outputs)

        if args.print_output:
            for i, output in enumerate(outputs):
                print("=" * 60)
                print(f"[Prompt {i}] {texts[i][:200]}...")
                print(f"[Output] {output.outputs[0].text}")

        if args.save_output:
            output_dir = os.path.dirname(args.save_output)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            records = []
            for i, output in enumerate(outputs):
                rec = {
                    "prompt": texts[i],
                    "output": output.outputs[0].text,
                    "num_tokens": len(output.outputs[0].token_ids),
                }
                if correct_letters is not None:
                    rec["correct_answer"] = correct_letters[i]
                records.append(rec)
            with open(args.save_output, "w") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
            print(f"Saved {len(records)} outputs to {args.save_output}")

        nd, ndt, nat, ac = collect_spec_decode_metrics(
            llm, args.num_spec_tokens)

        total_metrics = (nd, ndt, nat, ac, total_output_tokens, elapsed)
        print_report(args, len(prompts), total_output_tokens, elapsed,
                     total_metrics)


if __name__ == "__main__":
    main()
