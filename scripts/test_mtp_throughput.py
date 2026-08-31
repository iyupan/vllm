# SPDX-License-Identifier: Apache-2.0
"""Measure MTP throughput with thinking in one generation call."""

import argparse
import json
import os
import time

from transformers import AutoTokenizer

from test_mtp_acceptance_rate_pz import (
    apply_chat_template,
    collect_spec_decode_metrics,
    load_gpqa_as_mcq,
    load_hf_dataset,
    load_local_dataset,
    split_thinking,
)
from vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser(
        description="Single-call MTP throughput benchmark"
    )

    dataset = parser.add_argument_group("dataset")
    dataset.add_argument("--dataset", default="Idavidrein/gpqa")
    dataset.add_argument("--subset", default=None)
    dataset.add_argument("--split", default="train")
    dataset.add_argument("--text-column", default="Question")
    dataset.add_argument("--dataset-path", default=None)
    dataset.add_argument("--num-prompts", type=int, default=None)
    dataset.add_argument("--format", choices=["raw", "mcq"], default="raw")
    dataset.add_argument("--mcq-seed", type=int, default=42)

    model = parser.add_argument_group("model")
    model.add_argument("--model-dir", required=True)
    model.add_argument("--num-spec-tokens", type=int, default=3)
    model.add_argument(
        "--rejection-sample-method", choices=["standard"], default="standard"
    )
    model.add_argument(
        "--draft-sample-method",
        choices=["greedy", "probabilistic"],
        default="greedy",
    )
    model.add_argument("--tp", type=int, default=1)
    model.add_argument("--max-model-len", type=int, default=65536)
    model.add_argument("--max-num-seqs", type=int, default=8)
    model.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    model.add_argument("--enforce-eager", action="store_true")

    generation = parser.add_argument_group("generation")
    generation.add_argument("--max-tokens", type=int, default=32768)
    generation.add_argument("--temp", type=float, default=1.0)
    generation.add_argument("--top-p", type=float, default=0.95)
    generation.add_argument("--top-k", type=int, default=20)
    generation.add_argument("--min-p", type=float, default=0.0)
    generation.add_argument("--presence-penalty", type=float, default=1.5)
    generation.add_argument("--repetition-penalty", type=float, default=1.0)
    generation.add_argument("--seed", type=int, default=42)

    thinking = parser.add_argument_group("thinking")
    thinking.add_argument("--enable-thinking", action="store_true")
    thinking.add_argument("--reasoning-parser", default=None)

    output = parser.add_argument_group("output")
    output.add_argument("--print-output", action="store_true")
    output.add_argument("--save-output", default=None)
    return parser.parse_args()


def load_prompts(args, tokenizer):
    correct_letters = None
    if args.format == "mcq":
        texts, correct_letters = load_gpqa_as_mcq(
            args.dataset, args.subset, args.split, seed=args.mcq_seed
        )
    elif args.dataset == "local":
        if not args.dataset_path:
            raise ValueError("--dataset-path is required with --dataset=local")
        texts = load_local_dataset(args.dataset_path, args.text_column)
    else:
        texts = load_hf_dataset(
            args.dataset, args.subset, args.split, args.text_column
        )

    if args.num_prompts is not None:
        texts = texts[: args.num_prompts]
        if correct_letters is not None:
            correct_letters = correct_letters[: args.num_prompts]

    prompts = apply_chat_template(
        tokenizer, texts, enable_thinking=args.enable_thinking
    )
    return texts, prompts, correct_letters


def subtract_metrics(after, before):
    counters = tuple(after[i] - before[i] for i in range(3))
    positions = [a - b for a, b in zip(after[3], before[3])]
    return *counters, positions


def find_subsequence(values, subsequence):
    width = len(subsequence)
    for start in range(len(values) - width + 1):
        if values[start : start + width] == subsequence:
            return start
    return None


def count_output_phases(outputs, tokenizer, thinking_enabled):
    total_tokens = 0
    thinking_tokens = 0
    response_tokens = 0
    delimiter_count = 0
    length_capped = 0
    delimiter = tokenizer.encode("</think>", add_special_tokens=False)

    for request_output in outputs:
        output = request_output.outputs[0]
        token_ids = list(output.token_ids)
        total_tokens += len(token_ids)
        length_capped += output.finish_reason == "length"

        if not thinking_enabled:
            response_tokens += len(token_ids)
            continue

        delimiter_start = find_subsequence(token_ids, delimiter)
        if delimiter_start is None:
            thinking_tokens += len(token_ids)
            continue

        delimiter_count += 1
        thinking_end = delimiter_start + len(delimiter)
        thinking_tokens += thinking_end
        response_tokens += len(token_ids) - thinking_end

    return (
        total_tokens,
        thinking_tokens,
        response_tokens,
        delimiter_count,
        length_capped,
    )


def print_report(args, num_prompts, token_counts, elapsed, metrics):
    total_tokens, thinking_tokens, response_tokens = token_counts[:3]
    delimiter_count, length_capped = token_counts[3:]
    num_drafts, num_draft_tokens, num_accepted_tokens, positions = metrics

    print("\n" + "=" * 60)
    print("MTP Speculative Decoding - Single-call Throughput Report")
    print("=" * 60)
    print(f"Model:                  {args.model_dir}")
    print(f"Dataset:                {args.dataset}")
    print(f"Num prompts:            {num_prompts}")
    print(f"Num spec tokens:        {args.num_spec_tokens}")
    print(f"Rejection sampling:     {args.rejection_sample_method}")
    print(f"Draft sampling:         {args.draft_sample_method}")
    print(f"Thinking enabled:       {args.enable_thinking}")
    print("Generation calls:       1")
    print(f"Max output tokens:      {args.max_tokens}")
    print("Ignore EOS:             False")
    print(f"Total output tokens:    {total_tokens}")
    print(f"Thinking tokens:        {thinking_tokens}")
    print(f"Response tokens:        {response_tokens}")
    print(f"Thinking delimiters:    {delimiter_count}/{num_prompts}")
    print(f"Length-capped requests: {length_capped}/{num_prompts}")
    print("Phase timing:           N/A (single generation call)")
    print(f"Total inference time:   {elapsed:.2f}s")
    print(f"Total throughput:       {total_tokens / elapsed:.1f} tok/s")

    print("\n--- Overall ---")
    print(f"  Output tokens:         {total_tokens}")
    print(f"  Inference time:        {elapsed:.2f}s")
    print(f"  Throughput:            {total_tokens / elapsed:.1f} tok/s")
    if total_tokens:
        print(
            "  Amortized output time: "
            f"{elapsed * 1000 / total_tokens:.3f} ms/tok"
        )
    print(f"  Num drafts:            {num_drafts}")
    print(f"  Num draft tokens:      {num_draft_tokens}")
    print(f"  Num accepted tokens:   {num_accepted_tokens}")
    print(f"  Draft request rate:    {num_drafts / elapsed:.1f} drafts/s")
    print(f"  Draft token rate:      {num_draft_tokens / elapsed:.1f} tok/s")
    print(f"  Accepted token rate:   {num_accepted_tokens / elapsed:.1f} tok/s")
    if num_drafts and num_draft_tokens:
        acceptance = num_accepted_tokens / num_draft_tokens * 100
        mean_length = 1 + num_accepted_tokens / num_drafts
        print(
            "  Amortized draft time:  "
            f"{elapsed * 1000 / num_drafts:.3f} ms/draft"
        )
        print(f"  Acceptance rate:       {acceptance:.2f}%")
        print(f"  Mean accept length:    {mean_length:.4f}")
        print("  Per-position acceptance rate:")
        for index, count in enumerate(positions):
            print(
                f"    Position {index}: {count / num_drafts:.4f} "
                f"({count}/{num_drafts})"
            )
    print("=" * 60)


def save_outputs(path, texts, outputs, correct_letters):
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    records = []
    for index, request_output in enumerate(outputs):
        output = request_output.outputs[0]
        thinking, response = split_thinking(output.text)
        record = {
            "prompt": texts[index],
            "output": output.text,
            "thinking": thinking,
            "response": response,
            "num_tokens": len(output.token_ids),
            "finish_reason": output.finish_reason,
        }
        if correct_letters is not None:
            record["correct_answer"] = correct_letters[index]
        records.append(record)

    with open(path, "w") as output_file:
        json.dump(records, output_file, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, trust_remote_code=True
    )
    texts, prompts, correct_letters = load_prompts(args, tokenizer)
    print(
        f"Prepared {len(prompts)} prompts "
        f"(thinking={args.enable_thinking}, single_call=True)"
    )

    speculative_config = {
        "method": "mtp",
        "num_speculative_tokens": args.num_spec_tokens,
        "rejection_sample_method": args.rejection_sample_method,
        "draft_sample_method": args.draft_sample_method,
    }
    llm_kwargs = {}
    if args.enable_thinking and args.reasoning_parser:
        llm_kwargs["reasoning_parser"] = args.reasoning_parser

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
        **llm_kwargs,
    )
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temp,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        presence_penalty=args.presence_penalty,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
    )

    metrics_before = collect_spec_decode_metrics(llm, args.num_spec_tokens)
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    elapsed = time.perf_counter() - start
    metrics_after = collect_spec_decode_metrics(llm, args.num_spec_tokens)

    token_counts = count_output_phases(
        outputs, tokenizer, args.enable_thinking
    )
    metrics = subtract_metrics(metrics_after, metrics_before)
    print_report(args, len(prompts), token_counts, elapsed, metrics)

    if args.print_output:
        for index, request_output in enumerate(outputs):
            print("=" * 60)
            print(f"[Prompt {index}] {texts[index][:200]}...")
            print(request_output.outputs[0].text)
    if args.save_output:
        save_outputs(args.save_output, texts, outputs, correct_letters)
        print(f"Saved {len(outputs)} outputs to {args.save_output}")


if __name__ == "__main__":
    main()
