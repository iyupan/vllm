# SPDX-License-Identifier: Apache-2.0
"""
Measure MTP speculative decoding acceptance rate on SPEED-Bench.

SPEED-Bench is a multi-turn benchmark whose turn texts are placeholders
that must be resolved against external HuggingFace datasets first.  This
script consumes the *resolved* parquet produced by
``SPEEDBench.prepare_data`` (see scripts/speed.py).

Examples:

    # Single-turn (use only turns[0]) on qualitative split
    python scripts/test_mtp_acceptance_rate_speed.py \\
        --model-dir /path/to/model \\
        --parquet-path /data/speed_bench/qualitative/test.parquet \\
        --num-spec-tokens 3 \\
        --enable-thinking --reasoning-parser qwen3

    # Full multi-turn evaluation on throughput_8k
    python scripts/test_mtp_acceptance_rate_speed.py \\
        --model-dir /path/to/model \\
        --parquet-path /data/speed_bench/throughput_8k \\
        --multi-turn \\
        --num-spec-tokens 3

Output layout (--save-output is a directory):
    <save_output>/
        <category_a>.json         # per-category records + summary
        <category_b>.json
        ...
        _summary.json             # aggregate across all categories
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

# Allow importing helpers from sibling test_mtp_acceptance_rate_pz.py.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from test_mtp_acceptance_rate_pz import (  # noqa: E402
    collect_spec_decode_metrics,
    print_phase_report,
    split_thinking,
)
from transformers import AutoTokenizer  # noqa: E402

from vllm import LLM, SamplingParams  # noqa: E402

TURNS_PLACEHOLDER = (
    "FULL BENCHMARK DATA SHOULD BE FETCHED FROM THE SOURCE USING SPECDEC_BENCH"
)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_speed_bench(parquet_path: str,
                     num_samples: int | None = None) -> list[dict]:
    """Load resolved SPEED-Bench data from parquet.

    Args:
        parquet_path: Path to a parquet file or a directory containing one or
            more ``*.parquet`` files (typically ``test.parquet``).
        num_samples: If given, only the first ``num_samples`` rows are kept.

    Returns:
        list of dicts with keys ``turns`` (list[str]), ``category``,
        ``question_id`` and ``source``.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    p = Path(parquet_path)
    if p.is_dir():
        files = sorted(p.rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(
                f"No *.parquet files found under {parquet_path}")
        tables = [pq.read_table(str(f)) for f in files]
        table = tables[0] if len(tables) == 1 else pa.concat_tables(tables)
    else:
        table = pq.read_table(str(p))

    df = table.to_pandas()
    if num_samples is not None:
        df = df.iloc[:num_samples]

    requests: list[dict] = []
    for _, row in df.iterrows():
        turns = [str(t) for t in row["turns"]]
        for t in turns:
            if t.startswith(TURNS_PLACEHOLDER):
                raise ValueError(
                    f"Unresolved placeholder in question_id="
                    f"{row.get('question_id', '?')}; rerun "
                    f"`SPEEDBench.prepare_data` to fetch external data.")
        requests.append({
            "turns": turns,
            "category": str(row.get("category", "")),
            "question_id": str(row.get("question_id", "")),
            "source": str(row.get("source", "")),
        })
    return requests


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_chat_prompt(tokenizer, messages: list[dict],
                      enable_thinking: bool = False) -> str:
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=enable_thinking)


# ---------------------------------------------------------------------------
# Per-batch generation (handles optional thinking phase split)
# ---------------------------------------------------------------------------

def run_generation(llm, prompts, args, common_kwargs):
    """Run one batch of generation, optionally split into thinking + response.

    Returns dict with:
        outputs:           list of vLLM RequestOutput (response phase)
        response_texts:    list of clean response strings
        thinking_texts:    list of thinking strings (or "" if disabled)
        num_output_tokens: total tokens
        elapsed:           seconds
        metrics_total:     (num_drafts, num_draft_tokens, num_accepted, ac_list)
        metrics_thinking:  same 4-tuple + (tokens, elapsed) or None
        metrics_response:  same 4-tuple + (tokens, elapsed) or None
    """
    spec_n = args.num_spec_tokens

    if args.enable_thinking:
        # Phase 1: thinking
        m_before = collect_spec_decode_metrics(llm, spec_n)
        thinking_params = SamplingParams(
            max_tokens=args.max_tokens,
            stop=["</think>"],
            include_stop_str_in_output=True,
            **common_kwargs,
        )
        t1 = time.perf_counter()
        thinking_outputs = llm.generate(prompts, thinking_params)
        t2 = time.perf_counter()
        m_after_think = collect_spec_decode_metrics(llm, spec_n)
        elapsed_think = t2 - t1
        tok_think = sum(len(o.outputs[0].token_ids) for o in thinking_outputs)
        think_4 = (
            m_after_think[0] - m_before[0],
            m_after_think[1] - m_before[1],
            m_after_think[2] - m_before[2],
            [a - b for a, b in zip(m_after_think[3], m_before[3])],
        )

        # Phase 2: response
        continued = []
        for i, out in enumerate(thinking_outputs):
            txt = out.outputs[0].text
            if not txt.rstrip().endswith("</think>"):
                txt += "</think>"
            continued.append(prompts[i] + txt)
        response_params = SamplingParams(
            max_tokens=args.max_tokens, **common_kwargs,
        )
        t3 = time.perf_counter()
        response_outputs = llm.generate(continued, response_params)
        t4 = time.perf_counter()
        m_after_resp = collect_spec_decode_metrics(llm, spec_n)
        elapsed_resp = t4 - t3
        tok_resp = sum(len(o.outputs[0].token_ids) for o in response_outputs)
        resp_4 = (
            m_after_resp[0] - m_after_think[0],
            m_after_resp[1] - m_after_think[1],
            m_after_resp[2] - m_after_think[2],
            [a - b for a, b in zip(m_after_resp[3], m_after_think[3])],
        )

        thinking_texts: list[str] = []
        response_texts: list[str] = []
        for i in range(len(prompts)):
            t_text = thinking_outputs[i].outputs[0].text
            thinking_clean, _ = split_thinking(t_text)
            if thinking_clean is None:
                thinking_clean = t_text.replace("</think>", "").strip()
            thinking_texts.append(thinking_clean)
            response_texts.append(response_outputs[i].outputs[0].text)

        total_4 = (
            think_4[0] + resp_4[0],
            think_4[1] + resp_4[1],
            think_4[2] + resp_4[2],
            [a + b for a, b in zip(think_4[3], resp_4[3])],
        )
        return {
            "outputs": response_outputs,
            "thinking_texts": thinking_texts,
            "response_texts": response_texts,
            "num_output_tokens": tok_think + tok_resp,
            "elapsed": elapsed_think + elapsed_resp,
            "metrics_total": total_4,
            "metrics_thinking": (*think_4, tok_think, elapsed_think),
            "metrics_response": (*resp_4, tok_resp, elapsed_resp),
        }

    # Single-phase generation (no thinking)
    m_before = collect_spec_decode_metrics(llm, spec_n)
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens, **common_kwargs,
    )
    t1 = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    t2 = time.perf_counter()
    m_after = collect_spec_decode_metrics(llm, spec_n)
    elapsed = t2 - t1
    tok = sum(len(o.outputs[0].token_ids) for o in outputs)
    metrics_4 = (
        m_after[0] - m_before[0],
        m_after[1] - m_before[1],
        m_after[2] - m_before[2],
        [a - b for a, b in zip(m_after[3], m_before[3])],
    )
    return {
        "outputs": outputs,
        "thinking_texts": ["" for _ in prompts],
        "response_texts": [o.outputs[0].text for o in outputs],
        "num_output_tokens": tok,
        "elapsed": elapsed,
        "metrics_total": metrics_4,
        "metrics_thinking": None,
        "metrics_response": None,
    }


# ---------------------------------------------------------------------------
# Single-turn vs multi-turn drivers
# ---------------------------------------------------------------------------

def run_single_turn(llm, tokenizer, requests, args, common_kwargs):
    """Use only turns[0] for every request (matches the standard MTP bench)."""
    prompts = []
    for r in requests:
        if args.mode == "chat":
            prompts.append(build_chat_prompt(
                tokenizer,
                [{"role": "user", "content": r["turns"][0]}],
                enable_thinking=args.enable_thinking))
        else:
            prompts.append(r["turns"][0])

    print(f"Prepared {len(prompts)} prompts (mode={args.mode}, single-turn)")
    res = run_generation(llm, prompts, args, common_kwargs)

    records = []
    for i, r in enumerate(requests):
        rec = {
            "question_id": r["question_id"],
            "category": r["category"],
            "source": r["source"],
            "turns": [r["turns"][0]],
            "responses": [res["response_texts"][i]],
            "num_output_tokens": [
                len(res["outputs"][i].outputs[0].token_ids)],
        }
        if args.enable_thinking:
            rec["thinking"] = [res["thinking_texts"][i]]
        records.append(rec)

    return records, res


def run_multi_turn(llm, tokenizer, requests, args, common_kwargs):
    """Iterate every turn, accumulating chat history per request."""
    if args.mode != "chat":
        raise ValueError("--multi-turn requires --mode=chat")

    histories: list[list[dict]] = [[] for _ in requests]
    per_resp: list[list[str]] = [[] for _ in requests]
    per_think: list[list[str]] = [[] for _ in requests]
    per_tok: list[list[int]] = [[] for _ in requests]

    max_turns = max(len(r["turns"]) for r in requests)
    print(f"Multi-turn mode: max turns = {max_turns}, "
          f"requests = {len(requests)}")

    spec_n = args.num_spec_tokens
    tot_nd = tot_ndt = tot_nat = 0
    tot_ac = [0] * spec_n
    tot_tok = 0
    tot_el = 0.0
    th_nd = th_ndt = th_nat = th_tok = 0
    th_ac = [0] * spec_n
    th_el = 0.0
    rs_nd = rs_ndt = rs_nat = rs_tok = 0
    rs_ac = [0] * spec_n
    rs_el = 0.0

    for t in range(max_turns):
        active = [i for i, r in enumerate(requests) if t < len(r["turns"])]
        if not active:
            break

        prompts = []
        for i in active:
            histories[i].append(
                {"role": "user", "content": requests[i]["turns"][t]})
            prompts.append(build_chat_prompt(
                tokenizer, histories[i],
                enable_thinking=args.enable_thinking))

        print(f"\n[Turn {t + 1}/{max_turns}] Active requests: {len(active)}")
        res = run_generation(llm, prompts, args, common_kwargs)

        for k, i in enumerate(active):
            response = res["response_texts"][k]
            thinking = res["thinking_texts"][k]
            tok = len(res["outputs"][k].outputs[0].token_ids)
            per_resp[i].append(response)
            per_think[i].append(thinking)
            per_tok[i].append(tok)
            histories[i].append({"role": "assistant", "content": response})

        nd, ndt, nat, ac = res["metrics_total"]
        tot_nd += nd
        tot_ndt += ndt
        tot_nat += nat
        for j in range(spec_n):
            tot_ac[j] += ac[j]
        tot_tok += res["num_output_tokens"]
        tot_el += res["elapsed"]

        if res["metrics_thinking"] is not None:
            mt_nd, mt_ndt, mt_nat, mt_ac, mt_tok, mt_el = res["metrics_thinking"]
            th_nd += mt_nd
            th_ndt += mt_ndt
            th_nat += mt_nat
            for j in range(spec_n):
                th_ac[j] += mt_ac[j]
            th_tok += mt_tok
            th_el += mt_el
        if res["metrics_response"] is not None:
            mr_nd, mr_ndt, mr_nat, mr_ac, mr_tok, mr_el = res["metrics_response"]
            rs_nd += mr_nd
            rs_ndt += mr_ndt
            rs_nat += mr_nat
            for j in range(spec_n):
                rs_ac[j] += mr_ac[j]
            rs_tok += mr_tok
            rs_el += mr_el

    records = []
    for i, r in enumerate(requests):
        rec = {
            "question_id": r["question_id"],
            "category": r["category"],
            "source": r["source"],
            "turns": r["turns"],
            "responses": per_resp[i],
            "num_output_tokens": per_tok[i],
        }
        if args.enable_thinking:
            rec["thinking"] = per_think[i]
        records.append(rec)

    aggregated = {
        "metrics_total": (tot_nd, tot_ndt, tot_nat, tot_ac),
        "num_output_tokens": tot_tok,
        "elapsed": tot_el,
        "metrics_thinking": (
            (th_nd, th_ndt, th_nat, th_ac, th_tok, th_el)
            if args.enable_thinking else None),
        "metrics_response": (
            (rs_nd, rs_ndt, rs_nat, rs_ac, rs_tok, rs_el)
            if args.enable_thinking else None),
    }
    return records, aggregated


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_speed_report(args, num_requests, num_total_turns, result):
    print("\n" + "=" * 60)
    print("MTP Speculative Decoding - SPEED-Bench Acceptance Rate")
    print("=" * 60)
    print(f"Model:                {args.model_dir}")
    print(f"Parquet:              {args.parquet_path}")
    print(f"Num requests:         {num_requests}")
    print(f"Num total turns:      {num_total_turns}")
    print(f"Mode:                 {args.mode}")
    print(f"Multi-turn:           {args.multi_turn}")
    print(f"Num spec tokens:      {args.num_spec_tokens}")
    print(f"Thinking enabled:     {args.enable_thinking}")
    print(f"Total output tokens:  {result['num_output_tokens']}")
    print(f"Total inference time: {result['elapsed']:.2f}s")
    if result["elapsed"] > 0:
        print(f"Total throughput:     "
              f"{result['num_output_tokens'] / result['elapsed']:.1f} tok/s")

    nd, ndt, nat, ac = result["metrics_total"]
    print_phase_report("Overall", nd, ndt, nat, ac,
                       result["num_output_tokens"], result["elapsed"])

    if result.get("metrics_thinking") is not None:
        nd_t, ndt_t, nat_t, ac_t, tok_t, el_t = result["metrics_thinking"]
        print_phase_report(
            "Thinking Phase", nd_t, ndt_t, nat_t, ac_t, tok_t, el_t)
    if result.get("metrics_response") is not None:
        nd_r, ndt_r, nat_r, ac_r, tok_r, el_r = result["metrics_response"]
        print_phase_report(
            "Response Phase", nd_r, ndt_r, nat_r, ac_r, tok_r, el_r)
    print("=" * 60)


def print_per_category_summary(per_cat_summaries: dict):
    """Per-category breakdown including MTP acceptance rate.

    Each entry in ``per_cat_summaries`` is the summary dict produced by
    ``build_summary``.
    """
    print("\n--- Per-category breakdown ---")
    print(f"{'category':<35} {'reqs':>5} {'turns':>5} {'out tok':>10} "
          f"{'accept':>8} {'len':>6}")
    for c in sorted(per_cat_summaries.keys()):
        s = per_cat_summaries[c]
        ar = s.get("acceptance_rate")
        ml = s.get("mean_accept_length")
        ar_str = f"{ar * 100:6.2f}%" if ar is not None else "  N/A "
        ml_str = f"{ml:6.3f}" if ml is not None else " N/A  "
        print(f"{c:<35} {s['num_requests']:>5} {s['num_total_turns']:>5} "
              f"{s['num_output_tokens']:>10} {ar_str:>8} {ml_str:>6}")


# ---------------------------------------------------------------------------
# Filename / summary helpers
# ---------------------------------------------------------------------------

_FILENAME_RE = re.compile(r"[^\w\-.]+")


def sanitize_filename(name: str) -> str:
    """Convert an arbitrary category string to a safe filename component."""
    if not name:
        return "_uncategorized"
    safe = _FILENAME_RE.sub("_", name).strip("_.")
    return safe[:80] if safe else "_uncategorized"


def build_summary(args, records, result, category=None):
    """Build the JSON-serialisable summary dict for one category or the
    overall run."""
    nd, ndt, nat, ac = result["metrics_total"]
    summary = {
        "model": args.model_dir,
        "parquet_path": args.parquet_path,
        "category": category,
        "num_requests": len(records),
        "num_total_turns": sum(len(r["responses"]) for r in records),
        "num_spec_tokens": args.num_spec_tokens,
        "multi_turn": args.multi_turn,
        "enable_thinking": args.enable_thinking,
        "num_drafts": int(nd),
        "num_draft_tokens": int(ndt),
        "num_accepted_tokens": int(nat),
        "acceptance_counts_per_pos": [int(x) for x in ac],
        "acceptance_rate": (nat / ndt) if ndt > 0 else None,
        "mean_accept_length": (1 + nat / nd) if nd > 0 else None,
        "num_output_tokens": int(result["num_output_tokens"]),
        "elapsed_seconds": float(result["elapsed"]),
    }
    if result.get("metrics_thinking") is not None:
        nd_t, ndt_t, nat_t, ac_t, tok_t, el_t = result["metrics_thinking"]
        summary["thinking"] = {
            "num_drafts": int(nd_t),
            "num_draft_tokens": int(ndt_t),
            "num_accepted_tokens": int(nat_t),
            "acceptance_counts_per_pos": [int(x) for x in ac_t],
            "acceptance_rate": (nat_t / ndt_t) if ndt_t > 0 else None,
            "num_output_tokens": int(tok_t),
            "elapsed_seconds": float(el_t),
        }
    if result.get("metrics_response") is not None:
        nd_r, ndt_r, nat_r, ac_r, tok_r, el_r = result["metrics_response"]
        summary["response"] = {
            "num_drafts": int(nd_r),
            "num_draft_tokens": int(ndt_r),
            "num_accepted_tokens": int(nat_r),
            "acceptance_counts_per_pos": [int(x) for x in ac_r],
            "acceptance_rate": (nat_r / ndt_r) if ndt_r > 0 else None,
            "num_output_tokens": int(tok_r),
            "elapsed_seconds": float(el_r),
        }
    return summary


def merge_results(results: list[dict], spec_n: int) -> dict:
    """Aggregate per-category run_generation results into one overall result."""
    tot_nd = tot_ndt = tot_nat = 0
    tot_ac = [0] * spec_n
    tot_tok = 0
    tot_el = 0.0
    th_nd = th_ndt = th_nat = th_tok = 0
    th_ac = [0] * spec_n
    th_el = 0.0
    rs_nd = rs_ndt = rs_nat = rs_tok = 0
    rs_ac = [0] * spec_n
    rs_el = 0.0
    has_think = False

    for r in results:
        nd, ndt, nat, ac = r["metrics_total"]
        tot_nd += nd
        tot_ndt += ndt
        tot_nat += nat
        for j in range(spec_n):
            tot_ac[j] += ac[j]
        tot_tok += r["num_output_tokens"]
        tot_el += r["elapsed"]
        if r.get("metrics_thinking") is not None:
            has_think = True
            mt = r["metrics_thinking"]
            th_nd += mt[0]; th_ndt += mt[1]; th_nat += mt[2]
            for j in range(spec_n):
                th_ac[j] += mt[3][j]
            th_tok += mt[4]; th_el += mt[5]
        if r.get("metrics_response") is not None:
            mr = r["metrics_response"]
            rs_nd += mr[0]; rs_ndt += mr[1]; rs_nat += mr[2]
            for j in range(spec_n):
                rs_ac[j] += mr[3][j]
            rs_tok += mr[4]; rs_el += mr[5]

    return {
        "metrics_total": (tot_nd, tot_ndt, tot_nat, tot_ac),
        "num_output_tokens": tot_tok,
        "elapsed": tot_el,
        "metrics_thinking": (
            (th_nd, th_ndt, th_nat, th_ac, th_tok, th_el)
            if has_think else None),
        "metrics_response": (
            (rs_nd, rs_ndt, rs_nat, rs_ac, rs_tok, rs_el)
            if has_think else None),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="MTP acceptance rate on SPEED-Bench (resolved parquet)",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    ds = parser.add_argument_group("dataset")
    ds.add_argument("--parquet-path", type=str, required=True,
                    help="Path to a *.parquet file or directory produced by "
                         "SPEEDBench.prepare_data")
    ds.add_argument("--num-prompts", type=int, default=None,
                    help="Limit number of requests (default: all)")
    ds.add_argument("--multi-turn", action="store_true",
                    help="Iterate every conversation turn (default: only "
                         "turns[0])")

    md = parser.add_argument_group("model")
    md.add_argument("--model-dir", type=str, required=True)
    md.add_argument("--num-spec-tokens", type=int, default=3)
    md.add_argument("--tp", type=int, default=1)
    md.add_argument("--max-model-len", type=int, default=16384)
    md.add_argument("--enforce-eager", action="store_true")
    md.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    md.add_argument("--max-num-seqs", type=int, default=None)

    gen = parser.add_argument_group("generation")
    gen.add_argument("--max-tokens", type=int, default=1024)
    gen.add_argument("--temp", type=float, default=1.0)
    gen.add_argument("--top-p", type=float, default=0.95)
    gen.add_argument("--top-k", type=int, default=20)
    gen.add_argument("--min-p", type=float, default=0.0)
    gen.add_argument("--presence-penalty", type=float, default=1.5)
    gen.add_argument("--repetition-penalty", type=float, default=1.0)

    think = parser.add_argument_group("thinking")
    think.add_argument("--enable-thinking", action="store_true")
    think.add_argument("--reasoning-parser", type=str, default=None)

    parser.add_argument("--mode", type=str, default="chat",
                        choices=["chat", "completion"])
    parser.add_argument("--print-output", action="store_true")
    parser.add_argument("--save-output", type=str, default=None,
                        help="Output directory. Each category is written to "
                             "<category>.json; aggregated metrics are written "
                             "to _summary.json.")

    return parser.parse_args()


def main():
    args = parse_args()
    if args.enable_thinking and args.mode != "chat":
        raise ValueError("--enable-thinking requires --mode=chat")
    if args.multi_turn and args.mode != "chat":
        raise ValueError("--multi-turn requires --mode=chat")

    print(f"Loading SPEED-Bench parquet from {args.parquet_path} ...")
    requests = load_speed_bench(args.parquet_path, args.num_prompts)
    print(f"Loaded {len(requests)} requests")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir,
                                              trust_remote_code=True)

    speculative_config = {
        "method": "mtp",
        "num_speculative_tokens": args.num_spec_tokens,
    }
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

    common_kwargs = dict(
        temperature=args.temp,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        presence_penalty=args.presence_penalty,
        repetition_penalty=args.repetition_penalty,
    )

    # Group requests by category so each batch yields per-category MTP metrics.
    by_category: dict[str, list[dict]] = defaultdict(list)
    for r in requests:
        by_category[r["category"] or "_uncategorized"].append(r)
    print(f"Discovered {len(by_category)} categories: "
          f"{sorted(by_category.keys())}")

    save_dir: Path | None = None
    if args.save_output:
        save_dir = Path(args.save_output)
        save_dir.mkdir(parents=True, exist_ok=True)

    per_cat_summaries: dict[str, dict] = {}
    per_cat_results: list[dict] = []
    all_records: list[dict] = []
    used_filenames: set[str] = set()

    for category in sorted(by_category.keys()):
        cat_requests = by_category[category]
        print("\n" + "#" * 60)
        print(f"# Category: {category} ({len(cat_requests)} requests)")
        print("#" * 60)

        if args.multi_turn:
            records, result = run_multi_turn(
                llm, tokenizer, cat_requests, args, common_kwargs)
        else:
            records, raw = run_single_turn(
                llm, tokenizer, cat_requests, args, common_kwargs)
            result = {
                "metrics_total": raw["metrics_total"],
                "num_output_tokens": raw["num_output_tokens"],
                "elapsed": raw["elapsed"],
                "metrics_thinking": raw.get("metrics_thinking"),
                "metrics_response": raw.get("metrics_response"),
            }

        per_cat_results.append(result)
        all_records.extend(records)
        cat_summary = build_summary(args, records, result, category=category)
        per_cat_summaries[category] = cat_summary

        if args.print_output:
            for rec in records[:3]:
                print("=" * 60)
                print(f"[{rec['question_id']}] cat={rec['category']}")
                for ti, (q, a) in enumerate(
                        zip(rec["turns"], rec["responses"])):
                    print(f"  Turn {ti}: Q: {q[:120]}...")
                    print(f"           A: {a[:200]}...")

        if save_dir is not None:
            base = sanitize_filename(category)
            fname = base + ".json"
            i = 1
            while fname in used_filenames:
                i += 1
                fname = f"{base}_{i}.json"
            used_filenames.add(fname)
            cat_path = save_dir / fname
            with open(cat_path, "w") as f:
                json.dump({"summary": cat_summary, "records": records}, f,
                          ensure_ascii=False, indent=2)
            print(f"  -> wrote {len(records)} records to {cat_path}")

        # Per-category report (compact)
        print_speed_report(
            args, len(records),
            sum(len(r["responses"]) for r in records),
            result)

    # Aggregate across categories
    overall = merge_results(per_cat_results, args.num_spec_tokens)
    overall_summary = build_summary(
        args, all_records, overall, category="__overall__")

    print("\n" + "@" * 60)
    print("@@ OVERALL (all categories combined)")
    print("@" * 60)
    print_speed_report(
        args, len(all_records),
        sum(len(r["responses"]) for r in all_records),
        overall)
    print_per_category_summary(per_cat_summaries)

    if save_dir is not None:
        with open(save_dir / "_summary.json", "w") as f:
            json.dump({
                "overall": overall_summary,
                "per_category": per_cat_summaries,
            }, f, ensure_ascii=False, indent=2)
        print(f"\nSaved aggregate summary to {save_dir / '_summary.json'}")
        print(f"All outputs under {save_dir}")


if __name__ == "__main__":
    main()