# MTP Top-k Acceptance Rate — Development Notes

## Feature Goal

For each MTP speculative head position, report top-1/top-2/top-3 acceptance rates.
Top-2/top-3 are statistics only — the actual spec decode flow is unchanged (uses top-1).

## Architecture Decisions & Lessons Learned

### V1 vs V2 Model Runner

vLLM has TWO model runners:
- **V1 model runner** (`vllm/v1/worker/gpu_model_runner.py`): **DEFAULT**
- **V2 model runner** (`vllm/v1/worker/gpu/model_runner.py`): opt-in via `VLLM_USE_V2_MODEL_RUNNER=1`

Worker selection in `vllm/v1/worker/gpu_worker.py:296`:
```python
if self.use_v2_model_runner:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner as GPUModelRunnerV2
else:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner as GPUModelRunnerV1
```

### ModelRunnerOutput Cross-Process Transport Issue

New `@dataclass` fields added to `ModelRunnerOutput` (`vllm/v1/outputs.py`) are **silently dropped** during cross-process pickle transport (worker → engine core via shared memory message queue at `vllm/distributed/device_communicators/shm_broadcast.py:719`).

**Solution:** Bypass `ModelRunnerOutput`. Worker writes stats to `/tmp/vllm_topk_stats.json`, test script reads after generation. Path configurable via `VLLM_TOPK_STATS_PATH` env var.

### V1 Proposer: Greedy Sampling

V1 proposer (`SpecDecodeBaseProposer` in `vllm/v1/spec_decode/eagle.py`) uses **argmax** (greedy).
V2 speculator (`vllm/v1/worker/gpu/spec_decode/eagle/speculator.py`) uses **gumbel_sample**.

### Accurate Top-k: Use Target Logits Directly

Do NOT use `sampler_output.sampled_token_ids` — it has `-1` placeholders after rejection and recovered tokens at the rejection point.

Use target model's logits directly at each draft position:
```python
target_logits = logits[metadata.target_logits_indices]
target_tokens = target_logits.argmax(dim=-1)
```

This gives every draft position a definitive target token, making top-k rates accurate (not just lower bounds).

### TP Race Condition Fix

Multiple TP workers produce identical results but write to the same file concurrently. Fixed by checking `torch.distributed.get_rank() == 0` in `accumulate_topk_stats()` — only rank 0 writes.

## Files Modified

### Core (V1 pipeline — the working path)
| File | Change |
|------|--------|
| `vllm/v1/spec_decode/eagle.py` | `_greedy_sample_with_topk()`, `self.draft_topk` in `propose()` |
| `vllm/v1/worker/gpu_model_runner.py` | `_accumulate_topk_acceptance()` after rejection sampling in `_sample()` |
| `vllm/v1/worker/gpu/spec_decode/topk_stats.py` | **NEW** — file-based stats accumulator, rank-0 only write, auto-flush |

### Metrics Extension (Prometheus counters — registered but not populated via file-based path)
| File | Change |
|------|--------|
| `vllm/v1/spec_decode/metrics.py` | `SpecDecodingStats` / `SpecDecodingProm` / `SpecDecodingLogging` top-k fields |
| `vllm/v1/metrics/reader.py` | Handle `vllm:spec_decode_num_topk_*` metric names |

### V2 Model Runner (cleaned — no topk code, not used by default)
| File | Status |
|------|--------|
| `vllm/v1/worker/gpu/spec_decode/eagle/speculator.py` | Clean — topk code removed |
| `vllm/v1/worker/gpu/spec_decode/rejection_sampler.py` | Clean — topk code removed |
| `vllm/v1/worker/gpu/model_runner.py` | Clean — topk code removed |
| `vllm/v1/outputs.py` | Clean — no topk fields |
| `vllm/v1/core/sched/scheduler.py` | Clean — no topk code |

### Scripts
| File | Change |
|------|--------|
| `scripts/test_mtp_acceptance_rate_topk.py` | **NEW** — test script reading file-based stats |
| `scripts/run_mtp_benchmark_topk.sh` | **NEW** — launcher, cleans `/tmp/vllm_topk_stats.json` before run |
| `.claude/commands/run-benchmark-topk.md` | **NEW** — Claude Code `/run-benchmark-topk` skill |

## Data Flow

```
EagleProposer.propose()  (vllm/v1/spec_decode/eagle.py)
  └─ _greedy_sample_with_topk() → self.draft_topk [batch, num_spec, 3]

gpu_model_runner._sample()  (vllm/v1/worker/gpu_model_runner.py)
  ├─ rejection_sampler() → sampler_output (unchanged)
  └─ _accumulate_topk_acceptance(metadata, logits, draft_topk)
       ├─ target_tokens = logits[target_logits_indices].argmax(-1)  ← accurate at ALL positions
       ├─ Compare target_tokens vs draft_topk per position per k
       └─ accumulate_topk_stats()  (topk_stats.py, rank 0 only)
            └─ writes /tmp/vllm_topk_stats.json (auto-flush every batch)

test_mtp_acceptance_rate_topk.py
  └─ load_topk_from_file() → print_topk_report()

run_mtp_benchmark_topk.sh
  └─ rm -f /tmp/vllm_topk_stats.json → exec test script
```

## Key Design Points

1. **No impact on spec decode**: topk tracking is purely statistical, never modifies draft/target tokens or acceptance decisions
2. **File-based bypass**: `ModelRunnerOutput` new fields are lost in cross-process pickle, so stats go directly to disk
3. **Rank-0 only write**: avoids TP worker file race condition
4. **Target logits argmax**: not rejection sampler output, ensures every position has a valid target token
5. **Metrics kept**: Prometheus counters are registered (for future use if pickle issue is resolved) but currently show zeros in logs — this is expected

## TODO

- [ ] Investigate ModelRunnerOutput pickle field loss — could enable cleaner Prometheus pipeline
- [ ] Per-phase (thinking vs response) top-k breakdown in the report
- [ ] Wire up V2 model runner top-k path (if V2 becomes default)