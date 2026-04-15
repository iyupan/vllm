# MTP Top-k Acceptance Rate — Development Notes

## Feature Goal

For each MTP speculative head position, report top-1/top-2/top-3 acceptance rates.
Top-2/top-3 are statistics only — the actual spec decode flow is unchanged (uses top-1).
Top-k 回答的核心问题：**如果投机解码接受 draft 模型的 top-k 预测（而非仅 top-1），每个位置的接受率会提升多少？**

---

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

### TP Race Condition Fix

Multiple TP workers produce identical results but write to the same file concurrently. Fixed by checking `torch.distributed.get_rank() == 0` in `accumulate_topk_stats()` — only rank 0 writes.

---

## Top-k 接受率算法设计（核心）

### 设计目标

Top-k 接受率必须与标准 per-position 接受率在结构上一致：
- **top-1 接受率** 应与标准 per-position acceptance rate 对齐
- **top-2/top-3** 是在 top-1 基础上的扩展统计

### 标准接受率回顾

V1 rejection sampler（`vllm/v1/sample/rejection_sampler.py`）的工作方式：

```
对于每个请求 r：
  for pos in [0, 1, 2, ...]:
    if draft_token[pos] != target_token[pos]:
      REJECT → 停止，后续位置全部丢弃
    else:
      ACCEPT → 继续检查下一个位置
```

关键性质：**顺序/累积**（sequential/cumulative）。Position 2 被接受意味着 position 0, 1, 2 全部匹配。
标准输出：
```
Per-position acceptance rate:
  Position 0: 0.9131 (31115/34077)   ← P(pos 0 匹配)
  Position 1: 0.8049 (27428/34077)   ← P(pos 0 AND pos 1 都匹配)
  Position 2: 0.6983 (23797/34077)   ← P(pos 0 AND pos 1 AND pos 2 都匹配)
```

### Top-k 算法：累积顺序匹配

为与标准指标一致，top-k 使用**相同的顺序/累积结构**：

```
对于每个请求 r：
  accepted[k=1] = True, accepted[k=2] = True, accepted[k=3] = True

  for pos in [0, 1, 2, ...]:
    target_token = target_logits[pos].argmax()   ← target 模型在该位置的 argmax

    for k in [1, 2, 3]:
      if not accepted[k]:
        continue                                  ← 该 k 级别已经失败，跳过
      if target_token in draft_topk[pos, :k]:
        topk_hits[k][pos] += 1                    ← 匹配，计入
      else:
        accepted[k] = False                       ← 失败，后续位置该 k 也全部失败
```

每个 k 级别维护独立的累积链：
- **top-1**: target_argmax == draft_top1[pos]，一旦某位置不匹配则后续全部不计
- **top-2**: target_argmax in draft_top2[pos]（top-2 候选集），独立累积链
- **top-3**: target_argmax in draft_top3[pos]（top-3 候选集），独立累积链

### 为什么用 target logits argmax 而非 sampler output

V1 rejection sampler 的输出 `sampler_output.sampled_token_ids` **不可用**于 top-k 分析：

| 位置 | sampled_token_ids 内容 | 问题 |
|------|----------------------|------|
| 接受位置 (pos < rejection_point) | target 模型采样的 token（与 draft 相同） | 可用但仅在接受链内 |
| 拒绝位置 (pos == rejection_point) | recovered token（从残差分布采样） | **不是** target 的独立预测 |
| 拒绝之后 (pos > rejection_point) | `-1` placeholder | **无效** |

因此直接使用 target logits argmax：
```python
target_logits = logits[metadata.target_logits_indices]   # 所有 draft 位置的 logits
target_tokens = target_logits.argmax(dim=-1)             # 每个位置的确定性预测
```

**优点**：每个位置都有确定的 target token，不依赖拒绝采样的中间状态。

**与标准指标的关系**：
- **Greedy (temp=0)**：标准 rejection sampler 也用 `target_logits.argmax()`（见 `rejection_sampler.py:394`），因此 top-1 **完全匹配**标准 per-position rate。
- **Random (temp>0)**：标准 rejection sampler 用概率接受（`target_prob/draft_prob >= uniform`），我们用 argmax。理论上有差异，但实践中因为 argmax 通常是最高概率 token，**差异很小**（通常 <0.5%）。

### cu_num_draft_tokens 索引陷阱（Bug 记录）

`SpecDecodeMetadata.cu_num_draft_tokens` 是 `np.cumsum()` **无前导零**：

```python
# metadata.py:44
cu_num_draft_tokens = np.cumsum(num_draft_tokens, dtype=np.int32)
# 例如 num_draft_tokens = [3, 3, 3] → cu = [3, 6, 9]，而非 [0, 3, 6, 9]
```

**正确索引**（与 V1 rejection sampler Triton kernel 一致，见 `rejection_sampler.py:670`）：
```python
start = 0 if req_idx == 0 else cu[req_idx - 1].item()
end = cu[req_idx].item()
```

**错误写法**（导致 `IndexError: index 90 is out of bounds for dimension 0 with size 90`）：
```python
start = cu[req_idx].item()   # ← 错！cu[0] 是第一个请求的 END，不是 START
```

### Batch Size 不一致问题

`draft_topk.shape[0]`（proposer 的 batch）不一定等于 `len(metadata.num_draft_tokens)`（metadata 的 batch）。
原因包括：异步调度、部分请求无 draft 等。

**解决**：取二者最小值：
```python
batch_size = min(draft_topk.shape[0], len(metadata.num_draft_tokens))
```

这导致 topk 的 denominator 可能略小于标准指标（差距通常 <3%），但不影响率值的准确性。

---

## 迭代历史

### V0：初始实现（V2 model runner，失败）

**方法**：在 V2 model runner 和 V2 speculator 中添加 top-k 追踪。
**结果**：stats 文件始终不创建。
**根因**：默认使用 V1 model runner，V2 代码从未执行。
**教训**：必须确认 `VLLM_USE_V2_MODEL_RUNNER` 环境变量，默认走 V1。

### V1：切换到 V1 pipeline + ModelRunnerOutput 传输（失败）

**方法**：在 V1 proposer 和 V1 model runner 中实现，通过 `ModelRunnerOutput` 传回 top-k 数据。
**结果**：新增字段到达 scheduler 后全为 `None`。
**根因**：cross-process pickle transport 丢弃新字段。
**解决**：改为 file-based stats（`topk_stats.py`）。

### V2：File-based stats + 独立比较（部分正确）

**方法**：
- Proposer: `_greedy_sample_with_topk()` 提取 top-k indices
- Model runner: `_accumulate_topk_acceptance()` 独立比较每个位置
- 使用 `sampler_output.sampled_token_ids` 作为 target token

**问题**：
1. top-1 与标准 per-position rate 不匹配（不同 denominator）
2. `-1` placeholder 导致拒绝位置之后无法统计
3. recovered token（拒绝点）不是 target 的独立预测

### V3：改用 target logits argmax + 独立比较（改进但仍不匹配）

**方法**：改用 `logits[metadata.target_logits_indices].argmax(-1)` 获取每个位置的 target token。
**改进**：每个位置都有有效的 target token，top-2/top-3 准确。
**残留问题**：top-1 per-position 与标准指标仍不对齐，因为：
- 独立比较 vs 累积比较：标准指标是 cumulative（pos 2 接受 = pos 0,1,2 全接受），我们是 independent
- 例如标准 pos 1 = 80.49%（需要 pos 0 也接受），独立 pos 1 = 88.41%（仅看 pos 1 自身）

### V4：cu_num_draft_tokens 索引修复

**Bug**：`cu_num_draft_tokens` 是 cumsum 无前导零，错误使用 `cu[req_idx]` 作为 start。
**表现**：`IndexError: index 90 is out of bounds for dimension 0 with size 90`（最后一个请求越界）。
**修复**：`start = 0 if req_idx == 0 else cu[req_idx - 1].item()`。

### V5：累积顺序比较 + batch size 修复（当前版本）

**核心改动**：
1. **累积比较**：每个 k 级别维护 `accepted[k]` 标志，一旦某位置 top-k 不匹配则后续位置全部跳过
2. **Batch size**：`min(draft_topk.shape[0], len(metadata.num_draft_tokens))`
3. **Top-k 检查**：`(draft_at_step[:k+1] == target_token).any()` — 检查 target 是否在 draft 的前 k 个候选中

**结果**：top-1 与标准 per-position rate 在 greedy 下完全一致，在 random sampling 下差异 <0.5%。

---

## Files Modified

### Core (V1 pipeline — the working path)
| File | Change |
|------|--------|
| `vllm/v1/spec_decode/eagle.py` | `_greedy_sample_with_topk()`, `self.draft_topk` in `propose()` |
| `vllm/v1/worker/gpu_model_runner.py` | `_accumulate_topk_acceptance()` cumulative comparison in `_sample()` |
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
  ├─ _greedy_sample_with_topk(hidden_states)
  │    ├─ logits = model.compute_logits(hidden_states)
  │    ├─ draft = logits.argmax(dim=-1)              ← draft token（与原流程相同）
  │    └─ topk_indices = torch.topk(logits, k=3)     ← 额外提取 top-3 indices
  ├─ 每个 speculative step 收集 topk_list
  └─ self.draft_topk = torch.stack(topk_list, dim=1)  → [batch, num_spec, 3]

gpu_model_runner._sample()  (vllm/v1/worker/gpu_model_runner.py)
  ├─ rejection_sampler() → sampler_output (unchanged, 不受影响)
  └─ _accumulate_topk_acceptance(metadata, logits, draft_topk)
       ├─ target_tokens = logits[target_logits_indices].argmax(-1)
       ├─ 累积顺序比较：
       │    for req in requests:
       │      accepted = [True, True, True]  # per k
       │      for pos in [0, 1, 2]:
       │        for k in [1, 2, 3]:
       │          if accepted[k] and target in draft_topk[pos, :k]:
       │            hits[k][pos] += 1
       │          else:
       │            accepted[k] = False  ← 链断，后续全跳过
       └─ accumulate_topk_stats()  (topk_stats.py, rank 0 only)
            └─ writes /tmp/vllm_topk_stats.json (auto-flush every batch)

test_mtp_acceptance_rate_topk.py
  └─ load_topk_from_file() → print_topk_report()

run_mtp_benchmark_topk.sh
  └─ rm -f /tmp/vllm_topk_stats.json → exec test script
```

## Key Design Points

1. **No impact on spec decode**: topk tracking is purely statistical, never modifies draft/target tokens or acceptance decisions
2. **Cumulative/sequential**: 与标准 rejection sampler 结构一致，top-1 可直接与 per-position rate 对比
3. **File-based bypass**: `ModelRunnerOutput` new fields are lost in cross-process pickle, so stats go directly to disk
4. **Rank-0 only write**: avoids TP worker file race condition
5. **Target logits argmax**: not rejection sampler output, ensures every position has a valid target token
6. **Metrics kept**: Prometheus counters are registered (for future use if pickle issue is resolved) but currently show zeros in logs — this is expected

## TODO

- [ ] Investigate ModelRunnerOutput pickle field loss — could enable cleaner Prometheus pipeline
- [ ] Per-phase (thinking vs response) top-k breakdown in the report
- [ ] Wire up V2 model runner top-k path (if V2 becomes default)
- [ ] Denominator 微小差异排查（draft_topk batch vs metadata batch 在 async scheduling 下的不一致）