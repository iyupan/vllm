# `infer_mtp_benchmark_speed.sh` 使用指南

SPEED-Bench 专用的 MTP（Multi-Token Prediction）接受率测试入口脚本，支持单轮 / 多轮、按 `category` 拆分输出。

底层调用 `scripts/test_mtp_acceptance_rate_speed.py`。

## 前置条件：准备 parquet

脚本不下载 / 不解析数据，**只负责加载已经准备好的 parquet 跑评估**。

默认数据位置由 shell 里的 `DATA_BASE` + `SPEED_CONFIG` 拼出：

```
${DATA_BASE}/${SPEED_CONFIG}/test.parquet
# 默认 = /extra_panyu/data/speed/qualitative/test.parquet
```

需要换位置时用 `--data-base` / `--speed-config` / `--parquet-path` 覆盖（见后文"全部选项"）。`--parquet-path` 既支持单个 `*.parquet`，也支持包含 parquet 文件的目录。

## 基本语法

```bash
bash scripts/infer_mtp_benchmark_speed.sh [选项]
```

`--parquet-path` 可选，可以是单个 `*.parquet` 文件，也可以是包含 parquet 文件的目录。不传时默认 `${DATA_BASE}/${SPEED_CONFIG}/test.parquet`，即默认读 `/extra_panyu/data/speed/qualitative/test.parquet`（用 `--speed-config` 切换子目录、`--data-base` 切换根目录）。

## 常见用法

```bash
# 1. 默认配置：qualitative + temp=0.6 + thinking + 单轮（仅 turns[0]）
#    隐式读 /extra_panyu/data/speed/qualitative/test.parquet
bash scripts/infer_mtp_benchmark_speed.sh

# 2. 多轮完整评估（保留 SPEED-Bench 多轮设计意图）
bash scripts/infer_mtp_benchmark_speed.sh --multi-turn

# 3. throughput 配置 + 关闭 thinking
bash scripts/infer_mtp_benchmark_speed.sh --speed-config throughput_8k --no-thinking

# 4. 限制样本数做 smoke test
bash scripts/infer_mtp_benchmark_speed.sh --num-prompts 20

# 5. 切换温度档
bash scripts/infer_mtp_benchmark_speed.sh --temp 0.0    # 贪婪
bash scripts/infer_mtp_benchmark_speed.sh --temp 1.0    # thinking 推荐

# 6. 改 spec token 数 / 模型路径
bash scripts/infer_mtp_benchmark_speed.sh --num-spec-tokens 2
bash scripts/infer_mtp_benchmark_speed.sh --model-dir /path/to/other/model

# 7. 显式覆盖采样参数（覆盖会写进输出目录名）
bash scripts/infer_mtp_benchmark_speed.sh --temp 0.6 --top-p 0.9 --top-k 50

# 8. 自定义 parquet 路径或数据根目录
bash scripts/infer_mtp_benchmark_speed.sh \
    --parquet-path /custom/path/to/test.parquet
bash scripts/infer_mtp_benchmark_speed.sh \
    --data-base /shared/speed_data --speed-config throughput_8k
```

## 三档温度预设（`--temp`）

| temp  | top_p | top_k | min_p | presence_penalty | repetition_penalty |
| ----- | ----- | ----- | ----- | ---------------- | ------------------ |
| `0.0` | —     | —     | —     | —                | —                  |
| `0.6` | 0.95  | 20    | 0.0   | 0.0              | 1.0                |
| `1.0` | 0.95  | 20    | 0.0   | 1.5              | 1.0                |
| 其他  | 退化为贪婪（无采样参数）                                        |

显式传入 `--top-p` 等参数会覆盖预设。

## 全部选项

| 选项                                                                              | 默认值                                       | 说明                                                |
| --------------------------------------------------------------------------------- | -------------------------------------------- | --------------------------------------------------- |
| `--parquet-path`                                                                  | `${DATA_BASE}/${SPEED_CONFIG}/test.parquet`  | SPEED-Bench parquet 文件或目录；不传时从 `--data-base` 和 `--speed-config` 拼出 |
| `--speed-config`                                                                  | `qualitative`                                | SPEED-Bench 配置；既决定默认 parquet 子目录，也决定输出目录命名 |
| `--data-base`                                                                     | `/extra_panyu/data/speed`                    | 默认 parquet 根目录                                 |
| `--model-dir`                                                                     | `/public/panyu/hf/ckpt/Qwen/Qwen3.5-35B-A3B` | 模型目录                                            |
| `--output-base`                                                                   | `/extra_panyu/output_text_pz`                | 输出根目录                                          |
| `--temp`                                                                          | `0.6`                                        | 温度（见上表）                                      |
| `--num-spec-tokens`                                                               | `3`                                          | MTP spec token 数                                   |
| `--tp`                                                                            | `8`                                          | tensor parallel                                     |
| `--max-tokens`                                                                    | `32768`                                      | 单次最大输出 token                                  |
| `--max-model-len`                                                                 | `262144`                                     | 最大上下文长度                                      |
| `--max-num-seqs`                                                                  | `256`                                        | 并发序列数                                          |
| `--num-prompts`                                                                   | —                                            | 限制读取的 request 数（smoke test 用）              |
| `--multi-turn`                                                                    | （默认关闭，仅用 `turns[0]`）                | 迭代所有 turn，按 chat history 累积                 |
| `--mode`                                                                          | `chat`                                       | chat / completion                                   |
| `--no-thinking`                                                                   | （默认开启 thinking）                        | 关闭 thinking                                       |
| `--save-prompt`                                                                   | （默认关闭）                                 | 把 chat-template 渲染后的 prompt 写入 records       |
| `--top-p` / `--top-k` / `--min-p` / `--presence-penalty` / `--repetition-penalty` | —                                            | 显式覆盖                                            |
| `--help` / `-h`                                                                   | —                                            | 打印帮助                                            |

## 输出路径规则

```
{output-base}/{model-name}/speed-bench/{speed-config}/output-{single|multi}-{think|nothink}-{max-tokens}-spec{N}-temp{T}[-覆盖项]/
    <category_a>.json       # 每个 category 一个文件
    <category_b>.json
    ...
    _summary.json           # 跨 category 汇总
```

默认运行（qualitative + thinking + 单轮）会得到：

```
/extra_panyu/output_text_pz/Qwen3.5-35B-A3B/speed-bench/qualitative/output-single-think-32768-spec3-temp0.6/
    qa_short.json
    summarization.json
    code_generation.json
    role_play.json
    ...
    _summary.json
```

如果显式传 `--top-p 0.9`，目录名追加 `-topp0.9`；预设里隐含的 top_p 不会写入。

## 输出文件格式

### 每个 category 的 `<category>.json`

```json
{
  "summary": {
    "category": "code_generation",
    "num_requests": 24,
    "num_total_turns": 24,
    "num_spec_tokens": 3,
    "multi_turn": false,
    "enable_thinking": true,
    "total_thinking_tokens": 30210,
    "total_response_tokens": 14913,
    "overall": {
      "num_drafts": 12345,
      "num_draft_tokens": 37035,
      "num_accepted_tokens": 15678,
      "acceptance_counts_per_pos": [8901, 4567, 2210],
      "per_head": [
        {"position": 0, "accepted": 8901, "rate": 0.7211},
        {"position": 1, "accepted": 4567, "rate": 0.3699},
        {"position": 2, "accepted": 2210, "rate": 0.1791}
      ],
      "acceptance_rate": 0.4233,
      "mean_accept_length": 2.27,
      "num_output_tokens": 45123,
      "elapsed_seconds": 12.34,
      "throughput_tok_per_sec": 3656.6
    },
    "thinking": "...同 overall 结构（_phase_dict）...",
    "response": "...同 overall 结构（_phase_dict）..."
  },
  "records": [
    {
      "question_id": "...",
      "category": "code_generation",
      "source": "livecodebench/code_generation_lite",
      "turns": ["..."],
      "responses": ["..."],
      "thinking": ["..."],
      "thinking_tokens": [856],
      "num_output_tokens": [378],
      "prompt": ["...chat-template-rendered string fed to LLM..."]
    }
  ]
}
```

字段说明：

- `summary.overall` / `summary.thinking` / `summary.response` 是平级三段，结构都由 `_phase_dict()` 生成（字段一致）；之前的"overall 字段铺在顶层 + thinking/response 子 dict"格式已废弃。
- `per_head[i].rate = acceptance_counts_per_pos[i] / num_drafts`，**累积口径**（与 vLLM 内部日志 `vllm/v1/spec_decode/metrics.py:97` 一致），物理含义是"草稿至少被接受到位置 i 的概率"，必然单调不增。
- `acceptance_rate` 为小数（0–1），stdout 显示百分数（×100）。
- `throughput_tok_per_sec = num_output_tokens / elapsed_seconds`。
- 记录里 `num_output_tokens[t]` 是第 t 轮 response 的 token 数；`thinking_tokens[t]` 是第 t 轮 thinking 段的 token 数（仅 `enable_thinking` 时存在）。`summary.overall.num_output_tokens` 是 thinking + response 合计；`summary.total_thinking_tokens` / `total_response_tokens` 是分别的合计（仅 `enable_thinking` 时存在）。
- `prompt[t]` 是第 t 轮 chat-template 渲染后实际喂给 LLM 的字符串（多轮模式下含累积 history）。**仅 `--save-prompt` 时存在**（默认关闭，因为 prompt 在多轮 + 长上下文下可能非常大）。

### 汇总 `_summary.json`

```json
{
  "overall": { 同上 summary 结构，category="__overall__" },
  "per_category": {
    "code_generation": { summary },
    "summarization":   { summary },
    ...
  }
}
```

## Per-category 测试机制

vLLM 的 MTP 接受率指标（`vllm:spec_decode_num_drafts` 等）是**全局累加 counter**。要拿到每个 category 独立的接受率，必须把每个 category 跑成独立 batch，并在 batch 前后取 counter 差值。

底层 Python 脚本的 `main()` 流程：

1. 加载 parquet → 全部 requests
2. 按 `category` 分组（dict）
3. 对每个 category：
   - 调用 `run_single_turn` / `run_multi_turn`
   - 内部 `run_generation` 在 `llm.generate` 前后取 metrics 差值
   - 把差值打包成该 category 的 4-tuple `(num_drafts, num_draft_tokens, num_accepted, acceptance_counts_per_pos)`
   - 写一个 `<category>.json`
4. `merge_results` 累加所有 category 的指标 → 写 `_summary.json`

代价：每个 category 跑一次 `llm.generate` 比一次性跑全部样本略低效（vLLM batch 调度有少量启动开销），但对几十-几百条样本可以忽略。

## 单轮 vs 多轮

SPEED-Bench 的 `qualitative` 配置里很多数据集是**多轮对话**（MTBench-101、ChatRAG-Bench、MMLU-Pro、RoleBench）；`throughput_*` 配置基本都是单轮。

| 模式            | 行为                                                          | 适用场景                                                              |
| --------------- | ------------------------------------------------------------- | --------------------------------------------------------------------- |
| **单轮**（默认）| 只用 `turns[0]`，每条 request 推理一次                        | 与原 MTP bench 行为对齐，样本数即 request 数，便于和其他数据集对比    |
| **多轮**        | 按 turn index 迭代；每轮 batch 化所有"还没结束的"requests     | 完整覆盖 SPEED-Bench 多轮设计意图，能看到 multi-turn 下的接受率变化   |

多轮模式下：
- 每条 request 维护独立 chat history（`messages` list）
- 每轮把 `turns[t]` 加入 history → 应用 chat template → 推理 → 把 response 加回 history
- thinking 被剥离不进 history（符合 Qwen3 多轮规范）
- 每轮的 MTP 指标都被累加到该 category 总计

`--multi-turn` 必须配合 `--mode chat`。

## 注意事项

- 脚本带 `set -euo pipefail`，未识别选项直接报错退出。
- `--parquet-path` 不传时会拼成 `${DATA_BASE}/${SPEED_CONFIG}/test.parquet`；若该路径不存在会立即退出并提示。
- 模型路径、output-base、data-base 都是硬编码的服务器绝对路径，换机器需用 `--model-dir` / `--output-base` / `--data-base` 覆盖。
- `--speed-config` 在自动拼路径模式下既影响默认 parquet 子目录，又影响输出目录命名；显式传 `--parquet-path` 时退化为只是输出标签，加载行为完全由 `--parquet-path` 决定（理论上可以配置不一致，但不推荐）。
- `category` 名字若含空格、`/` 等字符会被 `_` 替换；同名碰撞会自动加 `_2`、`_3` 后缀避免覆盖。
- `_summary.json` 永远在最后写出 — 中途 Ctrl+C 会丢这个文件，但已写出的 per-category 文件保留。
- 与 `infer_mtp_benchmark.sh` 的差异：那个脚本对应的底层 py（`test_mtp_acceptance_rate_pz.py`）不分 category；SPEED-Bench 因为天然带 `category` 列，per-category 拆分才有意义。

---

## 开发过程记录

下面记录这套脚本的设计/实现过程，便于后续维护时理解为什么这么写。

### Phase 1：兼容性分析

最初的问题：`scripts/speed.py`（NVIDIA SPEED-Bench 的数据加载类）能否配合现成的 `infer_mtp_benchmark.sh` 直接跑？

读完两边代码后结论是**不能直接复用**：

| 维度                | `infer_mtp_benchmark.sh` / `test_mtp_acceptance_rate_pz.py` | `scripts/speed.py`                                       |
| ------------------- | ----------------------------------------------------------- | -------------------------------------------------------- |
| 数据来源            | 标准 HF `load_dataset(name, split=...)` 或本地 jsonl/json   | 自定义 `SPEEDBench(Dataset)` 类，输出 `Request` 对象     |
| 数据结构            | 单一文本列；list 列简单 `\n.join`                            | `turns: list[str]` + `category` + `question_id`，多轮   |
| 数据获取            | 直接 `row[text_column]`                                     | 先 `prepare_data` 解析 `TURNS_PLACEHOLDER` 才有真实内容  |
| 框架依赖            | 仅 `datasets`                                               | 继承自 `.base import Dataset, Request`（SPECDEC_BENCH） |
| MCQ 路径            | 硬编码 GPQA                                                 | 不适用                                                   |

直接套 `infer_mtp_benchmark.sh` 会读到 `"FULL BENCHMARK DATA SHOULD BE FETCHED FROM..."` 这串占位符，根本没意义。

### Phase 2：方案选型

候选方案：

1. **桥接转换**：把 SPEED-Bench parquet 转成扁平 jsonl，走 `--dataset local` — 简单但丢失多轮、category 信息。
2. **改造现有 py**：在 `test_mtp_acceptance_rate_pz.py` 里加 `--dataset speed-bench` 分支 — 单文件膨胀，且与现有 GPQA/MCQ 逻辑纠缠。
3. **独立新脚本** — 接受 parquet，复用现有的辅助函数，专门处理 SPEED-Bench 的 schema。

最终选了方案 3。理由：SPEED-Bench 的多轮 + 占位符 + per-category 都是它特有的，不应该污染通用脚本。

### Phase 3：第一版实现（单文件输出）

第一版做了：

- `load_speed_bench()`：直接用 PyArrow 读 parquet，做占位符校验
- `run_generation()`：把原脚本里 thinking/no-thinking 的两套生成逻辑抽出来，复用以支持多轮场景
- `run_single_turn()` / `run_multi_turn()`：两条 driver
  - 单轮：仅用 `turns[0]`，与原 bench 对齐
  - 多轮：按 turn index 迭代，每条 request 维护独立 `messages` history；assistant 历史只保留 response（剥掉 thinking，符合 Qwen3 规范）
- 复用 `test_mtp_acceptance_rate_pz.py` 里的 `collect_spec_decode_metrics` / `print_phase_report` / `split_thinking`，通过 `sys.path.insert` 兄弟模块导入避免代码重复
- 输出单个 JSON 文件，含 `summary` + `records`

新 shell `infer_mtp_benchmark_speed.sh` 沿用原 shell 的温度预设和 override 优先级，新增 `--parquet-path`、`--speed-config`、`--multi-turn`、`--num-prompts`。

### Phase 4：发现 per-category 才是正确粒度

第一版输出是**全局聚合**接受率，但 SPEED-Bench 的核心价值就是 13 个外部数据集 × 多种 category（writing / code / qa / role-play / ...），混在一起测掩盖了不同任务下 MTP 表现差异。

vLLM 的 MTP 接受率指标（`vllm:spec_decode_num_drafts` 等）是**全局累加 counter**，不能按 category tag 分组。要做 per-category 必须：**每个 category 单独跑一个 batch，在 batch 前后取 counter 差值**。

幸运的是 `run_generation()` 内部就是这么实现的（`m_before`/`m_after` 取差），所以只需要把外层的 batching 粒度改成 per-category，逻辑不动。

### Phase 5：第二版（per-category 输出）

改动：

1. `main()` 重写：先按 `category` 分组成 dict，循环跑每个 category
2. 抽出 `build_summary()` — per-category 和 overall 共用 summary 构建逻辑
3. 新增 `merge_results()` — 把每个 category 的 `(metrics_total, metrics_thinking, metrics_response)` 累加成 overall
4. 新增 `sanitize_filename()` — 把 category 名（可能含空格、`/`）转成安全文件名，并对碰撞做 `_2`/`_3` 去重
5. `print_per_category_summary()` 升级 — 不再只显示 turn/token 数，新增每个 category 的接受率和 mean accept length
6. `--save-output` 语义从**单文件**改为**目录**

shell 同步：`OUTPUT_FILE`（含 `.json`）→ `OUTPUT_RUN_DIR`（目录）。

### Phase 6：尝试自动下载，最后回归"只 load 已有 parquet"

中间一度想把数据获取也内聚进 shell：先后试过 `huggingface_hub.snapshot_download` 走 `allow_patterns` 拉部分 shard、用 `datasets.load_dataset` 走 metadata 拉完整 shard、以及调 `SPEEDBench.prepare_data` 触发占位符解析。每一步都有 corner case（partial 下载、上游版本不一致、需要给 `speed.py` 补 `__init__.py + base.py` 桩才能 import 等）。

最终决定回归原样：脚本只负责加载用户已经准备好的 parquet 跑评估，数据获取由用户在 README "前置条件" 里按需选 A 路（`load_dataset` + `to_parquet`）或 B 路（`SPEEDBench.prepare_data`）手动产出。这样：

- shell 不需要管 HF 缓存目录、不需要清理残留 shard、不需要兼容上游 placeholder/resolved 两种状态
- 评估流程的边界清晰：`--parquet-path` 是契约入口，由调用方负责数据正确性

### 最终结构

```
scripts/
    infer_mtp_benchmark.sh             ← 原版（aime25 / gpqa / gsm8k / mmlu / mt-bench）
    test_mtp_acceptance_rate_pz.py     ← 原版底层 py（被新脚本复用辅助函数）
    speed.py                            ← NVIDIA SPEED-Bench 数据加载类（按需用 prepare_data 解析占位符）
    infer_mtp_benchmark_speed.sh       ← 新版 shell（per-category 输出目录）
    test_mtp_acceptance_rate_speed.py  ← 新版底层 py（按 category 拆 batch）
    infer_mtp_benchmark_speed.md       ← 本文档
```

### 关键设计取舍

1. **单轮为默认，多轮 opt-in**：SPEED-Bench 虽然天然多轮，但单轮用法和原 bench 直接对比，多轮调试成本更高，所以保留两条路径。
2. **复用而非复制**：通过 `sys.path` 注入从兄弟脚本导入辅助函数，避免代码漂移；副作用是两个文件之间存在隐式依赖，重命名 `test_mtp_acceptance_rate_pz.py` 时要同步改 import。
3. **per-category 必须独立 batch**：vLLM 全局 counter 限制下没有更优方案，代价是每 category 启动开销（对几十条样本可以忽略）。
4. **`--speed-config` 只是标签**：实际加载行为完全由 `--parquet-path` 决定，给 shell 留了一个解耦设计，便于未来在同一份 parquet 上跑不同的输出目录布局。
5. **thinking 不进多轮 history**：Qwen3 等模型的多轮规范不要求保留过往 thinking；如果以后接其他模型规范不同（要保留），改一处 `histories[i].append(...)` 即可。
6. **数据获取留给上游**：尝试过把下载/解析内聚进 shell，发现上游 parquet 在 placeholder / resolved 之间状态不稳定，每加一种自动模式都得加一堆 corner case。最后回归"只 load 现成 parquet"，把数据正确性交给调用方。

---

## 附：per-head 接受率 + thinking token 长度落盘

### 背景

原 `build_summary()` 只写 `acceptance_counts_per_pos`（raw counts，没分母）和总 `num_output_tokens`（thinking + response 合并），看不出**每个 MTP 头自己的接受率**，也看不出 **thinking 段单独贡献了多少 token**。需要：

1. 每个 MTP 头（默认 3 头）的接受率写入输出文件。
2. 三段（overall / thinking / response）都给 `mean_accept_length` 和 `per_head` 接受率。
3. thinking token 长度按 record（每轮一项）和按 summary（合计）都落盘。
4. 加派生吞吐量字段，方便后处理对比。

### 口径

per-head 接受率使用**累积口径**，与 vLLM 内部日志 `vllm/v1/spec_decode/metrics.py:97` 一致：

```
rate[i] = acceptance_counts_per_pos[i] / num_drafts
```

来源：`SpecDecodingStats.observe_draft()`（`vllm/v1/spec_decode/metrics.py:38`）在草稿被接受 k 个 token 时，对 `pos[0..k-1]` 各 +1。由于 spec decoding 是**前缀匹配**（一旦某位拒绝，后面所有位置都不再算接受），`acceptance_counts_per_pos[i]` 的物理含义是"草稿至少被接受到位置 i+1 的次数"，因此 `rate[i]` = "草稿至少被接受到位置 i 的概率"，序列必然单调不增。

> 条件接受率（给定前 i-1 个头都接受时第 i 个头的命中率）可由 `acceptance_counts_per_pos[i] / acceptance_counts_per_pos[i-1]` 反推，本次未输出。

### 改动（speed.py）

1. **新增 `_phase_dict()`**：把单相位 JSON 字段构造抽出来，返回包含
   - `num_drafts` / `num_draft_tokens` / `num_accepted_tokens` / `acceptance_counts_per_pos`（旧字段保留兼容）
   - `per_head`：长度 = `num_spec_tokens` 的列表，每项 `{position, accepted, rate}`
   - `acceptance_rate` = `nat / ndt`（小数；stdout 显示百分数 ×100）
   - `mean_accept_length` = `1 + nat / nd`
   - `num_output_tokens` / `elapsed_seconds`
   - `throughput_tok_per_sec` = `num_output_tokens / elapsed_seconds`

2. **`build_summary()` 重构**：`overall` / `thinking` / `response` 是平级三段子 dict，结构都由 `_phase_dict()` 生成（之前是 overall 字段铺在 summary 顶层、thinking/response 作为子 dict，现已统一）。`enable_thinking` 时在 summary 顶层新增 `total_thinking_tokens` / `total_response_tokens` 聚合字段。`print_per_category_summary()` 同步改为读 `summary["overall"]`。

3. **per-prompt thinking token 跟踪**：`run_generation()` 把原来一次性 `sum(...)` 改成先列出 `per_thinking_tokens` / `per_response_tokens`，再求 sum；返回 dict 同时携带列表和合计。

4. **records 加 `thinking_tokens` 字段**：`run_single_turn()` / `run_multi_turn()` 在 `enable_thinking` 时给每条 record 写 `thinking_tokens: list[int]`（与 `responses` / `num_output_tokens` 等长，每轮一项）。

5. **`merge_results()` 累加 token**：跨 category 累加 `total_thinking_tokens` / `total_response_tokens`，传播到 overall summary。

6. **`run_multi_turn()` 内部**：新增 `per_think_tok` 列表（按 request × turn）；返回 aggregated 时带 `total_thinking_tokens` (= `th_tok`) / `total_response_tokens` (= `rs_tok` 或 `tot_tok`，取决于 `enable_thinking`)。

7. **`main()`**：`run_single_turn` 返回的 raw 重打包到 result 时，把新加的 `total_thinking_tokens` / `total_response_tokens` 也带过去。

8. **`--save-prompt` 开关**：新增 argparse flag（默认关）。打开时 records 写入 `prompt: list[str]`（chat-template 渲染后实际喂给 LLM 的字符串，每轮一项）。`run_multi_turn()` 在 flag 为 false 时跳过 `per_prompt` 列表分配以避免内存浪费。shell 同步加 `--save-prompt` 透传。

### 兼容性

- **summary 三段平级（破坏性变更）**：之前 overall 字段（`num_drafts` / `acceptance_rate` / `num_output_tokens` 等）直接挂在 `summary` 顶层，现在统一收进 `summary["overall"]` 子 dict，与 `summary["thinking"]` / `summary["response"]` 平级。读旧 JSON 的下游脚本需要把 `summary["xxx"]` 改成 `summary["overall"]["xxx"]`。
- **`acceptance_counts_per_pos` 字段保留**：在每段（overall / thinking / response）里都没删 raw counts，只是新增 `per_head` 等字段。
- **records 字段**：原有 `responses` / `num_output_tokens` / `thinking` 不变；新增 `thinking_tokens`（仅 `enable_thinking` 时）和 `prompt`（仅 `--save-prompt` 时）。
- **stdout 输出**：`print_speed_report` / `print_phase_report` 没改，与之前完全一致。
- **`test_mtp_acceptance_rate_pz.py` 不动**：那个脚本有它自己的 `--save-output` 行为，本次不涉及。