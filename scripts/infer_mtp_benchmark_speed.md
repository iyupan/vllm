# `infer_mtp_benchmark_speed.sh` 使用指南

SPEED-Bench 专用的 MTP（Multi-Token Prediction）接受率测试入口脚本，支持单轮 / 多轮、按 `category` 拆分输出。

底层调用 `scripts/test_mtp_acceptance_rate_speed.py`。

## 前置条件：准备 parquet

脚本不下载 / 不解析数据，**只负责加载已经准备好的 parquet 跑评估**。需要你自己提前把 SPEED-Bench parquet 放到本地，可选两条路：

```python
# A. 上游 parquet 已经预解析过（NVIDIA 在 HuggingFace 上传的 nvidia/SPEED-Bench）
from datasets import load_dataset
ds = load_dataset("nvidia/SPEED-Bench", "qualitative", split="test")
ds.to_parquet("/extra_panyu/data/speed/qualitative/test.parquet")

# B. 上游若是占位符版，则用 scripts/speed.py 里的 SPEEDBench.prepare_data
#    （会拉 13 个外部 HF 数据集来填充 turns 列）
from scripts.speed import SPEEDBench
SPEEDBench.prepare_data(
    output_dir="/extra_panyu/data/speed/qualitative",
    config_name="qualitative",
)
```

后续推理直接读这个 parquet，不再联网。

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
    "num_drafts": 12345,
    "num_draft_tokens": 37035,
    "num_accepted_tokens": 15678,
    "acceptance_counts_per_pos": [8901, 4567, 2210],
    "acceptance_rate": 0.4233,
    "mean_accept_length": 2.27,
    "num_output_tokens": 45123,
    "elapsed_seconds": 12.34,
    "thinking": { "num_drafts": ..., "acceptance_rate": ... },
    "response": { "num_drafts": ..., "acceptance_rate": ... }
  },
  "records": [
    {
      "question_id": "...",
      "category": "code_generation",
      "source": "livecodebench/code_generation_lite",
      "turns": ["..."],
      "responses": ["..."],
      "thinking": ["..."],
      "num_output_tokens": [1234]
    },
    ...
  ]
}
```

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