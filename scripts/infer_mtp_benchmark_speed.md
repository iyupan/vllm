# `infer_mtp_benchmark_speed.sh` 使用指南

SPEED-Bench 专用的 MTP（Multi-Token Prediction）接受率测试入口脚本，支持单轮 / 多轮、按 `category` 拆分输出。

底层调用 `scripts/test_mtp_acceptance_rate_speed.py`。

## 数据获取

`nvidia/SPEED-Bench` 在 HuggingFace 上的 parquet 已经是**预解析过的**（占位符已被 NVIDIA 替换为真实内容），可以直接下载使用。

不传 `--parquet-path` 时，shell 会自动调 `huggingface_hub.snapshot_download` 把对应 `--speed-config` 的 parquet 拉到：

```
${HF_DATA_BASE}/nvidia/SPEED-Bench/${SPEED_CONFIG}/test-*.parquet
```

`HF_DATA_BASE` 默认 `/extra_panyu/hf/data`，可用 `--hf-data-base` 覆盖。已经存在的 parquet 不会重复下载。

如果你想显式指向自己已经准备好的 parquet（例如用 `scripts/speed.py` 里的 `SPEEDBench.prepare_data` 自己跑了一遍占位符解析），传 `--parquet-path` 即可，会跳过自动下载。

## 基本语法

```bash
bash scripts/infer_mtp_benchmark_speed.sh [选项]
```

所有选项都有默认值；`--parquet-path` 可选，不传则自动下载。也支持单个 `*.parquet` 文件或包含 parquet 文件的目录。

## 常见用法

```bash
# 1. 默认配置：qualitative + temp=0.6 + thinking + 单轮（仅 turns[0]）
#    第一次跑时会自动下载到 /extra_panyu/hf/data/nvidia/SPEED-Bench/qualitative/
bash scripts/infer_mtp_benchmark_speed.sh --speed-config qualitative

# 2. 多轮完整评估（保留 SPEED-Bench 多轮设计意图）
bash scripts/infer_mtp_benchmark_speed.sh --speed-config qualitative --multi-turn

# 3. throughput 配置 + 关闭 thinking
bash scripts/infer_mtp_benchmark_speed.sh --speed-config throughput_8k --no-thinking

# 4. 限制样本数做 smoke test
bash scripts/infer_mtp_benchmark_speed.sh --speed-config qualitative --num-prompts 20

# 5. 切换温度档
bash scripts/infer_mtp_benchmark_speed.sh --speed-config qualitative --temp 0.0    # 贪婪
bash scripts/infer_mtp_benchmark_speed.sh --speed-config qualitative --temp 1.0    # thinking 推荐

# 6. 改 spec token 数 / 模型路径
bash scripts/infer_mtp_benchmark_speed.sh --speed-config qualitative --num-spec-tokens 2
bash scripts/infer_mtp_benchmark_speed.sh --speed-config qualitative --model-dir /path/to/other/model

# 7. 显式覆盖采样参数（覆盖会写进输出目录名）
bash scripts/infer_mtp_benchmark_speed.sh \
    --speed-config qualitative --temp 0.6 --top-p 0.9 --top-k 50

# 8. 自定义下载根目录 / 显式指定本地 parquet
bash scripts/infer_mtp_benchmark_speed.sh \
    --speed-config qualitative --hf-data-base /shared/hf_cache
bash scripts/infer_mtp_benchmark_speed.sh \
    --speed-config qualitative \
    --parquet-path /custom/path/to/test.parquet
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
| `--parquet-path`                                                                  | （自动下载）                                 | 已解析的 SPEED-Bench parquet 文件或目录；不传则按 `--speed-config` 自动下载 |
| `--speed-config`                                                                  | `qualitative`                                | SPEED-Bench 配置；同时决定自动下载子目录和输出目录命名 |
| `--hf-data-base`                                                                  | `/extra_panyu/hf/data`                       | 自动下载的根目录（最终路径：`{base}/nvidia/SPEED-Bench/{config}/`）|
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
- 不传 `--parquet-path` 时按 `--speed-config` 自动下载 `nvidia/SPEED-Bench` 对应配置的 parquet；只下载请求的那一档（用 `allow_patterns` 限定 `${SPEED_CONFIG}/*.parquet`），其余配置不会拉。已存在则跳过下载。
- 显式传入 `--parquet-path` 会跳过自动下载——文件不存在直接报错。
- 模型路径和 output-base 是硬编码的服务器绝对路径，换机器需用 `--model-dir` / `--output-base` 覆盖。
- `--speed-config` 现在同时决定**自动下载子目录**和**输出目录命名**。显式传 `--parquet-path` 时只影响输出命名；如果让自动下载决定 parquet，标签和数据保持一致。
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

### Phase 6：上游 parquet 已预解析，去掉手动 prepare 步骤

原方案要求用户先跑 `SPEEDBench.prepare_data` 把 13 个外部数据集解析成本地 parquet，原因是 `nvidia/SPEED-Bench` 的 `turns` 列只是占位符。后来确认 HuggingFace 上 `nvidia/SPEED-Bench` 的 parquet 已经被 NVIDIA 预解析过，可以直接用。

由此 shell 增加自动下载：不传 `--parquet-path` 时调用 `huggingface_hub.snapshot_download` 按 `--speed-config` 拉对应 parquet 到 `${HF_DATA_BASE}/nvidia/SPEED-Bench/${SPEED_CONFIG}/`。`SPEEDBench.prepare_data` / `scripts/speed.py` / `scripts/base.py` 都不再是入口的必需依赖（只在用户想自己重跑解析时才用）。

`load_speed_bench()` 早就支持 parquet 目录（`Path.rglob("*.parquet")`），所以 Python 端零改动。

### 最终结构

```
scripts/
    infer_mtp_benchmark.sh             ← 原版（aime25 / gpqa / gsm8k / mmlu / mt-bench）
    test_mtp_acceptance_rate_pz.py     ← 原版底层 py（被新脚本复用辅助函数）
    speed.py                            ← NVIDIA SPEED-Bench 数据加载类（占位符 + prepare_data）
    infer_mtp_benchmark_speed.sh       ← 新版 shell（per-category 输出目录）
    test_mtp_acceptance_rate_speed.py  ← 新版底层 py（按 category 拆 batch）
    infer_mtp_benchmark_speed.md       ← 本文档
```

### 关键设计取舍

1. **单轮为默认，多轮 opt-in**：SPEED-Bench 虽然天然多轮，但单轮用法和原 bench 直接对比，多轮调试成本更高，所以保留两条路径。
2. **复用而非复制**：通过 `sys.path` 注入从兄弟脚本导入辅助函数，避免代码漂移；副作用是两个文件之间存在隐式依赖，重命名 `test_mtp_acceptance_rate_pz.py` 时要同步改 import。
3. **per-category 必须独立 batch**：vLLM 全局 counter 限制下没有更优方案，代价是每 category 启动开销（对几十条样本可以忽略）。
4. **`--speed-config` 现在双重角色**：自动下载启用后，它既决定下载哪个 parquet，又决定输出目录命名。显式传 `--parquet-path` 时退化为只是标签——保留了在同一份 parquet 上跑不同输出布局的解耦能力。
5. **thinking 不进多轮 history**：Qwen3 等模型的多轮规范不要求保留过往 thinking；如果以后接其他模型规范不同（要保留），改一处 `histories[i].append(...)` 即可。
6. **自动下载只拉所需配置**：`snapshot_download` 用 `allow_patterns=["${SPEED_CONFIG}/*.parquet"]` 限定，不会把 6 个配置全拉下来；首次运行成本只为当前任务付出。