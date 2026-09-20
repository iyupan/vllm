# `infer_mtp_benchmark.sh` 使用指南

统一入口脚本，用于在多个数据集上跑 MTP（Multi-Token Prediction）接受率测试。

## 基本语法

```bash
bash scripts/infer_mtp_benchmark.sh [选项]
```

请从仓库根目录运行，并先激活通过 `uv` 创建的 `.venv`（`source .venv/bin/activate`），因为 shell 内部使用 `python` 启动推理。

不传任何参数时使用全部默认值（aime25 数据集，temp=0.6，TP=8，spec=3），并开启 thinking 两阶段测试。

## 生成阶段与统计口径

默认调用 `test_mtp_acceptance_rate_pz.py`，分两次执行 `llm.generate()`：

1. **Thinking 阶段**：从原始 prompt 生成，遇到 `</think>` 停止，并保留停止字符串。
2. **Response 阶段**：把 thinking 文本拼回原始 prompt，作为新请求继续生成 response。如果第一阶段未输出结尾的 `</think>`，脚本会手工补上。

两阶段分别通过累计 counter 的差值统计接受率，并分别计时；`overall` 合并两阶段的计数、输出 token 数和生成耗时。`--max-tokens` 对每个请求的每个阶段分别生效：默认 thinking 和 response 各最多生成 32768 个 token，合计上限为 65536，还受上下文长度限制。

传入 `--no-thinking` 后，只调用一次 `llm.generate()`，单次输出上限为 `--max-tokens`。`--mode completion` 必须同时传入 `--no-thinking`。

两阶段测试与一次连续生成的行为有区别：第二阶段会重新提交请求，thinking 成为 prompt 的一部分，presence/frequency penalty 的输出历史也随之重新开始。因此这里的吞吐量用于描述两阶段测试，不能直接当作一次连续生成 thinking + response 的吞吐量。

如需保留 thinking 并比较单次生成的吞吐量，可使用：

```bash
bash scripts/compare_mtp_throughput.sh --dataset aime25 --method both
```

该入口调用 `test_mtp_throughput.py`，每次测试只执行一次 `llm.generate()`，thinking 和 response 共用 `--max-tokens` 预算。它比较 greedy / probabilistic 草稿，默认温度为 1.0、`max-num-seqs=8`、`max-model-len=65536`，与本脚本的默认参数不同。它报告整体接受率和吞吐量，不提供分阶段接受率或耗时；输出为日志和 `runs.tsv`，shell 不传递保存生成文本的 `--save-output`。

## 常见用法

```bash
# 1. 默认配置：AIME25 + temp=0.6（中等采样）
bash scripts/infer_mtp_benchmark.sh

# 2. 切换数据集
bash scripts/infer_mtp_benchmark.sh --dataset gpqa
bash scripts/infer_mtp_benchmark.sh --dataset gsm8k
bash scripts/infer_mtp_benchmark.sh --dataset mmlu
bash scripts/infer_mtp_benchmark.sh --dataset mt-bench

# 3. 切换温度档（自动套用对应采样预设）
bash scripts/infer_mtp_benchmark.sh --dataset gpqa --temp 0.0   # 贪婪解码
bash scripts/infer_mtp_benchmark.sh --dataset gpqa --temp 1.0   # thinking 推荐
bash scripts/infer_mtp_benchmark.sh --dataset gpqa --temp 0.6   # 中等采样

# 4. 改 spec token 数 / 模型路径
bash scripts/infer_mtp_benchmark.sh --dataset aime25 --num-spec-tokens 2
bash scripts/infer_mtp_benchmark.sh --model-dir /path/to/other/model

# 5. 关闭 thinking 模式
bash scripts/infer_mtp_benchmark.sh --dataset gsm8k --no-thinking

# 6. 显式覆盖采样参数（覆盖会写进输出文件名）
bash scripts/infer_mtp_benchmark.sh --dataset aime25 --temp 0.6 --top-p 0.9 --top-k 50

# 7. 温度 1 + 概率 draft + 标准 rejection sampling
bash scripts/infer_mtp_benchmark.sh --dataset aime25 --temp 1.0 \
  --rejection-sample-method standard \
  --draft-sample-method probabilistic
```

## 温度预设与实际参数（`--temp`）

下表列出未显式覆盖时传给 `SamplingParams` 的参数值：

| temp        | top_p | top_k | min_p | presence_penalty | repetition_penalty |
| ----------- | ----- | ----- | ----- | ---------------- | ------------------ |
| `0.0` / `0` | 0.95  | 20    | 0.0   | 1.5              | 1.0                |
| `0.6`       | 0.95  | 20    | 0.0   | 0.0              | 1.0                |
| `1.0` / `1` | 0.95  | 20    | 0.0   | 1.5              | 1.0                |
| 其他        | 0.95  | 20    | 0.0   | 1.5              | 1.0                |

shell 按字符串匹配预设；例如 `0.60` 会进入“其他”分支。对于温度 0 和“其他”分支，shell 不传额外采样参数，底层 Python 使用上表所列的默认值。temperature 仍使用传入值，例如 `--temp 0.8` 仍是随机采样，不会自动改为贪婪。

温度 0 时走贪婪解码，top-p/top-k/min-p 不参与随机采样，但 presence penalty 仍可能改变 argmax；若不希望施加该惩罚，请显式传入 `--presence-penalty 0`。显式传入 `--top-p` 等参数会覆盖预设或 Python 默认值。

## 全部选项

| 选项                                                                              | 默认值                                       | 说明                                |
| --------------------------------------------------------------------------------- | -------------------------------------------- | ----------------------------------- |
| `--dataset`                                                                       | `aime25`                                     | aime25 / gpqa / gsm8k / mmlu / mt-bench |
| `--model-dir`                                                                     | `/public/panyu/hf/ckpt/Qwen/Qwen3.5-35B-A3B` | 模型目录                            |
| `--output-base`                                                                   | `/extra_panyu/output_text_pz`                | 输出根目录                          |
| `--temp`                                                                          | `0.6`                                        | 温度（见上表）                      |
| `--num-spec-tokens`                                                               | `3`                                          | MTP spec token 数                   |
| `--tp`                                                                            | `8`                                          | tensor parallel                     |
| `--max-tokens`                                                                    | `32768`                                      | 每个请求、每个生成阶段的输出 token 上限 |
| `--max-model-len`                                                                 | `262144`                                     | 最大上下文长度                      |
| `--max-num-seqs`                                                                  | `256`                                        | 并发序列数                          |
| `--rejection-sample-method`                                                       | `standard`                                   | 标准 rejection sampling             |
| `--draft-sample-method`                                                           | `greedy`                                     | greedy / probabilistic draft sampling |
| `--mode`                                                                          | `chat`                                       | chat / completion                   |
| `--no-thinking`                                                                   | （默认开启 thinking）                        | 关闭 thinking，改为单阶段生成        |
| `--top-p` / `--top-k` / `--min-p` / `--presence-penalty` / `--repetition-penalty` | —                                            | 显式覆盖                            |
| `--help` / `-h`                                                                   | —                                            | 打印帮助                            |

## 输出路径规则

```
{output-base}/{model-name}/{save-dir}/output-{think|nothink}-{max-tokens}-spec{N}-temp{T}[-覆盖项].json
```

例如默认运行 aime25 会得到：

```
/extra_panyu/output_text_pz/Qwen3.5-35B-A3B/aime25/output-think-32768-spec3-temp0.6.json
```

如果显式传 `--top-p 0.9`，文件名会追加 `-topp0.9`；预设里隐含的 top_p 不会写入文件名。

概率草稿会追加 `-draftprobabilistic`，非默认 completion 模式会追加 `-completion`。JSON 保存逐条生成结果；整体和分阶段的接受率、耗时及吞吐量打印到终端，不写入这个 JSON。

## 数据集映射表

| `--dataset` | HF 数据集            | subset         | split | 输入处理            |
| ----------- | -------------------- | -------------- | ----- | ------------------- |
| `aime25`    | MathArena/aime_2025  | —              | train | `--text-column problem`  |
| `gpqa`      | Idavidrein/gpqa      | gpqa_diamond   | train | `--format mcq`（选项打乱） |
| `gsm8k`     | openai/gsm8k         | main           | test  | `--text-column question` |
| `mmlu`      | cais/mmlu            | all            | test  | `--text-column question` |
| `mt-bench`  | philschmid/mt-bench  | —              | train | `--text-column turns`    |

这里的 MMLU 只读取 question，没有拼接选项；MT-Bench 将 turns 列表用换行拼成一个 prompt，并非逐轮对话。这两种配置可用于生成负载测试，不等同于标准 MMLU 正确率或 MT-Bench 多轮评估。

## 注意事项

- 脚本带 `set -euo pipefail`，未识别选项会直接报错退出。
- vLLM v0.26 下 Qwen3.5 MTP 使用 V1 Model Runner；当模型目录名包含 `Qwen3.5` 时，脚本会设置 `VLLM_USE_V2_MODEL_RUNNER=0`。
- `--rejection-sample-method probabilistic` 作为旧用法仍可用，但会转换为 `standard + draft_sample_method=probabilistic`并输出弃用警告。
- 模型路径和 output-base 是硬编码的服务器绝对路径（`/public/panyu/...`、`/extra_panyu/...`），换机器需用 `--model-dir` / `--output-base` 覆盖。
- 底层调用 `scripts/test_mtp_acceptance_rate_pz.py`。
