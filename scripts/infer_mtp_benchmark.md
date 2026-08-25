# `infer_mtp_benchmark.sh` 使用指南

统一入口脚本，用于在多个数据集上跑 MTP（Multi-Token Prediction）接受率测试。

## 基本语法

```bash
bash scripts/infer_mtp_benchmark.sh [选项]
```

不传任何参数时使用全部默认值（aime25 数据集，temp=0.6，TP=8，spec=3）。

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

| 选项                                                                              | 默认值                                       | 说明                                |
| --------------------------------------------------------------------------------- | -------------------------------------------- | ----------------------------------- |
| `--dataset`                                                                       | `aime25`                                     | aime25 / gpqa / gsm8k / mmlu / mt-bench |
| `--model-dir`                                                                     | `/public/panyu/hf/ckpt/Qwen/Qwen3.5-35B-A3B` | 模型目录                            |
| `--output-base`                                                                   | `/extra_panyu/output_text_pz`                | 输出根目录                          |
| `--temp`                                                                          | `0.6`                                        | 温度（见上表）                      |
| `--num-spec-tokens`                                                               | `3`                                          | MTP spec token 数                   |
| `--tp`                                                                            | `8`                                          | tensor parallel                     |
| `--max-tokens`                                                                    | `32768`                                      | 最大输出 token                      |
| `--max-model-len`                                                                 | `262144`                                     | 最大上下文长度                      |
| `--max-num-seqs`                                                                  | `256`                                        | 并发序列数                          |
| `--mode`                                                                          | `chat`                                       | chat / completion                   |
| `--no-thinking`                                                                   | （默认开启 thinking）                        | 关闭 thinking                       |
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

## 数据集映射表

| `--dataset` | HF 数据集            | subset         | split | 输入处理            |
| ----------- | -------------------- | -------------- | ----- | ------------------- |
| `aime25`    | MathArena/aime_2025  | —              | train | `--text-column problem`  |
| `gpqa`      | Idavidrein/gpqa      | gpqa_diamond   | train | `--format mcq`（选项打乱） |
| `gsm8k`     | openai/gsm8k         | main           | test  | `--text-column question` |
| `mmlu`      | cais/mmlu            | all            | test  | `--text-column question` |
| `mt-bench`  | philschmid/mt-bench  | —              | train | `--text-column turns`    |

## 注意事项

- 脚本带 `set -euo pipefail`，未识别选项会直接报错退出。
- `--help` 文本中 `∫ INT     Speculative tokens (default: 2)` 是显示乱码（实际是 `--num-spec-tokens`），且默认值 `2` 与代码里的 `3` 不一致——以代码为准。
- 模型路径和 output-base 是硬编码的服务器绝对路径（`/public/panyu/...`、`/extra_panyu/...`），换机器需用 `--model-dir` / `--output-base` 覆盖。
- 底层调用 `scripts/test_mtp_acceptance_rate_pz.py`。