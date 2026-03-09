# Colab 训练单元格检查与方案

## 一、你提的几项结论

| 项 | 结论 | 说明 |
|----|------|------|
| **Loss 欺骗** | 当前方案未根本解决 | 若仍用 `qwen3_vl` + 单标签数据（assistant 只有 "Happiness"），模型会优先学固定 `<think></think>` 串。根本解决二选一：(1) 用 `qwen3_vl_nothink`（无思维链）；(2) 用 CoT 数据：assistant 为 `<think>{每张图不同的描述}</think>`\n\nHappiness`，需 convert 时读 `description` 并写入。 |
| **断点重传** | 可做且已做 | 训练前扫描 `output_dir` 下 `checkpoint-*`，取最新并设 `resume_from_checkpoint`，LLaMA-Factory 会从该 checkpoint 续训。 |
| **重启 Colab 丢文件** | 不丢，前提是写到 Drive | `output_dir` 设为 `PROJECT_ROOT` 下的路径（如 `saves/...`），而 `PROJECT_ROOT` 在 Drive，重启后 checkpoint 仍在。`/tmp` 下的临时脚本会丢，但每次运行都会重新生成，无影响。 |
| **满足思维链** | 模板满足，数据决定是否“真”CoT | `template: qwen3_vl` 会带 `<think></think>`；只有训练数据里 assistant 含真实推理内容（如 `<think>{description}</think>`\n\n标签）才算真正在训思维链，否则仍是“空壳”CoT。 |
| **《think》+ 标签根本解决** | 需用“每条不同”的 think 内容 | 根本解决：每条样本的 `<think></think>` 里是**随样本变的描述**（例如来自 `rafdb_emo_au_train_mini_description.json` 的 description），而不是空或固定串。需在 **convert 脚本** 里增加“CoT 格式”输出（读 description，写成 `<think>{description}</think>`\n\n{emo}`），并用该 JSON 训练。 |
| **每张图输出真实标签和模型推测** | 可做 | 监控回调里用多张固定样本（如 5 张），每 N 步对这批图做推理，逐条打印「GT / 预测」。不能对“训练中每一张”都实时推理（太慢），只能对少量固定样本做代表。 |

## 二、数据侧：CoT 防 Loss 欺骗（需你本地/Colab 跑一次）

1. **convert 已支持 CoT**：`convert_rafdb_to_llamafactory(..., use_cot=True)` 时，若 item 含 `description`，assistant 写为 `<think>{description}</think>`\n\n{emo}`。`__main__` 已增加输出 `rafdb_train_emo_cot.json` 的调用。
2. 在 Colab/本地运行：`python scripts/convert_rafdb_to_llamafactory.py`，得到 `data_list/llamafactory/rafdb_train_emo_cot.json`。
3. 在 `data/dataset_info.json` 里增加一项，例如 `"rafdb_emo_cot": { "file_name": "data_list/llamafactory/rafdb_train_emo_cot.json", "formatting": "sharegpt", "columns": { "messages": "conversations", "images": "image" }, "tags": { "role_tag": "from", "content_tag": "value", "user_tag": "human", "assistant_tag": "gpt" } }`。
4. 训练单元格里把 `dataset` 改为 `"rafdb_emo_cot"`（或用你注册的名字）；模板仍用 `qwen3_vl`。

这样每条样本的 think 内容不同，模型无法靠背固定串降 loss，且满足思维链。

## 三、单元格内已完备内容（见 `scripts/colab_train_cot_one_cell.py`）

- 依赖安装 + 子进程运行（同一单元格，无需手动重启）
- `output_dir` 为 `PROJECT_ROOT/saves/...` 绝对路径在 Drive，重启 Colab 不丢
- 自动检测 `output_dir` 下最新 `checkpoint-*`，设置 `resume_from_checkpoint` 断点续传；`overwrite_output_dir: False` 避免覆盖
- 监控：`MONITOR_SAMPLES` 多张（路径+GT），每 20 步对这批图推理，逐条打印「GT | 预测」
- 防 loss 欺骗：单元格本身不生成数据；要根本解决需用上面 CoT 数据并 `dataset: rafdb_emo_cot`
