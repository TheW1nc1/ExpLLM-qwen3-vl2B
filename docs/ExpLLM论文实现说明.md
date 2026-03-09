# ExpLLM 论文实现说明

完全按论文 *ExpLLM: Towards Chain of Thought for Facial Expression Recognition* (arXiv:2409.02828) 实现，**仅基座替换为 Qwen3-VL-2B**，在 LLaMA-Factory 上微调。数据仅用 RAF-DB。

## 与论文的对应关系

| 论文 | 本实现 |
|------|--------|
| 基座 | Vicuna-7B + DINOv2 (LLaVA) | **Qwen3-VL-2B-Instruct** |
| 训练框架 | 自研 | **LLaMA-Factory** |
| 数据 | RAF-DB + AffectNet | **仅 RAF-DB** |
| CoT 数据 | Exp-CoT Engine 生成 | 使用已有 `rafdb_emo_au_train_mini_description.json`（含 AU、3 段式 description） |
| 训练比例 | 0.75 FER : 0.25 CoT | 0.75 : 0.25 |
| LoRA | r=8 | r=8 |
| 训练阶段 | Stage1 CC3M + Stage2 FER+CoT | **仅 Stage2**（Qwen3-VL 已对齐，跳过 CC3M） |
|  epochs | 60 (RAF-DB) | 60 |

## 数据格式

`rafdb_emo_au_train_mini_description.json` 已包含论文要求的 3 段式 CoT：

- **Key Observations**：AU 名称、强度、关联情绪
- **Overall Emotional Interpretation**：多 AU 交互、主导情绪
- **Conclusion**：最终表情标签

每张图生成 3 条 FER（instruction → 标签）+ 1 条 CoT（instruction → 描述），实现 0.75:0.25。

## 使用步骤

### 1. 数据准备（本地）

```bash
cd ExpLLM-TMM
python scripts/prepare_rafdb_expllm_paper.py
```

输出：`data/rafdb_expllm_paper.json`。

### 2. 数据增强（与论文一致：flip, rotation, erase, color jitter）

```bash
python scripts/augment_rafdb_expllm_paper.py
```

输出：`data/rafdb_expllm_paper_aug.json`，并更新 `dataset_info`。

### 3. 训练（LLaMA-Factory CLI）

```bash
llamafactory-cli train scripts/llamafactory_config_expllm_paper.yaml
```

或 Colab 单格运行：

```python
# 复制 scripts/colab_train_expllm_paper.py 整段到 Colab 运行
# 需先挂载 RAF-DB 并设置 PROJECT_ROOT、RAF_DB_ROOT
```

### 4. 评估

使用 `utils/valid_qwen3vl_llamafactory.py` 在 RAF-DB 测试集上评估：

```bash
python utils/valid_qwen3vl_llamafactory.py \
  --model_name_or_path saves/qwen3vl2b/lora/rafdb_expllm_paper
```

## 路径说明

- `media_dir`：RAF-DB 根目录（含 `basic/Image/aligned/`）
- `dataset_dir`：`data`，JSON 中 `file_name` 为 `rafdb_expllm_paper.json`
- 图片路径：`basic/Image/aligned/train_00001.jpg`（相对 `media_dir`）

## 注意事项

1. **Exp-CoT Engine**：论文用 AU 模型 + GPT-4o 生成 CoT。本实现直接使用项目已有的 `rafdb_emo_au_train_mini_description.json`，其 description 已符合 3 段式格式。
2. **模板**：使用 `qwen3_vl_nothink`，避免 `<think>`</think>` 干扰，与论文 Vicuna 输出方式一致。
3. **Stage 1**：论文用 CC3M 预训练 projector，Qwen3-VL 已对齐视觉-语言，故跳过。
