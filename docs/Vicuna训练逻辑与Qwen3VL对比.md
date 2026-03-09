# Vicuna 训练逻辑与 Qwen3-VL 对比

本文档说明本项目中 **Vicuna（原版 ExpLLM）** 的训练与评估流程，以及如何与 **Qwen3-VL + LLaMA-Factory** 对齐，便于达成「Qwen3-VL 达到 Vicuna 约 80% 精度」的目标。

---

## 一、Vicuna 训练逻辑（原版 ExpLLM）

### 1. 入口与脚本

| 环节 | 文件 | 说明 |
|------|------|------|
| 训练入口 | `scripts/train_rafdb.sh` | 调用 `utils/trainface.py`，指定 Vicuna-7B + DINOv2 |
| 训练核心 | `utils/trainface.py` | 使用 `LocLLMModel`、`FACETASKDataset`、`LLaVASimpleTrainer` |
| 评估 | `utils/validfaceEMO.py` | 同模型 + 同对话格式，生成预测并写 `result_all.json` |
| 指标 | `utils/eval_metrics.py` | 读 `result_all.json`，按任务 `emo_estim` 算 **Exp Accuracy** |

### 2. 模型与数据

- **模型**  
  - LLM：`vicuna-7b-v1.5`（`model_name_or_path` / `llama_path`）  
  - 视觉：DINOv2（`dino_path`），通过 `models.LocLLMModel` 做多模态  
  - 训练方式：`tune_mm_mlp_adapter=True`，`lora_llm_enable=True`，`lora_vision_enable=True`（见 `train_rafdb.sh`）

- **训练数据**  
  - 列表文件：`data_list/train/raf-db-des.txt`  
  - 每行格式：`json_file,image_folder[,repeat_time]`  
  - 读取逻辑：`datasets/facetask.py` 中的 `face_task_anno_read(data_path)`  
    - 按行解析，每行打开对应 JSON，将 `file_name` 与 `image_folder` 拼成绝对路径，合并成 `list_data_dict`

- **对话格式**  
  - 使用 `conv_face_task`（`datasets/convsersation.py`）：角色为 **TASK / USER / ASSISTANT**  
  - 任务与问题文案：`datasets/constants.py` 中的 `FaceTaskDescription`、`FaceTaskQuestion`  
  - 情感任务名：`TASK_NAME['emo'] == 'emo_estim'`  
  - 提示构建：`FaceTaskDescription['emo_estim']` + `FaceTaskQuestion['emo_estim']`，中间插入 `PREFIX_IMAGE` + `<im_patch>` 占位（见 `facetask.py` 与 `validfaceEMO.py`）

### 3. 训练流程（trainface.py）

1. 解析参数：`ModelArguments`、`DataArguments`、`TrainingArguments`、`LoRAArguments`
2. 加载 `LocLLMModel.from_pretrained(..., llama_path=..., dino_path=...)`，并挂上 LoRA
3. `make_supervised_data_module()` → `FACETASKDataset(tokenizer, data_path=data_args.data_path, multimodal_cfg=...)`，`DataCollatorForSupervisedDataset`
4. 冻结/解冻：mm_projector 可训；ViT 用 LoRA 可训；LLM 用 LoRA 可训（由 `train_rafdb.sh` 的 `freeze_*` / `lora_*` 控制）
5. `LLaVASimpleTrainer` 训练，保存到 `output_dir`（如 `checkpoints/ckpts/RAF-DB`）

### 4. 评估流程（validfaceEMO.py）

1. 加载训好的 checkpoint（同 `LocLLMModel`），`FACETASKDataset` 从 `--question-file`（如 `data_list/test/raf-db.txt`）读测试列表
2. 使用与训练相同的 `conv_face_task` 与 `FaceTaskDescription` / `FaceTaskQuestion` 构造输入，`model.generate(max_new_tokens=100, do_sample=False)`
3. 解码、strip 得到 pred，和 gt 一起写入每个 rank 的 `sub_results/result_{rank}.json`，rank0 合并为 **`result_all.json`**
4. 每条记录包含：`task`、`gt`、`pred`、`img_path`、`offset` 等

### 5. 指标（eval_metrics.py）

- 输入：`--eval_dir` 下的 **`result_all.json`**
- 对任务 **`emo_estim`**：
  - 合法标签：`["Neutral", "Happiness", "Sadness", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]`
  - 若 `res["pred"]` 不在上述列表，计为 dismatch，不参与准确率
  - **Exp Accuracy** = 预测与 gt 完全一致的样本数 / 参与统计的样本数（即合法 pred 的条数）

因此，**「Vicuna 精度」** 在本项目中指：**RAF-DB 测试集上、任务 emo_estim、上述 8 类的 Exp Accuracy**。

---

## 二、Qwen3-VL + LLaMA-Factory 的对应关系

| 环节 | Vicuna（原版） | Qwen3-VL + LLaMA-Factory |
|------|----------------|---------------------------|
| 训练入口 | `train_rafdb.sh` → `trainface.py` | LLaMA-Factory YAML + `llamafactory-cli train` |
| 训练数据 | `raf-db-des.txt` → JSON + 图片目录 | ShareGPT 格式 JSON（如 `rafdb_train_emo.json`），由 `convert_rafdb_to_llamafactory.py` 生成 |
| 模型 | Vicuna-7B + DINOv2（LocLLM） | Qwen3-VL-2B（自带视觉编码器） |
| 评估脚本 | `validfaceEMO.py` | `utils/valid_qwen3vl_llamafactory.py` |
| 评估输出 | `result_all.json`（含 task/gt/pred） | 同：`result_all.json`，且 **task='emo_estim'**，格式与 Vicuna 一致 |
| 指标 | `eval_metrics.py` 读 `result_all.json` 算 Exp Accuracy | **可直接用同一份 `eval_metrics.py`** 对 Qwen3-VL 的 `result_all.json` 算 Exp Accuracy |

只要测试集一致（同一批 RAF-DB 测试图 + 同一份标注），且 Qwen3-VL 评估输出写到 `result_all.json` 且含 `task: "emo_estim"`，则 **Vicuna 精度** 和 **Qwen3-VL 精度** 是在同一指标（Exp Accuracy）下可比的。

---

## 三、如何做到「Qwen3-VL 达到 Vicuna 约 80% 精度」

1. **测试集一致**  
   - Vicuna 用 `data_list/test/raf-db.txt`（或 `raf-db-91.03-des.txt` 等）  
   - Qwen3-VL 评估时（`valid_qwen3vl_llamafactory.py`）应使用 **同一份测试列表/JSON** 和 **同一图片目录**，保证同一批样本、同一套 gt。

2. **指标一致**  
   - Qwen3-VL 评估脚本已写 **`result_all.json`**，且含 `task: 'emo_estim'`、`gt`、`pred`。  
   - 直接对该输出目录跑：  
     `python utils/eval_metrics.py --eval_dir <Qwen3-VL 的 output_dir>`  
   - 得到的 **Exp Accuracy** 即与 Vicuna 同口径；dismatch 数也表示「预测不在 8 类中的条数」。

3. **数值目标**  
   - 若 Vicuna 在该测试集上 Exp Accuracy = A，则「80% 精度」可理解为：Qwen3-VL 的 Exp Accuracy ≥ 0.8 × A。  
   - 若尚未有 Vicuna 数值，可先在同一测试集上跑通 Vicuna 的 `validfaceEMO.py` + `eval_metrics.py` 得到 A，再对比 Qwen3-VL 的 Exp Accuracy。

4. **调优方向（在 LLaMA-Factory 侧）**  
   - 数据：与 Vicuna 尽量同源（同一 RAF-DB 划分、同一 8 类），仅格式转为 ShareGPT；必要时做数据增强与 Vicuna 对齐。  
   - 超参：epoch、lr、batch size、LoRA rank 等，按显存与收敛情况调。  
   - 推理：`valid_qwen3vl_llamafactory.py` 中 `max_new_tokens`、`do_sample`、`temperature` 等会影响 pred 稳定性，可与 Vicuna 的 `do_sample=False` 对比（例如 Qwen3-VL 用 `do_sample=False` 做一次官方对比）。

---

## 四、训练与推理参数对比（为何 Vicuna 能到 80%）

| 项目 | Vicuna（原版） | LLaMA-Factory（当前默认） | 说明 |
|------|----------------|---------------------------|------|
| **推理** | `do_sample=False`，`max_new_tokens=100` | 原为 `do_sample=True`，`temperature=0.7`，`max_new_tokens=20` | 采样会导致预测波动、dismatch 增多；现已支持 `--do-sample` 默认关闭，与 Vicuna 对齐 |
| **有效 batch** | 80×2=160 | 16×4=64 | Vicuna 有效 batch 更大，梯度更稳 |
| **epoch** | 10 | 15 | 可适当提高 LLaMA-Factory epoch 或保持 15 |
| **LoRA** | r=8, alpha=16（trainface 默认） | r=16, alpha=32 | 可尝试 r=8/alpha=16 与 Vicuna 一致 |
| **数据增强** | `data_augmentation=True` | 未在 YAML 中显式配置 | 若 LLaMA-Factory 支持，建议开启 |
| **提示词** | 含 system：FaceTaskDescription['emo_estim'] + 问题 | 仅问题："What is the expression..." | 训练/评估可考虑加上与 Vicuna 一致的任务描述 |

---

## 五、LLaMA-Factory 达到约 80% 精度的改进措施

1. **推理与 Vicuna 对齐（必做）**  
   - 评估时使用**确定性推理**，与 Vicuna 一致：  
     `python utils/valid_qwen3vl_llamafactory.py ...`（不加 `--do-sample` 时默认 `do_sample=False`）  
   - 可选：`--max-new-tokens 100` 与 Vicuna 的 100 一致（默认已改为 50，通常足够单标签输出）。

2. **测试集一致**  
   - 使用与 Vicuna 相同的测试列表/JSON 和图片目录（如 `data_list/test/raf-db.txt` 对应格式的 JSON + 同一 `image_folder`）。

3. **训练侧可调**  
   - 在 `llamafactory_config_qwen3vl2b.yaml` 中可尝试：`lora_rank: 8`、`lora_alpha: 16`（与 Vicuna LoRA 一致）；适当增大 `per_device_train_batch_size` 或 `gradient_accumulation_steps` 以提高有效 batch。  
   - 若 LLaMA-Factory 支持图像数据增强，建议开启，与 Vicuna 的 `data_augmentation=True` 对齐。

4. **提示词对齐（可选）**  
   - 训练/评估的 user 文本可与 Vicuna 一致：先加「Expression recognition involves identifying the emotional state of a person in a given image.」再加「What is the expression of the person in the image? Please provide the expression label.」  
   - 需在 `convert_rafdb_to_llamafactory.py` 的 instruction 与 `valid_qwen3vl_llamafactory.py` 的 `prompt_text` 中同步修改。

5. **指标计算**  
   - 评估完成后对 LLaMA-Factory 输出目录执行：  
     `python utils/eval_metrics.py --eval_dir <Qwen3-VL 的 output_dir>`  
   - 得到的 Exp Accuracy 与 Vicuna 同口径，目标：**Qwen3-VL Exp Accuracy ≥ 0.8 × Vicuna Exp Accuracy**。

---

## 六、关键代码位置速查

| 功能 | 文件与位置 |
|------|------------|
| Vicuna 训练入口参数 | `scripts/train_rafdb.sh`（model_name_or_path, data_path, conv_format, LoRA 等） |
| Vicuna 数据读取 | `datasets/facetask.py`：`face_task_anno_read()`、`FACETASKDataset` |
| Vicuna 对话模板 | `datasets/convsersation.py`：`conv_face_task`；`datasets/constants.py`：`FaceTaskDescription`、`FaceTaskQuestion`、`TASK_NAME` |
| Vicuna 训练循环 | `utils/trainface.py`：`train()`，`make_supervised_data_module()`，`LLaVASimpleTrainer` |
| Vicuna 评估与 result_all.json | `utils/validfaceEMO.py`：`worker()`，`generate(..., do_sample=False, max_new_tokens=100)` |
| Vicuna 指标计算 | `utils/eval_metrics.py`：读 `result_all.json`，过滤 `emo_estim`，Exp Accuracy 与 dismatch |
| Qwen3-VL 评估与 result_all.json | `utils/valid_qwen3vl_llamafactory.py`：写 `result_all.json`（含 task='emo_estim'）；支持 `--do-sample`/`--max-new-tokens`，默认确定性推理 |

以上即为本项目中 Vicuna 训练逻辑的完整梳理，以及和 Qwen3-VL 的对比与改进措施；按「五、改进措施」执行即可在同一指标下追求「Qwen3-VL 达到 Vicuna 约 80% 精度」。
