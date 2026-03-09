# Colab 方案一修正要点

针对你在 Colab 上的三格流程（自写增强 → dataset_info → 高精度 YAML 训练），与项目内 **方案一** 和 **eval_metrics（7 类）** 对齐的必改点。

---

## 1. 标签与评估一致（7 类，无 Contempt）

项目 RAF-DB 使用 **7 类**：`Neutral, Happiness, Sadness, Surprise, Fear, Disgust, Anger`（无 Contempt）。  
你的 `LABELS` 已是 7 类，顺序可与 `datasets.constants.EMO_LABELS_RAF_DB_7` 一致，便于评估一致：

```python
LABELS = ["Neutral", "Happiness", "Sadness", "Surprise", "Fear", "Disgust", "Anger"]
```

---

## 2. 指令与 convert 的 accuracy_focused 一致

训练/评估都用同一句「仅答一词」指令，eval 时 `valid_qwen3vl_llamafactory.py` 的强标签提取才能对齐：

```python
CONSTRAINT_PROMPT = (
    "What is the expression of the person in the image? "
    "Answer with exactly one word from: Neutral, Happiness, Sadness, Surprise, Fear, Disgust, Anger."
)
```

用户侧用 `<image>\n` + 上句即可，与 `convert_rafdb_to_llamafactory.py` 的 `accuracy_focused=True` 一致。

---

## 3. ShareGPT 字段：用 `image`（单图）

项目 `dataset_info` 里 ShareGPT 使用 **`"images": "image"`**，即 JSON 里键名为 **`image`**（单图路径）。  
`create_item` 建议改为：

```python
def create_item(img_path, label_text):
    return {
        "conversations": [
            {"from": "human", "value": "<image>\n" + CONSTRAINT_PROMPT},
            {"from": "gpt", "value": label_text}
        ],
        "image": img_path   # 单键 "image"，与 dataset_info columns 一致
    }
```

Cell 2 的 `dataset_info` 保持与项目一致：

```json
"columns": {
  "messages": "conversations",
  "images": "image"
},
"tags": {
  "role_tag": "from",
  "content_tag": "value",
  "user_tag": "human",
  "assistant_tag": "gpt"
}
```

---

## 4. 图片路径：相对路径 + media_dir（推荐）

JSON 里用 **相对 media_dir 的路径**，YAML 里设 **media_dir**，可移植且与方案一一致。

- 原始图：如 `basic/Image/aligned/xxx.jpg`（相对 RAF-DB 根）。
- 增强图：如 `basic/Image/aligned/aug/xxx_flip.jpg`，且增强脚本把图写到 `RAF-DB/...` 下对应路径。
- `create_item` 传入的 `img_path` 用上述相对路径；Colab 上 `PROJECT_ROOT` 即项目根时，`media_dir: RAF-DB` 即可。

若暂时用绝对路径，YAML 可不设 media_dir，但换机器/盘符会断；建议尽早改为相对路径 + media_dir。

---

## 5. YAML 必改

- **template**：情感分类用 **`template: qwen3_vl_nothink`**（不用 `qwen3_vl`，否则 `<think>...</think>` 固定串会摊薄分类 loss、loss 虚假下降）。
- **media_dir**：若 JSON 为相对路径，设 **`media_dir: RAF-DB`**（或 Colab 上 RAF-DB 的实际路径，如 `/content/drive/MyDrive/ExpLLM/ExpLLM-TMM/RAF-DB`）。
- **dataset**：与 Cell 2 注册名一致，如 `rafdb_sharegpt_aug`；**file_name** 相对 `dataset_dir`，如 `rafdb_sharegpt_aug.json` → 实际文件为 `data/rafdb_sharegpt_aug.json`。

示例（片段）：

```yaml
dataset: rafdb_sharegpt_aug
template: qwen3_vl_nothink
media_dir: RAF-DB
```

---

## 6. 与项目方案一的两种用法

- **A. 用项目脚本（推荐）**  
  先 `convert_rafdb_to_llamafactory.py`（accuracy_focused=True）→ 再 `augment_rafdb_for_llamafactory.py` → 训练用 `llamafactory_config_qwen3vl2b_accuracy.yaml`，dataset 改为 `rafdb_emo_aug`。数据与路径见 `docs/方案一可行性分析.md`。

- **B. 保留你的 Colab 自写增强**  
  按上面 1～5 修正后，与项目 7 类、eval 提示词和 dataset_info 一致，评估用 `valid_qwen3vl_llamafactory.py` 即可得到可比精度。

---

## 7. 自检清单（Colab）

- [ ] `LABELS` 为 7 类，且指令含 "Answer with exactly one word from: Neutral, Happiness, Sadness, Surprise, Fear, Disgust, Anger."
- [ ] ShareGPT 使用 `"image": path`，dataset_info 中 `columns.images` 对应 `"image"`，并含 `tags`。
- [ ] YAML：`template: qwen3_vl_nothink`，`media_dir` 与 JSON 中图片相对路径一致。
- [ ] 若用相对路径：增强图写入 RAF-DB 下子目录，JSON 中路径相对 RAF-DB 根。
