# RAF-DB 情感类别：7 类（不接 AffectNet）

## 结论

- **RAF-DB 官方为 7 类，不含 Contempt。**
- **不接 AffectNet 时，本项目统一用 7 类。**

**7 类顺序**（`datasets.constants.EMO_LABELS_RAF_DB_7`）：

```text
Neutral, Happiness, Sadness, Surprise, Fear, Disgust, Anger
```

## 必改点（已统一为 7 类）

| 文件 | 改动 |
|------|------|
| `utils/eval_metrics.py` | `emo_estim` 使用 `EMO_LABELS_RAF_DB_7`（7 类） |
| `utils/valid_qwen3vl_llamafactory.py` | 标签列表与提示用 7 类（`_emo_order`、`emo_labels` 来自 `EMO_LABELS_RAF_DB_7`） |
| `scripts/convert_rafdb_to_llamafactory.py` | `emo_labels` 与 accuracy_focused 提示均为 7 类，不含 Contempt |

## 若后续接 AffectNet（8 类）

- 使用 `datasets.constants.EMO_LABELS_8`，并在 eval/valid/convert 中切换为 8 类逻辑。
