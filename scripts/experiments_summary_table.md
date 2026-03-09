# ExpLLM Qwen3-VL-2B RAF-DB 训练方案对比（向导师说明用）

## 实验配置与结果总表

| 方案 | 数据构成 | 训练方式 | 学习率 | 总轮数 | 总步数/checkpoint | RAF-DB FER 精度 | 说明 |
|------|----------|----------|--------|--------|-------------------|-----------------|------|
| **A. 先 5 轮纯 FER** | 仅 FER，~12k 条/轮 | 从头训练 | 2e-5 | 5 | ~385 步 | （作为基线，未单独报精度） | 为后续续训提供 checkpoint-385 |
| **B. 5 轮纯 FER + 3 轮续训（lr=2e-5）** | 仅 FER，~12k 条/轮 | 从 checkpoint-385 续训 | 2e-5 | 5+3=8 | 613 | **36.11%** | 续训阶段 lr 偏大，精度明显下降 |
| **C. 5 轮纯 FER + 3 轮续训（lr=1e-5）** | 仅 FER，~12k 条/轮 | 从 checkpoint-385 续训 | 1e-5（续训） | 5+3=8 | 613 | **较好（优于 B、D）** | 推荐方案：先打好 FER 基础再小 lr 续训 |
| **D. 纯 8 轮 5:1 FER+CoT** | 每图 5×FER + 1×CoT，~73k 条/轮 | 从头训练 | 2e-5 | 8 | 3688 | **22.10%** | 一上来混训 FER+CoT，FER 精度最差 |

---

## 关键结论（可口头/书面向导师说明）

1. **FER 精度优先时**：采用 **方案 C**（先 5 轮纯 FER，再 3 轮 1e-5 续训）效果最好；方案 B（续训仍用 2e-5）会掉到约 36%，方案 D（纯 8 轮 5:1）仅约 22%。
2. **续训学习率**：从 checkpoint-385 续训时，用 **1e-5** 比 **2e-5** 更稳，避免破坏已学好的 FER。
3. **数据与目标**：若**主要评估 RAF-DB FER**，不宜从头就 5:1 混训；**先纯 FER 再（可选）加 CoT/续训** 更利于 FER 指标。

---

## 训练与测评脚本对应关系

| 方案 | 训练脚本 | 输出目录 | 测评脚本 | 测评 CHECKPOINT |
|------|----------|----------|----------|-----------------|
| A | colab_train_expllm_4aug.py（FER_ONLY=True, RESUME=None, 5 epoch 后停或取 385） | rafdb_expllm_4aug | colab_valid_rafdb_4aug.py | 385 |
| B | 同上，RESUME=checkpoint-385, RESUME_LEARNING_RATE=2e-5（或默认） | rafdb_expllm_4aug | colab_valid_rafdb_4aug.py | 613 |
| C | 同上，RESUME=checkpoint-385, RESUME_LEARNING_RATE=1e-5 | rafdb_expllm_4aug | colab_valid_rafdb_4aug.py | 613 |
| D | colab_train_expllm_4aug.py（FER_ONLY=False, FER_COT_RATIO=(5,1), RESUME=None） | rafdb_expllm_5_1 | colab_valid_rafdb_5_1.py | 3688 |

---

## 简要表述示例（可直接用于汇报）

> 我们对比了四种训练策略：**A** 仅 5 轮纯 FER；**B** 在 A 基础上用 2e-5 再训 3 轮；**C** 在 A 基础上用 1e-5 再训 3 轮；**D** 从头 8 轮 5:1 FER+CoT。  
> RAF-DB FER 上，**C 最好**，**B 约 36%**，**D 仅约 22%**。结论：**先纯 FER 再小学习率续训** 比 **从头 FER+CoT 混训** 更利于 FER 精度，续训时学习率不宜过大（建议 1e-5）。
