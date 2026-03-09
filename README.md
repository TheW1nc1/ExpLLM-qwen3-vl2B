# ExpLLM: Towards Chain of Thought for Facial Expression Recognition

[[`arXiv`](https://arxiv.org/abs/2409.02828)][[`Paper`](https://ieeexplore.ieee.org/document/10948346)][[`Project`](https://starhiking.github.io/ExpLLM_Page/)]

> [ExpLLM: Towards Chain of Thought for Facial Expression Recognition](https://github.com/TheW1nc1/ExpLLM-qwen3-vl2B)  
> The_wind
> ```(Modified Version)```

![overview](./img/pipeline.jpg)

## Installation

### 1. Clone code
```shell
    git clone https://github.com/starhiking/ExpLLM_TMM
    cd ./ExpLLM_TMM
```
### 2. Create a conda environment for this repo
```shell
    conda create -n ExpLLM python=3.10
    conda activate ExpLLM
```
### 3. Install CUDA 11.7 (other version may not work)
```shell
    conda install -c conda-forge cudatoolkit-dev
```
### 4. Install PyTorch following official instruction (should match cuda version)
```shell
    conda install pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```
### 5. Install other dependency python packages (do not change package version)
```shell
    pip install pycocotools
    pip install opencv-python
    pip install accelerate==0.21.0
    pip install sentencepiece==0.1.99
    pip install transformers==4.31.0
```
### 6. Prepare dataset
Download RAF-DB and AffectNet-Kaggle from website and put the zip file under the directory following below structure, (xxx.json) denotes their original name.

```
RAF-DB/basic/Image/aligned/
├── train*.jpg
└── test*.jpg

AffectNet-kaggle/
├── README.md
├── train-sample-affectnet.csv
├── valid-sample-affectnet.csv
├── train_class/
│   ├── class001/
│   │   └── *.jpg
│   ├── class002/
│   ├── class003/
│   ├── ...
│   └── class008/
└── val_class/
    ├── class001/
    ├── class002/
    ├── ...
    └── class008/
```
## Usage

### 1. Download trained model

```shell
    git lfs install

    git clone https://huggingface.co/starhiking/ExpLLM/tree/main

    mv ExpLLM/ckpts checkpoints/ckpts
    mv ExpLLM/model_weights checkpoints/model_weights

    # clone vicuna1.5
    cd checkpoints/model_weights
    git clone https://huggingface.co/lmsys/vicuna-7b-v1.5
```

### 2. Train and Eval Model
Change `IDX` option in script to specify the gpu ids for evaluation, multiple ids denotes multiple gpu evaluation.

```shell
    # train on raf-db
    bash scripts/train_rafdb.sh

    # evaluate on raf-db val set
    bash scripts/valid_rafdb.sh
```

Accuracy:

**1. Original ExpLLM (Vicuna-7B):**
![Accuracy result](./img/result.jpg)

**2. Qwen3-VL-2B Fine-tuned (RAF-DB):**
- **Exp dismatch number:** 0
- **Exp Accuracy:** 0.3869
- **Checkpoint:** `saves/qwen3vl2b/lora/rafdb_expllm_4aug/checkpoint-1765`

![Qwen3 Accuracy result](./img/qwen3_result.png)

Note that GPU memory should not be less than 24GB.

### 3. 使用 LLaMA-Factory 微调 Qwen3-VL（含 Colab/T4）

若在 Colab 等环境用 LLaMA-Factory 训练时出现：

```text
ValueError: The number of images does not match the number of <image> tokens in [...]
```

**原因**：训练用的 JSON 里，每条带图片的样本必须在**用户消息的文本**中包含与图片数量相同的 `<image>` 占位符，否则 LLaMA-Factory 会报错。

**做法**：

1. **务必用转换脚本生成训练 JSON**（不要直接用 `rafdb_train.json` 等原始标注）：
   ```bash
   python scripts/convert_rafdb_to_llamafactory.py
   ```
   在 **Colab 代码单元**里请用 `!` 执行：`!python scripts/convert_rafdb_to_llamafactory.py`
   生成的 `data_list/llamafactory/rafdb_train_emo.json` 中，每条 human 消息的 `value` 已包含 `<image>\n` 和问题文本。

2. **若已有 ShareGPT 格式 JSON 但缺少 `<image>`**，本地可运行：
   ```bash
   python scripts/ensure_llamafactory_image_token.py data_list/llamafactory/rafdb_train_emo.json --fix
   ```

3. **在 Colab 里直接修复当前环境下的 JSON**（不依赖本地脚本，改的是 Colab 磁盘上的文件）：  
   把下面整段贴到 Colab 的一个代码单元里，改掉 `JSON_PATH` 为你在 Colab 上用的训练 JSON 路径（和 `dataset_info` 里 `file_name` 一致），然后运行该单元。运行完后同一路径下的 JSON 会被原地改好，再启动 LLaMA-Factory 训练即可。

```python
import json
import os

JSON_PATH = "data_list/llamafactory/rafdb_train_emo.json"  # 改成你在 Colab 上的训练 JSON 路径

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

fixed = 0
for item in data:
    if "image" not in item or item["image"] is None:
        continue
    n_img = len(item["image"]) if isinstance(item["image"], list) else 1
    convs = item.get("conversations", [])
    for m in convs:
        if isinstance(m, dict) and m.get("from") == "human":
            val = m.get("value", "")
            if isinstance(val, str) and "<image>" not in val:
                m["value"] = "<image>\n" + val
                fixed += 1
            break

with open(JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
print(f"已在 Colab 环境内修复 {fixed} 条，保存到: {JSON_PATH}")
```

4. 保证 `dataset_info` 中的 `file_name` 指向上述 JSON，且该文件内每条样本格式为：`conversations`（含 `from`/`value`）+ `image`（图片路径），且第一条 human 的 `value` 以 `<image>` 开头。

**若出现大量警告并仍报 “image does not match &lt;image&gt; tokens”**：

```text
[WARNING] Media basic/Image/aligned/train_xxxxx_aligned.jpg does not exist in `media_dir`. Use original path.
ValueError: The number of images does not match the number of <image> tokens in [...]
```

说明 LLaMA-Factory 在 `media_dir` 下找不到图片。JSON 里图片路径是**相对 media_dir** 的（如 `basic/Image/aligned/train_07352_aligned.jpg`），完整路径 = `media_dir` + 该路径。

- **本地**：若 RAF-DB 在项目根下（如 `RAF-DB/basic/Image/aligned/`），在训练用的 YAML 里设置 `media_dir: RAF-DB`（或你实际放 RAF-DB 的目录）。
- **Colab**：在 YAML 里把 `media_dir` 改成你解压 RAF-DB 的**根目录**（如 `/content/drive/MyDrive/ExpLLM/ExpLLM-TMM/RAF-DB`）。**不要**用命令行传 `--media_dir`，否则可能报 `HfArgumentParser` 不认识该参数。  
  确认目录下存在 `basic/Image/aligned/` 且里面有 `train_*.jpg`（磁盘上多为 `.jpg`，不是 `_aligned.jpg`）再跑训练。

**Colab 一直报错（WARNING Media ... _aligned.jpg does not exist / image does not match &lt;image&gt; tokens）时**：

1. **根因**：JSON 里写的是 `train_xxxxx_aligned.jpg` 而磁盘上是 `train_xxxxx.jpg`；或 **HuggingFace datasets 缓存**里仍是旧路径，训练时读的是缓存而不是磁盘上的 JSON。
2. **必须带新缓存**：清掉 `~/.cache/huggingface/datasets` 后，训练命令**必须**带 `HF_DATASETS_CACHE=/tmp/hf_datasets_cache_rafdb`（或同目录），否则 `llamafactory-cli` 子进程可能仍用默认缓存里的旧 _aligned.jpg 数据。
3. **一键修复并训练**（二选一）：
   - **推荐**：在 Colab 里运行 `!cd /content/drive/MyDrive/ExpLLM/ExpLLM-TMM && python scripts/colab_fix_and_train.py`。脚本会修 JSON、写 media_dir、清缓存、设 `HF_DATASETS_CACHE`，并在**同一进程**里启动训练，从而保证用新缓存。
   - 若没有 `scripts/colab_fix_and_train.py`，把下面整段贴到一个代码单元运行（改 `PROJECT_ROOT`、`MEDIA_DIR` 为你的路径）；脚本结束会在同一 cell 内自动跑训练（`RUN_TRAIN=True` 时）：
   ```python
   import json, os, shutil, subprocess
   PROJECT_ROOT = "/content/drive/MyDrive/ExpLLM/ExpLLM-TMM"
   MEDIA_DIR = "/content/drive/MyDrive/ExpLLM/ExpLLM-TMM/RAF-DB"
   JSON_PATH = os.path.join(PROJECT_ROOT, "data_list/llamafactory/rafdb_train_emo.json")
   YAML_PATH = os.path.join(PROJECT_ROOT, "llamafactory_config_qwen3vl2b.yaml")
   HF_CACHE_NEW = "/tmp/hf_datasets_cache_rafdb"
   RUN_TRAIN = True
   os.chdir(PROJECT_ROOT)
   # 修 JSON _aligned.jpg -> .jpg
   if os.path.exists(JSON_PATH):
       with open(JSON_PATH, "r", encoding="utf-8") as f: data = json.load(f)
       n = sum(1 for i in data if isinstance(i.get("image"), str) and i["image"].endswith("_aligned.jpg"))
       for i in data:
           if isinstance(i.get("image"), str) and i["image"].endswith("_aligned.jpg"): i["image"] = i["image"][:-12] + ".jpg"
       with open(JSON_PATH, "w", encoding="utf-8") as f: json.dump(data, f, indent=2, ensure_ascii=False)
       print(f"已把 _aligned.jpg 改成 .jpg，共 {n} 条")
   # 写 media_dir（绝对路径）和 preprocessing_num_workers: 1 到 YAML
   if os.path.exists(YAML_PATH):
       with open(YAML_PATH, "r", encoding="utf-8") as f: lines = f.readlines()
       media_abs = os.path.abspath(MEDIA_DIR) if not os.path.isabs(MEDIA_DIR) else MEDIA_DIR
       out = []
       for l in lines:
           if l.strip().startswith("media_dir:"): out.append(f'media_dir: "{media_abs}"\n')
           elif l.strip().startswith("preprocessing_num_workers:"): out.append("preprocessing_num_workers: 1\n")
           else: out.append(l)
       if not any(l.strip().startswith("media_dir:") for l in lines): out.append(f'media_dir: "{media_abs}"\n')
       if not any(l.strip().startswith("preprocessing_num_workers:") for l in lines): out.append("preprocessing_num_workers: 1\n")
       with open(YAML_PATH, "w", encoding="utf-8") as f: f.writelines(out)
       print("已写入 media_dir（绝对路径）和 preprocessing_num_workers: 1 到 YAML")
   # 清默认缓存并强制新缓存
   cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
   if os.path.isdir(cache_dir): shutil.rmtree(cache_dir); print("已删除默认 datasets 缓存")
   os.makedirs(HF_CACHE_NEW, exist_ok=True); os.environ["HF_DATASETS_CACHE"] = HF_CACHE_NEW
   print(f"已设置 HF_DATASETS_CACHE={HF_CACHE_NEW}")
   # 确保 data/dataset_info.json 存在
   src = os.path.join(PROJECT_ROOT, "scripts/llamafactory_dataset_info.json")
   dst = os.path.join(PROJECT_ROOT, "data/dataset_info.json")
   if os.path.exists(src): os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy2(src, dst); print("已复制 dataset_info")
   if RUN_TRAIN:
       subprocess.run(["llamafactory-cli", "train", os.path.basename(YAML_PATH)], cwd=PROJECT_ROOT, env={**os.environ, "HF_DATASETS_CACHE": HF_CACHE_NEW})
   else:
       print(f"下一格运行: !HF_DATASETS_CACHE={HF_CACHE_NEW} llamafactory-cli train {os.path.basename(YAML_PATH)}")
   ```
4. **若 “Converting format” 已 100% 且无 _aligned 警告，但 “Running tokenizer” 报 “The number of images does not match”**：  
   说明多进程下 worker 进程解析 `media_dir` 失败（相对路径 + 不同 cwd），导致图片未加载（0 张）。  
   - **YAML 里**：`media_dir` 必须为**绝对路径**（如 `/content/drive/MyDrive/ExpLLM/ExpLLM-TMM/RAF-DB`）；并把 `preprocessing_num_workers` 设为 **1**，让 tokenizer 在主进程跑，避免 worker 路径问题。  
   - 运行 `scripts/colab_fix_and_train.py` 时会自动写入绝对路径的 media_dir 和 `preprocessing_num_workers: 1`。

5. **从源头避免**：本地或 Colab 重新跑一次 `python scripts/convert_rafdb_to_llamafactory.py`，脚本已改为输出 `train_xxxxx.jpg`，不再写 `_aligned.jpg`。



## Contact me
If you have any questions about this code, feel free to contact me at 3066257338@qq.com.

## Citations
If you find this code useful for your research, please cite our paper:

```
@ARTICLE{lan2025expllm,
  author={Lan, Xing and Xue, Jian and Qi, Ji and Jiang, Dongmei and Lu, Ke and Chua, Tat-Seng},
  journal={IEEE Transactions on Multimedia}, 
  title={ExpLLM: Towards Chain of Thought for Facial Expression Recognition}, 
  year={2025},
  volume={27},
  number={},
  pages={3069-3081},
  doi={10.1109/TMM.2025.3557704}}
```

## Acknowledgement
The code is mainly encouraged by [LocLLM](https://github.com/kennethwdk/LocLLM), [Pink](https://github.com/SY-Xuan/Pink) and [LLaVA](https://github.com/haotian-liu/LLaVA).
