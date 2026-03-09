# ========== ExpLLM 源码对齐：Qwen3-VL-2B + LLaMA-Factory ==========
# 按原项目源码 datasets/facetask.py emo_train_transforms 做 on-the-fly 增强
# 训练超参与 train_rafdb.sh 一致：10 epochs，batch 80×2=160

print(">>> 脚本开始执行 <<<", flush=True)
import os
import sys
import json
import subprocess

print("[Step 0/5] 安装/升级依赖（transformers, peft, llamafactory 等）...", flush=True)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U",
    "transformers>=4.45", "peft", "accelerate", "llamafactory", "pillow", "numpy", "torchvision"])
print("   ✅ 依赖安装完成", flush=True)

print("[Step 1/5] 导入 torch、transformers、llamafactory...", flush=True)
try:
    import torch
    from transformers import set_seed
    import llamafactory.data  # 先加载 data 模块
    print("   ✅ 导入成功", flush=True)
except ImportError as e:
    print(f"⚠️ 导入失败: {e}\n请：Runtime -> Restart runtime，然后重新运行本单元格", flush=True)
    raise

print("[Step 2/5] 注入自定义 Collator（emo_train_transforms 全量 on-the-fly）...", flush=True)
def _install_expllm_collator():
    import random
    try:
        from PIL import Image, ImageOps, ImageEnhance, ImageFilter
        import numpy as np
        from torchvision import transforms
        from torchvision.transforms import functional as F
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pillow", "numpy", "torchvision"])
        from PIL import Image, ImageOps, ImageEnhance, ImageFilter
        import numpy as np
        from torchvision import transforms
        from torchvision.transforms import functional as F

    # 源码 facetask.py emo_train_transforms 全量复现（输出 PIL，不含 ToTensor/Normalize）
    def _emo_train_augment(pil_img):
        if not hasattr(pil_img, "convert"):
            return pil_img
        # 1. RandomHorizontalFlip (p=0.5)
        if random.random() < 0.5:
            pil_img = ImageOps.mirror(pil_img)
        # 2. RandomChoice: ColorJitter OR RandomGrayscale(p=0.2)
        if random.random() < 0.5:
            pil_img = transforms.ColorJitter(0.25, 0.25, 0.25, 0.1)(pil_img)
        else:
            pil_img = transforms.RandomGrayscale(p=0.2)(pil_img)
        # 3. RandomApply(p=0.5): RandomRotation(5) + RandomChoice(RandomResizedCrop OR RandomCrop)
        if random.random() < 0.5:
            pil_img = pil_img.rotate(random.uniform(-5, 5), resample=Image.BICUBIC, expand=False)
            w, h = pil_img.size
            if random.random() < 0.5:
                # RandomResizedCrop(224, scale=(0.8,1), ratio=(0.75,1.3333))
                rrc = transforms.RandomResizedCrop(224, scale=(0.8, 1), ratio=(0.75, 1.3333))
                i, j, ch, cw = rrc.get_params(pil_img, rrc.scale, rrc.ratio)
                pil_img = F.resized_crop(pil_img, i, j, ch, cw, (224, 224))
            else:
                # RandomCrop(224, padding=12)：先 pad 再 crop
                if min(w, h) < 224:
                    pil_img = pil_img.resize((224, 224), Image.BICUBIC)
                pil_img = F.pad(pil_img, 12, fill=0)
                i, j, _, _ = transforms.RandomCrop(224).get_params(pil_img, (224, 224))
                pil_img = F.crop(pil_img, i, j, 224, 224)
        # 4. RandomApply(p=0.5): RandomAffine + GaussianBlur
        if random.random() < 0.5:
            w, h = pil_img.size
            angle = random.uniform(-15, 15)
            tx = random.uniform(-0.1, 0.1) * w
            ty = random.uniform(-0.1, 0.1) * h
            scale = random.uniform(0.9, 1.1)
            pil_img = F.affine(pil_img, angle=angle, translate=(tx, ty), scale=scale, shear=0, fill=0)
            sigma = random.uniform(0.1, 2.0)
            try:
                pil_img = F.gaussian_blur(pil_img, kernel_size=[5, 5], sigma=[sigma, sigma])
            except Exception:
                pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=max(1, min(2, int(sigma)))))
        # 5. RandomErasing(scale=(0.05, 0.12))
        img = np.array(pil_img)
        h, w, c = img.shape
        area = h * w
        for _ in range(10):
            erase_area = area * random.uniform(0.05, 0.12)
            eh = int(round((erase_area ** 0.5) * random.uniform(0.3, 1 / 0.3)))
            ew = int(round((erase_area ** 0.5) * random.uniform(0.3, 1 / 0.3)))
            if eh < h and ew < w:
                y, x = random.randint(0, h - eh), random.randint(0, w - ew)
                fill = np.array([int(random.uniform(0, 255)) for _ in range(c)], dtype=img.dtype)
                img[y:y+eh, x:x+ew] = fill
                return Image.fromarray(img)
        return Image.fromarray(img)

    def augment_images_in_features(features: list):
        for f in features:
            imgs = f.get("images") or []
            if not imgs:
                continue
            f["images"] = [_emo_train_augment(img) if hasattr(img, "convert") else img for img in imgs]

    from llamafactory.data.collator import SFTDataCollatorWith4DAttentionMask as _Base
    class AugmentedSFTDataCollatorWith4DAttentionMask(_Base):
        def __call__(self, features: list):
            if getattr(self, "model", None) is not None and self.model.training:
                augment_images_in_features(features)
            return super().__call__(features)

    import llamafactory.data as lf_data
    import llamafactory.data.collator as collator_mod
    collator_mod.SFTDataCollatorWith4DAttentionMask = AugmentedSFTDataCollatorWith4DAttentionMask
    lf_data.SFTDataCollatorWith4DAttentionMask = AugmentedSFTDataCollatorWith4DAttentionMask

_install_expllm_collator()
print("   ✅ Collator 注入完成", flush=True)

print("[Step 3/5] 导入 run_exp、设置环境...", flush=True)
from llamafactory.train.tuner import run_exp
os.environ["DISABLE_VERSION_CHECK"] = "1"
set_seed(42)
print("   ✅ 环境就绪", flush=True)

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/content/drive/MyDrive/ExpLLM/ExpLLM-TMM")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "saves/qwen3vl2b/lora/rafdb_expllm_paper")
MODEL_PATH = "Qwen/Qwen3-VL-2B-Instruct"
MEDIA_DIR = os.environ.get("RAF_DB_ROOT", os.path.join(PROJECT_ROOT, "RAF-DB"))
# 续训：从 checkpoint-3070 继续训 5 个 epoch（方案 A）
RESUME_CHECKPOINT = "checkpoint-3070"
RESUME_EXTRA_EPOCHS = 5

print("\n" + "=" * 60, flush=True)
print("ExpLLM 源码对齐：Qwen3-VL-2B + LLaMA-Factory", flush=True)
print("=" * 60, flush=True)
print(f"   PROJECT_ROOT: {PROJECT_ROOT}", flush=True)
print(f"   DATA_DIR:     {DATA_DIR}", flush=True)
print(f"   OUTPUT_DIR:   {OUTPUT_DIR}", flush=True)
print(f"   MEDIA_DIR:    {MEDIA_DIR}", flush=True)
print(f"   MODEL:        {MODEL_PATH}", flush=True)

# ========== 1. 数据准备 ==========
print("\n[Step 4/5] 数据准备...", flush=True)
input_json = os.path.join(PROJECT_ROOT, "data_list/anno_json/rafdb_emo_au_train_mini_description.json")
output_json = os.path.join(DATA_DIR, "rafdb_expllm_paper.json")
print(f"   读取输入: {input_json}", flush=True)
if not os.path.exists(input_json):
    raise FileNotFoundError(f"需要 {input_json}，请先准备 RAF-DB 标注")
print(f"   ✅ 输入文件存在", flush=True)

# 论文消融最优：FER:CoT = 0.75:0.25（91.03% FER，0.78 Exp-CoT Score）
# 实现：每样本 3 条 FER + 1 条 CoT → 3/4=0.75, 1/4=0.25
print("   生成 FER + CoT 数据（论文 0.75:0.25）...", flush=True)
INSTR_FER = "<image>\nWhat is the expression of the person in the image? Answer with exactly one word from: Neutral, Happiness, Sadness, Surprise, Fear, Disgust, Anger."
INSTR_COT = "<image>\nProvide a step-by-step analysis of the facial expression in this image. Include key observations, overall emotional interpretation, and conclusion."

def _norm(f):
    return f[:-12] + ".jpg" if isinstance(f, str) and f.endswith("_aligned.jpg") else f

with open(input_json, "r") as f:
    raw = json.load(f)
out_list = []
for item in raw:
    fname = _norm(item.get("file_name", ""))
    img_path = os.path.join("basic/Image/aligned", fname).replace("\\", "/")
    emo, desc = item.get("emo", "Neutral"), item.get("description", "").strip()
    conv_fer = {"conversations": [{"from": "human", "value": INSTR_FER}, {"from": "gpt", "value": emo}], "image": img_path}
    conv_cot = {"conversations": [{"from": "human", "value": INSTR_COT}, {"from": "gpt", "value": desc}], "image": img_path}
    for _ in range(3):
        out_list.append(conv_fer)
    out_list.append(conv_cot)
os.makedirs(DATA_DIR, exist_ok=True)
with open(output_json, "w") as f:
    json.dump(out_list, f, indent=2, ensure_ascii=False)
print(f"   ✅ 已写入 {output_json}，共 {len(out_list)} 条 (FER:{len(raw)*3} CoT:{len(raw)})", flush=True)

# 更新 dataset_info
print("   更新 dataset_info.json...", flush=True)
info_path = os.path.join(DATA_DIR, "dataset_info.json")
info = json.load(open(info_path)) if os.path.exists(info_path) else {}
info["rafdb_expllm_paper"] = {
    "file_name": "rafdb_expllm_paper.json",
    "formatting": "sharegpt",
    "columns": {"messages": "conversations", "images": "image"},
    "tags": {"role_tag": "from", "content_tag": "value", "user_tag": "human", "assistant_tag": "gpt"},
}
os.makedirs(DATA_DIR, exist_ok=True)
with open(info_path, "w") as f:
    json.dump(info, f, indent=2, ensure_ascii=False)
print("   ✅ dataset_info 已更新", flush=True)

DATASET_NAME = "rafdb_expllm_paper"
print(f"   数据增强：Collator on-the-fly（源码 facetask.py emo_train_transforms 全量）", flush=True)
print("   ✅ 数据准备完成", flush=True)

# ========== 2. 训练 ==========
print("\n[Step 5/5] 开始训练...", flush=True)
resume_path = os.path.join(OUTPUT_DIR, RESUME_CHECKPOINT)
resume_enabled = os.path.isdir(resume_path)
if resume_enabled:
    print(f"   续训: 从 {RESUME_CHECKPOINT} 继续，再训 {RESUME_EXTRA_EPOCHS} 个 epoch", flush=True)
    print(f"   配置: LoRA rank 8（与 checkpoint 一致）, lr 5e-4, batch 40×4=160", flush=True)
else:
    raise FileNotFoundError(f"未找到 checkpoint：{resume_path}\n请确认 Drive 已挂载且 checkpoint-3070 存在；或修改 RESUME_CHECKPOINT 为实际路径。")
print(f"   输出目录: {OUTPUT_DIR}", flush=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

args = {
    "model_name_or_path": MODEL_PATH,
    "stage": "sft",
    "do_train": True,
    "finetuning_type": "lora",
    "lora_target": "q_proj,v_proj",
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "freeze_vision_tower": False,
    "freeze_multi_modal_projector": False,
    "dataset": DATASET_NAME,
    "dataset_dir": DATA_DIR,
    "template": "qwen3_vl_nothink",
    "media_dir": MEDIA_DIR,
    "cutoff_len": 2048,
    "overwrite_cache": True,
    "preprocessing_num_workers": 8,
    "output_dir": OUTPUT_DIR,
    "logging_steps": 50,
    "save_steps": 100,
    "save_total_limit": 2,
    "overwrite_output_dir": not resume_enabled,
    "per_device_train_batch_size": 40,
    "gradient_accumulation_steps": 4,
    "learning_rate": 5.0e-4,
    "num_train_epochs": 10 + RESUME_EXTRA_EPOCHS if resume_enabled else 10,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.03,
    "weight_decay": 0.05,
    "bf16": True,
    "tf32": True,
    "gradient_checkpointing": True,
    "dataloader_num_workers": 8,
    "report_to": "tensorboard",
}
if resume_enabled:
    args["resume_from_checkpoint"] = resume_path

print("   调用 run_exp()，LLaMA-Factory 将加载模型、构建数据集、开始训练...", flush=True)
print("-" * 60, flush=True)
run_exp(args)
print("-" * 60, flush=True)
print("\n✅ 训练完成！输出目录:", OUTPUT_DIR, flush=True)
