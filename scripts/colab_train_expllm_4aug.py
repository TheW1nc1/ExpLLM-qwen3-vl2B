# ========== ExpLLM 5:1 FER+CoT | Qwen3-VL-2B + LLaMA-Factory ==========
# 数据：每图 5 条 FER + 1 条 CoT，可续训从 checkpoint-385

print(">>> 脚本开始执行 <<<", flush=True)
import os
import sys
import json
import subprocess

print("[Step 0/5] 安装/升级依赖...", flush=True)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U",
    "transformers>=4.45", "peft", "accelerate", "llamafactory", "pillow", "numpy", "torchvision"])
print("   ✅ 依赖安装完成", flush=True)

print("[Step 1/5] 导入 torch、transformers、llamafactory...", flush=True)
try:
    import torch
    from transformers import set_seed
    import llamafactory.data
    print("   ✅ 导入成功", flush=True)
except ImportError as e:
    print(f"⚠️ 导入失败: {e}\n请：Runtime -> Restart runtime，然后重新运行本单元格", flush=True)
    raise

print("[Step 2/5] 注入 Collator（无增强，Qwen3-VL 用官方 Processor）...", flush=True)
def _install_expllm_collator():
    # import random
    # try:
    #     from PIL import Image, ImageOps
    #     import numpy as np
    #     from torchvision import transforms
    # except ImportError:
    #     subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pillow", "numpy", "torchvision"])
    #     from PIL import Image, ImageOps
    #     import numpy as np
    #     from torchvision import transforms

    # def _emo_train_augment(pil_img):
    #     if not hasattr(pil_img, "convert"):
    #         return pil_img
    #     if random.random() < 0.5:
    #         pil_img = ImageOps.mirror(pil_img)
    #     pil_img = transforms.ColorJitter(0.25, 0.25, 0.25, 0.1)(pil_img)
    #     if random.random() < 0.5:
    #         pil_img = pil_img.rotate(random.uniform(-5, 5), resample=Image.BICUBIC, expand=False)
    #     img = np.array(pil_img)
    #     h, w, c = img.shape
    #     area = h * w
    #     if random.random() < 0.5:
    #         for _ in range(10):
    #             erase_area = area * random.uniform(0.05, 0.12)
    #             eh = int(round((erase_area ** 0.5) * random.uniform(0.3, 1 / 0.3)))
    #             ew = int(round((erase_area ** 0.5) * random.uniform(0.3, 1 / 0.3)))
    #             if eh < h and ew < w:
    #                 y, x = random.randint(0, h - eh), random.randint(0, w - ew)
    #                 img[y:y+eh, x:x+ew] = 0
    #                 break
    #     return Image.fromarray(img)

    # def augment_images_in_features(features: list):
    #     for f in features:
    #         imgs = f.get("images") or []
    #         if not imgs:
    #             continue
    #         f["images"] = [_emo_train_augment(img) if hasattr(img, "convert") else img for img in imgs]

    from llamafactory.data.collator import SFTDataCollatorWith4DAttentionMask as _Base
    class AugmentedSFTDataCollatorWith4DAttentionMask(_Base):
        def __call__(self, features: list):
            # 自定义增强已注释：Qwen3-VL 文档建议不修改 Processor 内部 transform，保持预训练分布
            # if getattr(self, "model", None) is not None and self.model.training:
            #     augment_images_in_features(features)
            return super().__call__(features)

    import llamafactory.data as lf_data
    import llamafactory.data.collator as collator_mod
    collator_mod.SFTDataCollatorWith4DAttentionMask = AugmentedSFTDataCollatorWith4DAttentionMask
    lf_data.SFTDataCollatorWith4DAttentionMask = AugmentedSFTDataCollatorWith4DAttentionMask

_install_expllm_collator()
print("   ✅ Collator 注入完成（无自定义增强，仅用 LLaMA-Factory/Processor 预处理）", flush=True)

print("[Step 3/5] 导入 run_exp、设置环境...", flush=True)
from llamafactory.train.tuner import run_exp
os.environ["DISABLE_VERSION_CHECK"] = "1"
set_seed(42)
print("   ✅ 环境就绪", flush=True)

# ========== 参数（文献 + 项目实际，Qwen3-VL-2B 专用） ==========
# 参考文献：LoRA rank collapse、VLM freeze/unfreeze、prompt 影响、FER+VLM 微调
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/content/drive/MyDrive/ExpLLM/ExpLLM-TMM")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "saves/qwen3vl2b/lora/rafdb_expllm_5_1")
MODEL_PATH = "Qwen/Qwen3-VL-2B-Instruct"
MEDIA_DIR = os.environ.get("RAF_DB_ROOT", os.path.join(PROJECT_ROOT, "RAF-DB"))

# 数据：FER_ONLY=True 仅 FER(~12k)；False 则 n×FER+1×CoT 每图 (~49k @5:1)
FER_ONLY = False
FER_COT_RATIO = (5, 1)  # 5:1 FER+CoT，从头 8 轮
RESUME_CHECKPOINT = None  # 从头训练
RESUME_EXTRA_EPOCHS = 3  # 仅续训时生效

# 训练超参（2B 易过拟合，文献建议降 lr、强正则）
# 续训时用更低 lr 避免 overshoot；2e-5 适合从头训，1e-5/5e-6 适合续训
NUM_EPOCHS = 8
LEARNING_RATE = 2.0e-5
RESUME_LEARNING_RATE = 1.0e-5  # 续训专用，None 则沿用 LEARNING_RATE
WEIGHT_DECAY = 0.1
WARMUP_RATIO = 0.06
Per_DEVICE_BATCH = 40
GRAD_ACCUM = 4

# LoRA：文献称 rank 16 对小模型易过拟合，rank 8 有隐式正则（2405.09673, rank collapse）
# 推荐 alpha=32（q/v 对注意力影响最大，alpha 略高利于适配）
LORA_RANK = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
# lora_target：qv 参数量少；加 o_proj 可增容量但易过拟合。2B 先 qv。
LORA_TARGET = "q_proj,v_proj"  # 可选 "q_proj,v_proj,o_proj" 若 qv 不够

# 冻结：freeze_vision_tower=False 允许视觉编码器训练；仅 LoRA(q_proj,v_proj)，与 PeftModel 兼容
FREEZE_VISION = False
FREEZE_PROJECTOR = False

# prompt：文献称 fine-tuning 后 prompt 影响较小，但 train-eval 必须完全一致
# 当前显式选项 "Answer with exactly one word from: ..." 利于减少 dismatch

print("\n" + "=" * 60, flush=True)
print("ExpLLM 4aug | Qwen3-VL-2B 文献推荐配置", flush=True)
print("=" * 60, flush=True)
print(f"   OUTPUT_DIR:   {OUTPUT_DIR}", flush=True)
print(f"   FER_ONLY:     {FER_ONLY} | freeze_vision: {FREEZE_VISION}", flush=True)
print(f"   epochs:       {NUM_EPOCHS} | lr: {LEARNING_RATE} | 续训lr: {RESUME_LEARNING_RATE}", flush=True)
print(f"   LoRA:         {LORA_TARGET} rank={LORA_RANK} alpha={LORA_ALPHA}", flush=True)
print(f"   weight_decay: {WEIGHT_DECAY} | warmup: {WARMUP_RATIO}", flush=True)

# ========== 1. 数据准备 ==========
print("\n[Step 4/5] 数据准备...", flush=True)
input_json = os.path.join(PROJECT_ROOT, "data_list/anno_json/rafdb_emo_au_train_mini_description.json")
output_json = os.path.join(DATA_DIR, "rafdb_expllm_4aug.json")
if not os.path.exists(input_json):
    raise FileNotFoundError(f"需要 {input_json}")
print(f"   输入: {input_json}", flush=True)

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
    if FER_ONLY:
        out_list.append(conv_fer)
    else:
        n_fer, n_cot = FER_COT_RATIO
        for _ in range(n_fer):
            out_list.append(conv_fer)
        for _ in range(n_cot):
            out_list.append(conv_cot)

os.makedirs(DATA_DIR, exist_ok=True)
with open(output_json, "w") as f:
    json.dump(out_list, f, indent=2, ensure_ascii=False)
print(f"   ✅ 已写入 {output_json}，共 {len(out_list)} 条", flush=True)

info_path = os.path.join(DATA_DIR, "dataset_info.json")
info = json.load(open(info_path)) if os.path.exists(info_path) else {}
info["rafdb_expllm_4aug"] = {
    "file_name": "rafdb_expllm_4aug.json",
    "formatting": "sharegpt",
    "columns": {"messages": "conversations", "images": "image"},
    "tags": {"role_tag": "from", "content_tag": "value", "user_tag": "human", "assistant_tag": "gpt"},
}
with open(info_path, "w") as f:
    json.dump(info, f, indent=2, ensure_ascii=False)
print("   ✅ dataset_info 已更新", flush=True)

DATASET_NAME = "rafdb_expllm_4aug"

# ========== 2. 训练 ==========
print("\n[Step 5/5] 开始训练...", flush=True)
# tokenized 缓存：仅依赖数据集内容；改 lr 不改数据可复用；12k→49k 必须清空重算
TOKENIZED_PATH = "/content/tokenized_rafdb_expllm"
OVERWRITE_CACHE = True  # 5:1 数据量变，需重新 tokenize
if OVERWRITE_CACHE and os.path.isdir(TOKENIZED_PATH):
    import shutil
    shutil.rmtree(TOKENIZED_PATH)
    print(f"   已删除旧 tokenized 缓存，将重新 tokenize", flush=True)
resume_enabled = False
resume_path = None
if RESUME_CHECKPOINT:
    resume_path = os.path.join(OUTPUT_DIR, RESUME_CHECKPOINT)
    resume_enabled = os.path.isdir(resume_path)
    if not resume_enabled:
        raise FileNotFoundError(f"未找到 checkpoint：{resume_path}\n设 RESUME_CHECKPOINT=None 从头训练。")
    # 方案 B：续训可用 FER_ONLY=False（3:1 FER+CoT），切换数据需清 tokenized 缓存
    if not FER_ONLY and os.path.isdir(TOKENIZED_PATH):
        import shutil
        shutil.rmtree(TOKENIZED_PATH)
        print(f"   方案 B：续训加 CoT(3:1)，已清 tokenized 缓存，将重新 tokenize", flush=True)
    # 续训 5+3=8：原 385 步=5 epoch，再训 3 epoch
    # 12k(FER_ONLY) 约 77 步/epoch；49k(FER+CoT) 约 307 步/epoch
    steps_per_epoch = max(1, len(out_list) // (Per_DEVICE_BATCH * GRAD_ACCUM))
    resume_max_steps = 385 + RESUME_EXTRA_EPOCHS * steps_per_epoch
    eff_lr = RESUME_LEARNING_RATE if RESUME_LEARNING_RATE is not None else LEARNING_RATE
    print(f"   ✅ 续训: 从 {RESUME_CHECKPOINT}，再训 {RESUME_EXTRA_EPOCHS} epoch（约 {resume_max_steps} 步，lr={eff_lr}）", flush=True)
else:
    print(f"   全新训练: {NUM_EPOCHS} epochs", flush=True)
    resume_max_steps = None
os.makedirs(OUTPUT_DIR, exist_ok=True)

args = {
    "model_name_or_path": MODEL_PATH,
    "stage": "sft",
    "do_train": True,
    "finetuning_type": "lora",
    "lora_target": LORA_TARGET,
    "lora_rank": LORA_RANK,
    "lora_alpha": LORA_ALPHA,
    "lora_dropout": LORA_DROPOUT,
    "freeze_vision_tower": FREEZE_VISION,
    "freeze_multi_modal_projector": FREEZE_PROJECTOR,
    "dataset": DATASET_NAME,
    "dataset_dir": DATA_DIR,
    "template": "qwen3_vl_nothink",  # 与原作者 Vicuna 一致：ASSISTANT 直接输出，无 think 块（见 datasets/convsersation.py）
    "media_dir": MEDIA_DIR,
    "cutoff_len": 2048,
    "overwrite_cache": OVERWRITE_CACHE,
    "tokenized_path": "/content/tokenized_rafdb_expllm",  # Colab 本地缓存，同会话内复用
    "preprocessing_num_workers": 8,  # T4 上可用，若卡在 Running tokenizer 则改为 1
    "output_dir": OUTPUT_DIR,
    "logging_steps": 50,
    "save_steps": 100,       # 每 100 步保存，总步 385 时生成 checkpoint-100,200,300,385；若要 150 则改为 50
    "save_total_limit": 4,   # 最多保留 4 个 checkpoint，超出时删除最旧的
    "overwrite_output_dir": not resume_enabled,
    "per_device_train_batch_size": Per_DEVICE_BATCH,
    "gradient_accumulation_steps": GRAD_ACCUM,
    "learning_rate": LEARNING_RATE,
    "num_train_epochs": NUM_EPOCHS,  # 续训时也为目标总 epoch（如 8），trainer 会从 checkpoint 继续到该值
    "lr_scheduler_type": "cosine",
    "warmup_ratio": WARMUP_RATIO,
    "weight_decay": WEIGHT_DECAY,
    "bf16": True,
    "tf32": True,
    "gradient_checkpointing": True,
    "dataloader_num_workers": 8,
    "report_to": "tensorboard",
}
if resume_enabled:
    args["resume_from_checkpoint"] = resume_path
    if RESUME_LEARNING_RATE is not None:
        args["learning_rate"] = RESUME_LEARNING_RATE
    if resume_max_steps is not None:
        args["max_steps"] = resume_max_steps  # 限制为 5+3=8，避免 trainer 按 8 全 epochs 重算
        args.pop("num_train_epochs", None)    # max_steps 与 num_train_epochs 二选一

print("-" * 60, flush=True)
run_exp(args)
print("-" * 60, flush=True)
print("\n✅ 训练完成！输出目录:", OUTPUT_DIR, flush=True)
print("   测评时 MODEL_DIR 设为:", OUTPUT_DIR, flush=True)
