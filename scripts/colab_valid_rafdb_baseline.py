# ========== 复制整段到 Colab 一个单元格运行：RAF-DB FER 测评（纯基座，无 LoRA） ==========
# 与 colab_valid_rafdb_4aug.py 完全一致，仅不加载 LoRA，用于对比基座 Qwen3-VL-2B 的零样本准确率
# 同样使用 enable_thinking=False（与训练 qwen3_vl_nothink 一致），保证基座与 LoRA 评测条件相同

import os
import sys
import re
import json
import subprocess

print(">>> RAF-DB FER 测评脚本开始（纯基座 Qwen3-VL-2B，无训练） <<<", flush=True)

print("[Step 0/4] 安装依赖...", flush=True)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U",
                      "Pillow>=10.2.0", "torchvision", "transformers>=4.45", "peft", "accelerate", "numpy"])
print("   依赖就绪", flush=True)

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/content/drive/MyDrive/ExpLLM/ExpLLM-TMM")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(PROJECT_ROOT, "eval_rafdb_baseline"))
QUESTION_FILE = os.environ.get("QUESTION_FILE", os.path.join(PROJECT_ROOT, "data_list/test/raf-db.txt"))
IMAGE_FOLDER = os.environ.get("IMAGE_FOLDER", None)
BATCH_SIZE = 96
MAX_NEW_TOKENS = 100

os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

print("[Step 1/4] 配置...", flush=True)
print(f"   PROJECT_ROOT: {PROJECT_ROOT}", flush=True)
print(f"   OUTPUT_DIR: {OUTPUT_DIR}", flush=True)
print(f"   QUESTION_FILE: {QUESTION_FILE}", flush=True)
print(f"   BATCH_SIZE: {BATCH_SIZE}", flush=True)
print(f"   模型: Qwen3-VL-2B-Instruct（纯基座，无 LoRA）", flush=True)
print(f"   enable_thinking=False（与 qwen3_vl_nothink 一致）", flush=True)

if QUESTION_FILE.endswith(".json") and not IMAGE_FOLDER:
    IMAGE_FOLDER = os.path.join(PROJECT_ROOT, "RAF-DB/basic/Image/aligned")
if not os.path.isabs(QUESTION_FILE):
    QUESTION_FILE = os.path.normpath(os.path.join(PROJECT_ROOT, QUESTION_FILE))
if not os.path.isabs(OUTPUT_DIR):
    OUTPUT_DIR = os.path.normpath(os.path.join(PROJECT_ROOT, OUTPUT_DIR))

import torch
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor
from torch.utils.data import Dataset, DataLoader
from transformers import Qwen3VLForConditionalGeneration

FaceTaskDescription_emo = "Expression recognition involves identifying the emotional state of a person in a given image."
FaceTaskQuestion_emo = "What is the expression of the person in the image? Answer with exactly one word from: Neutral, Happiness, Sadness, Surprise, Fear, Disgust, Anger."
EMO_LABELS_RAF_DB_7 = ["Neutral", "Happiness", "Sadness", "Surprise", "Fear", "Disgust", "Anger"]


def face_task_anno_read(data_path):
    data_path = os.path.normpath(data_path)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"data path not exists: {data_path}")
    if data_path.endswith(".txt"):
        with open(data_path, "r", encoding="utf-8") as f:
            file_list = [x.strip() for x in f.readlines() if x.strip()]
        json_list = []
        for line in file_list:
            parts = line.split(",")
            json_file = parts[0].strip()
            image_folder = parts[1].strip()
            if not os.path.isabs(json_file):
                json_file = os.path.normpath(os.path.join(PROJECT_ROOT, json_file))
            if not os.path.isabs(image_folder):
                image_folder = os.path.normpath(os.path.join(PROJECT_ROOT, image_folder))
            with open(json_file, "r", encoding="utf-8") as jf:
                json_data = json.load(jf)
            for data in json_data:
                d = dict(data)
                d["file_name"] = os.path.normpath(os.path.join(image_folder, d.get("file_name", "")))
                json_list.append(d)
        return json_list
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


class RAFDBEvalDataset(Dataset):
    def __init__(self, question_file, image_folder=None):
        self.image_folder = image_folder
        if question_file.endswith(".txt"):
            self.data = face_task_anno_read(question_file)
        else:
            raw = json.load(open(question_file, "r", encoding="utf-8"))
            if image_folder:
                base = image_folder if os.path.isabs(image_folder) else os.path.join(PROJECT_ROOT, image_folder)
                self.data = []
                for item in raw:
                    d = dict(item)
                    d["file_name"] = os.path.normpath(os.path.join(base, item.get("file_name", "")))
                    self.data.append(d)
            else:
                self.data = raw

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        fpath = item["file_name"]
        orig_fname = item.get("file_name", "")
        if isinstance(orig_fname, dict):
            orig_fname = orig_fname.get("value", orig_fname.get("name", str(orig_fname)))
        if not os.path.exists(fpath) and self.image_folder:
            fpath = os.path.join(self.image_folder, orig_fname)
        if not os.path.exists(fpath):
            match = re.search(r"test_(\d+)_?aligned?\.?jpe?g", str(orig_fname), re.I)
            if match:
                num = int(match.group(1))
                base = os.path.dirname(fpath) if os.path.dirname(fpath) else (self.image_folder or ".")
                for fmt in [f"test_{num:05d}.jpg", f"test_{num:04d}.jpg", f"test_{num}_aligned.jpg"]:
                    p = os.path.join(base, fmt)
                    if os.path.exists(p):
                        fpath = p
                        break
        return {"image_path": fpath, "label": item.get("emo", "Neutral"), "file_name": orig_fname}


_EMO_ALIASES = {"happy": "Happiness", "sad": "Sadness", "angry": "Anger", "neutral": "Neutral",
                "surprise": "Surprise", "surprised": "Surprise", "fear": "Fear", "fearful": "Fear",
                "disgust": "Disgust", "disgusted": "Disgust", "anger": "Anger", "happiness": "Happiness"}


def _extract_pred_label(pred_text, emo_labels):
    pred_text = (pred_text or "").strip()
    if "</think>" in pred_text:
        pred_text = pred_text.split("</think>")[-1].strip()
    pred_text = pred_text.strip(".,!?;:")
    pred_lower = pred_text.lower()
    for emo in emo_labels:
        if emo.lower() == pred_lower:
            return emo, pred_text
    for emo in emo_labels:
        if pred_lower.startswith(emo.lower()) or pred_lower.endswith(emo.lower()) or f" {emo.lower()} " in f" {pred_lower} ":
            return emo, pred_text
    if pred_lower in _EMO_ALIASES:
        return _EMO_ALIASES[pred_lower], pred_text
    first = pred_text.split()[0] if pred_text.split() else ""
    if first:
        fc = first.capitalize()
        if fc in emo_labels:
            return fc, pred_text
        if first.lower() in _EMO_ALIASES:
            return _EMO_ALIASES[first.lower()], pred_text
        for emo in emo_labels:
            if emo.lower() in first.lower():
                return emo, pred_text
    return None, pred_text


def _collate_batch_inputs(single_inputs_list, pad_token_id):
    if not single_inputs_list:
        return None, []
    B = len(single_inputs_list)
    max_len = max(inp["input_ids"].shape[1] for inp in single_inputs_list)
    ref = single_inputs_list[0]["input_ids"]
    device, dtype = ref.device, ref.dtype
    input_ids = torch.full((B, max_len), pad_token_id, dtype=dtype, device=device)
    attention_mask = torch.zeros((B, max_len), dtype=torch.long, device=device)
    input_lens = []
    for i, inp in enumerate(single_inputs_list):
        seq = inp["input_ids"].squeeze(0)
        L = seq.shape[0]
        input_lens.append(L)
        input_ids[i, max_len - L:] = seq
        attention_mask[i, max_len - L:] = inp["attention_mask"].squeeze(0)
    batched = {"input_ids": input_ids, "attention_mask": attention_mask}
    if "pixel_values" in single_inputs_list[0] and single_inputs_list[0]["pixel_values"] is not None:
        batched["pixel_values"] = torch.stack([inp["pixel_values"].squeeze(0) for inp in single_inputs_list], dim=0)
        if "image_grid_thw" in single_inputs_list[0] and single_inputs_list[0]["image_grid_thw"] is not None:
            try:
                batched["image_grid_thw"] = torch.stack([inp["image_grid_thw"].squeeze(0) for inp in single_inputs_list], dim=0)
            except Exception:
                pass
    return batched, input_lens


print("\n[Step 2/4] 加载模型...", flush=True)
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct", trust_remote_code=True)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-2B-Instruct", torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
model.eval()

print("\n[Step 3/4] 加载数据并推理...", flush=True)
sys_desc = FaceTaskDescription_emo
user_text = FaceTaskQuestion_emo
system_text = f"You are a helpful face analysis assistant.\n{sys_desc}"

dataset = RAFDBEvalDataset(QUESTION_FILE, IMAGE_FOLDER)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
n_samples = len(dataset)
n_batches = len(dataloader)
print(f"   测试集: {n_samples} 样本, {n_batches} 批 (batch_size={BATCH_SIZE})", flush=True)

emo_labels = list(EMO_LABELS_RAF_DB_7)
pad_token_id = processor.tokenizer.eos_token_id

all_results = []
for batch in tqdm(dataloader, desc="   推理中", total=n_batches):
    image_paths = batch["image_path"]
    labels = batch["label"]
    file_names = batch["file_name"]

    images = []
    for p in image_paths:
        try:
            images.append(Image.open(p).convert("RGB"))
        except Exception:
            images.append(Image.new("RGB", (224, 224), (128, 128, 128)))

    single_inputs = []
    for img in images:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_text}]},
            {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": user_text}]},
        ]
        try:
            inp = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
                enable_thinking=False,
            )
        except TypeError:
            inp = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
            )
        inp.pop("token_type_ids", None)
        single_inputs.append(inp)

    batched, input_lens = _collate_batch_inputs(single_inputs, pad_token_id)
    batched = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batched.items()}

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model.generate(
            **batched,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=pad_token_id,
            eos_token_id=pad_token_id,
        )

    for i, (L, label, fname) in enumerate(zip(input_lens, labels, file_names)):
        pred_raw = processor.tokenizer.decode(out[i, L:], skip_special_tokens=True).strip()
        pred_label, _ = _extract_pred_label(pred_raw, emo_labels)
        all_results.append({
            "task": "emo_estim",
            "gt": label,
            "pred": pred_label if pred_label is not None else pred_raw,
            "img_path": str(file_names[i]),
        })

print("\n[Step 4/4] 保存结果...", flush=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
out_path = os.path.join(OUTPUT_DIR, "result_all.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

acc = 0
dismatch = 0
emo_id_dict = {e: i for i, e in enumerate(emo_labels)}
for r in all_results:
    if r["pred"] not in emo_labels:
        dismatch += 1
        continue
    if emo_id_dict[r["gt"]] == emo_id_dict[r["pred"]]:
        acc += 1
total = len(all_results)
exp_acc = acc / total if total else 0

print(f"\n   Exp dismatch number: {dismatch}")
print(f"   Exp Accuracy (纯基座): {exp_acc:.4f}")
print(f"   结果已保存: {out_path}")
print("\n>>> 测评完成（纯基座，无 LoRA） <<<", flush=True)
