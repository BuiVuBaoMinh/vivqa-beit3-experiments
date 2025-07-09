import torch
from dotenv import load_dotenv
import os
import sys
import json
from tqdm import tqdm

from transformers import (
    AutoTokenizer, AutoModelForImageTextToText, AutoProcessor, AutoModelForVision2Seq,
    PhobertTokenizer,
)

from huggingface_hub import login

from smolvlm_dataset import ViVQASmolVLMDataset, create_smolvlm_dataset_by_split
from mySmolVLM import get_vivqa_smolvlm, get_vivqa_smolvlm_phobert

import torch
print(torch.cuda.get_device_capability())

sys.exit()

class TempArgs(object):
    def __init__(self):
        self.batch_size= 8
        self.eval_batch_size= 8
        self.epochs= 50
        self.layer_decay= 0.99
        self.update_freq= 1
        self.warmup_epochs= 0
        self.data_path= "/home/21khac.dd/bm/data/vivqa"
        self.output_dir= "/home/21khac.dd/bm/blip-pb-5"
        self.num_workers=10
        self.weight_decay= 0.05
        self.save_ckpt_freq= 1
        self.task_head_lr_weight= 0
        self.opt_betas = (0.9, 0.98)
        self.early_stopping= "val_score"
        self.patience= 6
        self.lr_sched_type= "cos"
        self.answer2label ="/home/21khac.dd/bm/data/vivqa/annotations/dicts/answer2label_en_gemini_translated.txt"
        self.device = "cuda:5"
        self.phobert= True
        self.pin_mem = True
        self.dist_eval = False

args = TempArgs()

phobert_tokenizer = PhobertTokenizer.from_pretrained("vinai/phobert-base-v2", use_fast=True)

dataset_train, dataloader_train = create_smolvlm_dataset_by_split(args, split="train", phobert_tokenizer=phobert_tokenizer)
dataset_test, dataloader_test = create_smolvlm_dataset_by_split(args, split="test", phobert_tokenizer=phobert_tokenizer)
dataset_val, dataloader_val = create_smolvlm_dataset_by_split(args, split="val", phobert_tokenizer=phobert_tokenizer)

max_len_id = 0
lengths = []
over_512_count = 0
total_count = 0

# The check_dataset function has been updated
def check_dataset(dataset, dataloader, name):
    global max_len_id, over_512_count, total_count
    print(f"\nChecking {name} ({len(dataset)} samples):")
    for batch in tqdm(dataloader, desc=f"Checking {name}", leave=False):
        # 💡 CORRECTED: Use .shape[1] to get the sequence length
        input_len = batch["input_ids"].shape[1]
        
        lengths.append({
            # Note: qid is a list of strings, so we can't save just one
            "qids_in_batch": batch["qid"],
            "length": input_len,
            "split": name
        })
        max_len_id = max(max_len_id, input_len)
        
        # We check the length of each sample in the batch now
        for qid, single_input_ids in zip(batch["qid"], batch["input_ids"]):
             total_count += 1
             # Note: This check is against the PADDED length of the batch
             if input_len > 512:
                 over_512_count += 1
    # This print was moved outside the loop to only show the final result
    print(f"Current max : {max_len_id}")

check_dataset(dataset_train, dataloader_train, "train")
check_dataset(dataset_val, dataloader_val, "val")
check_dataset(dataset_test, dataloader_test, "test")

print(f"\n📊 Max input_ids length: {max_len_id}")
print(f"📏 Examples >512 tokens: {over_512_count}/{total_count} ({over_512_count / total_count * 100:.2f}%)")

# Save lengths to file
output_file = "input_id_lengths.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(lengths, f, ensure_ascii=False, indent=2)
print(f"✅ Saved input lengths to: {output_file}")

sys.exit(0)

device = torch.device("cuda:5")

# processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-500M-Instruct", cache_dir="/home/21khac.dd/bm/my-cache-dir")
# model = AutoModelForImageTextToText.from_pretrained(
#     "HuggingFaceTB/SmolVLM-500M-Instruct", 
#     torch_dtype=torch.bfloat16, 
#     cache_dir="/home/21khac.dd/bm/my-cache-dir",
#     _attn_implementation="flash_attention_2" if device == "cuda" else "eager"
# ).to(device, non_blocking=True)

# processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-2.2B-Instruct", cache_dir="/home/21khac.dd/bm/my-cache-dir")
# model = AutoModelForImageTextToText.from_pretrained(
#     "HuggingFaceTB/SmolVLM2-2.2B-Instruct", 
#     torch_dtype=torch.bfloat16, 
#     cache_dir="/home/21khac.dd/bm/my-cache-dir",
#     _attn_implementation="flash_attention_2" if device == "cuda" else "eager"
# ).to(device, non_blocking=True)

# print(type(processor))
# print(type(model))
# print(type(model.model.text_model))

# model1 = get_vivqa_smolvlm(
#     device=device
# ).to(device, non_blocking=True, dtype=torch.bfloat16)

# model1 = get_vivqa_smolvlm_phobert(
#     device=device
# ).to(device, non_blocking=True, dtype=torch.bfloat16)

# sys.exit(0)