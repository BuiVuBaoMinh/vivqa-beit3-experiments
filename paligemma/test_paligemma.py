import torch
from dotenv import load_dotenv
import os

from paligemma import PaligemmaForVQAClassification, get_vivqa_paligemma, get_vivqa_paligemma_phobert
from paligemma_dataset import create_paligemma_dataset_by_split

from transformers import PhobertTokenizer

from huggingface_hub import login

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
        self.phobert= False
        self.pin_mem = True
        self.dist_eval = False

args = TempArgs()

phobert_tokenizer = PhobertTokenizer.from_pretrained("vinai/phobert-base-v2", use_fast=True)

dataset_train, dataloader_train = create_paligemma_dataset_by_split(args, split="train", phobert_tokenizer=phobert_tokenizer)
dataset_test, dataloader_test = create_paligemma_dataset_by_split(args, split="test", phobert_tokenizer=phobert_tokenizer)
dataset_val, dataloader_val = create_paligemma_dataset_by_split(args, split="val", phobert_tokenizer=phobert_tokenizer)

max_len_id = 0
max_len_img = 0
for item in dataset_train:
    max_len_id = max(max_len_id, len(item["input_ids"]))
    max_len_img = max(max_len_img, len(item["pixel_values"][1]) * len(item["pixel_values"][2]))

for item in dataset_val:
    max_len_id = max(max_len_id, len(item["input_ids"]))
    max_len_img = max(max_len_img, len(item["pixel_values"][1]) * len(item["pixel_values"][2]))

for item in dataset_test:
    max_len_id = max(max_len_id, len(item["input_ids"]))
    max_len_img = max(max_len_img, len(item["pixel_values"][1]) * len(item["pixel_values"][2]))

print(max_len_id)
print(max_len_img)

# device = torch.device("cuda:5")

# model = get_vivqa_paligemma(
#     answer2label_path="/home/21khac.dd/bm/data/vivqa/annotations/dicts/answer2label_en_gemini_translated.txt",
#     device=device
# ).to(device, non_blocking = True)

# model = get_vivqa_paligemma_phobert(
#     answer2label_path="/home/21khac.dd/bm/data/vivqa/annotations/dicts/answer2label_en_gemini_translated.txt",
#     device=device
# ).to(device, non_blocking = True)

