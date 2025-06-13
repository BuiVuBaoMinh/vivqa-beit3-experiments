import json
import os
import torch
from PIL import Image
from transformers import MT5ForConditionalGeneration, MT5Tokenizer, MT5Model, ViTConfig, ViTModel, ViTImageProcessor


class ViVQAPaLIDataset(torch.utils.data.Dataset):
    def __init__(self, json_path, image_dir, \
                 vit_model_name="google/vit-base-patch16-224", mt5_model_name="google/mt5-base"):
        
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.image_dir = image_dir

        self.image_processor = ViTImageProcessor.from_pretrained(vit_model_name)

        self.tokenizer = MT5Tokenizer.from_pretrained(mt5_model_name)

        self.input_max_length = 30
        self.label_max_length = 4

    def __getitem__(self, idx):
        qid = self.data[idx]["id"]

        image_id = self.data[idx]["image"]
        image_path = self.image_dir + f"/{image_id}.jpg"
        image = Image.open(image_path).convert("RGB")

        pixel_values = self.image_processor(images=image, return_tensors="pt").pixel_values.squeeze(0)

        tokenized_text = self.tokenizer(self.data[idx]["question"], return_tensors="pt",
                                        padding="max_length", max_length=self.input_max_length, truncation=True)
        input_ids = tokenized_text.input_ids.squeeze(0)
        attention_mask = tokenized_text.attention_mask.squeeze(0)

        labels = self.tokenizer(self.data[idx]["answer"], return_tensors="pt",
                                padding="max_length", max_length=self.label_max_length, truncation=True).input_ids.squeeze(0)

        return {
            "qid": qid,
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    
    def __len__(self):
        return len(self.data)
        

