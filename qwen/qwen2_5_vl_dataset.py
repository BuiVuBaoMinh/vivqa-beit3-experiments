import json
import os
import torch
import sys

from PIL import Image
from transformers import Qwen2_5_VLProcessor, PhobertTokenizer, Qwen2TokenizerFast, Qwen2VLImageProcessor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import utils
from glossary import segment_normalize
from Qwen25VLFitPhobertTokenizer import Qwen2_5_VLFitPhobertTokenizer
from qwen2_5_vl import QWEN_2_5_VL_3B_ID

QWEN2_5_VL_ID = QWEN_2_5_VL_3B_ID


def remove_under_score(text: str):
    return text.replace("_", " ")
        
class ViVQAQwen2_5_VLProcessor(Qwen2_5_VLProcessor):
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast", "Qwen25VLFitPhobertTokenizer")

    def __init__(self, image_processor, tokenizer, **kwargs):
        super().__init__(image_processor, tokenizer, **kwargs)

class ViVQAQwen2_5_VLDataset(torch.utils.data.Dataset):
    def __init__(self,
                 args,
                 json_path, image_dir, split,
                 qwen2_5_vl_model_id=QWEN2_5_VL_ID,
                 phobert_tokenizer: Qwen2_5_VLFitPhobertTokenizer = None):
        
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        assert args.answer2label is not None, "ViVQAQwen2_5_VLDataset: answer2label.txt path is None!"
        self.answer_to_label, self.label_to_answer = self._load_answer_mappings(args.answer2label)

        self.image_dir = image_dir

        # Use max_pixels from args, default to 224*224 if not present
        max_pixels_to_use = getattr(args, 'max_pixels', 224*224)
        print(f"Initializing Qwen2_5_VLProcessor with max_pixels = {max_pixels_to_use}")
        self.processor = Qwen2_5_VLProcessor.from_pretrained(
            qwen2_5_vl_model_id, use_fast=True,
            max_pixels = max_pixels_to_use,
        )

        self.split = split
        self.max_length = 15
    
    def __len__(self):
        return len(self.data)

    def _load_answer_mappings(self, file_path):
        """
        Loads the answer to label mappings from the provided JSONL file.
        Returns:
            - A dictionary mapping answer strings to integer labels.
            - A dictionary mapping integer labels to answer strings.
        """
        answer_to_label = {}
        label_to_answer = {}
        seen_answers = set()
        unique_answers = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                answer = segment_normalize(item['answer'])
                label_to_answer[item['label']] = answer

                if answer in seen_answers:
                    continue  # Skip duplicates
                seen_answers.add(answer)
                unique_answers.append(answer)

                answer_to_label[answer] = item['label']

        if len(answer_to_label) != len(label_to_answer):
            print(f"Warning: answer_to_label and label_to_answer dicts mismatch {len(answer_to_label)} vs {len(label_to_answer)}!")
            print("This could be translation or data source error.")
            print("Modifying label_to_answer to match answer_to_label..")
            label_to_answer = {}
            label_to_answer = {v: k for k, v in answer_to_label.items()}

        # Duplication removal could results in labels out of bound
        answer_to_label = {ans: idx for idx, ans in enumerate(unique_answers)}
        label_to_answer = {idx: ans for ans, idx in answer_to_label.items()}
        print(f"Reindexed {len(answer_to_label)} unique answers to labels 0 through {len(answer_to_label) - 1}.")

        answer_to_label["UNKNOWN"] = len(answer_to_label)
        label_to_answer[len(label_to_answer)] = "UNKNOWN"
        print("Append a default UNKNOWN label.")

        print(f"ViVQAPaligemmaDataset: Loaded {len(answer_to_label)} answer-to-label mappings from {file_path}.")
        return answer_to_label, label_to_answer

    def __getitem__(self, idx):
        item = self.data[idx]
        qid = item["id"]

        image_id = item["image"]
        image_path = self.image_dir + f"/{image_id}.jpg"
        image = Image.open(image_path).convert("RGB")

        prompt = f"<|image_pad|>\n{segment_normalize(item['question'])}"

        inputs = self.processor(
            text = prompt,
            images = image,
            return_tensors = "pt",
            padding = 'max_length',
            truncation= False,
            max_length=self.max_length,
        )

        # --- ADD THIS DEBUGGING CODE ---
        # Use the tokenizer inside the processor to decode the input_ids
        # .squeeze(0) removes the batch dimension for a single item
        # decoded_text = self.processor.tokenizer.decode(
        #     inputs['input_ids'].squeeze(0), 
        #     skip_special_tokens=False # Set to False to see all special tokens
        # )
        # print("--- Decoded Input ---")
        # print(decoded_text)
        # print(f"Length: {len(inputs['input_ids'].squeeze(0))}") # Verify the length
        # print(f"Length: {len(inputs['pixel_values'].squeeze(0))}")
        # print(f"Length: {len(inputs['attention_mask'].squeeze(0))}")
        # print("---------------------")
        # --- END DEBUGGING CODE ---

        pixel_values = inputs['pixel_values'].squeeze(0)
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)
        image_grid_thw = inputs['image_grid_thw'].squeeze(0)

        answer_str = segment_normalize(item["answer"])
        label = self.answer_to_label.get(answer_str, len(self.answer_to_label)-1) # Default to "UNKNOWN" if answer not in map
        if label == len(self.answer_to_label)-1:
            print(f"Warning: Answer '{answer_str}' for qid {qid} not found in answer map. Returning label {len(self.answer_to_label)-1}: {self.label_to_answer[len(self.answer_to_label)-1]}.")

        return {
            "qid": qid,
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image_grid_thw": image_grid_thw,
            "labels": torch.tensor(label, dtype=torch.long),
        }

    def get_tokenizer(self):
        return self.processor.tokenizer
    
    def __len__(self):
        return len(self.data)
    

def create_qwen25vl_dataloader(dataset, is_train, batch_size, num_workers, pin_mem, dist_eval=False):
    if is_train:
        sampler = torch.utils.data.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)
    
    return torch.utils.data.DataLoader(
        dataset, sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=True, # Always drop last batch --> should set batch size that divides evenly the data.
        collate_fn=utils.merge_batch_tensors_by_dict_key,
    )

def create_qwen25vl_dataset_by_split(args, split, is_train=True, phobert_tokenizer=None):
    en_json_suffix = 'en'
    vi_json_suffix = 'vi_en_ans'

    json_path = args.data_path + f"/vqa/{split}_{en_json_suffix if phobert_tokenizer is None else vi_json_suffix}.json"
    print(f"Creating dataset '{split}' from data path {json_path}")
    if split=="train" or split=="val":
        image_dir = args.data_path + "/images/train"
    if split=="test":
        image_dir = args.data_path + "/images/test"

    dataset = ViVQAQwen2_5_VLDataset(
        args=args,
        json_path=json_path,
        image_dir=image_dir,
        split=split,
        phobert_tokenizer=phobert_tokenizer
    )

    if is_train:
        batch_size = args.batch_size
    elif hasattr(args, "eval_batch_size") and args.eval_batch_size is not None:
        batch_size = args.eval_batch_size
    else:
        batch_size = int(args.batch_size * 1.5)

    data_loader = create_qwen25vl_dataloader(
        dataset, is_train=is_train, batch_size=batch_size, 
        num_workers=args.num_workers, pin_mem=args.pin_mem, dist_eval=args.dist_eval, 
    )
    
    return dataset, data_loader

def create_qwen2_5_vl_datasets(args, is_eval=False, phobert_tokenizer=None):
    if is_eval:
        return create_qwen25vl_dataset_by_split(args, split="test", is_train=False, phobert_tokenizer=phobert_tokenizer)
    else:
        dataset_train, data_loader_train =create_qwen25vl_dataset_by_split(
            args, split="train", is_train=True, phobert_tokenizer=phobert_tokenizer
        )
        dataset_val, data_loader_val = create_qwen25vl_dataset_by_split(
            args, split="val", is_train=True, phobert_tokenizer=phobert_tokenizer
        )
        return dataset_train, data_loader_train, dataset_val, data_loader_val