import json
import os
import torch
import sys

from PIL import Image
from transformers import (
    SmolVLMProcessor, SmolVLMImageProcessor,
    PhobertTokenizer, AutoTokenizer, AutoProcessor
)
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mySmolVLM import SMOLVLM_ID, SMOLVLM_2B_MODEL_ID

import utils
from glossary import segment_normalize


def remove_under_score(text: str):
    return text.replace("_", " ")

class ViVQASmolVLMProcessor(SmolVLMProcessor):
    tokenizer_class = ("AutoTokenizer", "PhobertTokenizer")

    def __init__(self, image_processor, tokenizer, **kwargs):
        super().__init__(image_processor, tokenizer, **kwargs)

class ViVQASmolVLMDataset(torch.utils.data.Dataset):
    def __init__(self,
                 args,
                 json_path, image_dir, split,
                 smolvlm_model_id=SMOLVLM_ID,
                 phobert_tokenizer: PhobertTokenizer = None):
        
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        assert args.answer2label is not None, "ViVQASmolVLMDataset: answer2label.txt path is None!"
        self.answer_to_label, self.label_to_answer = self._load_answer_mappings(args.answer2label)

        self.image_dir = image_dir

        self.processor = SmolVLMProcessor.from_pretrained(smolvlm_model_id, use_fast=True)
        # self.processor = AutoProcessor.from_pretrained(smolvlm_model_id, use_fast=True)

        self.processor.image_processor.do_resize = True
        self.processor.image_processor.size = {"longest_edge": 2 * 384}
        self.processor.image_processor.max_image_size = {"longest_edge": 384}

        if args.phobert and phobert_tokenizer is not None:
            tokens_to_add = {
                "additional_special_tokens": [
                    self.processor.fake_image_token,
                    self.processor.image_token,
                    self.processor.end_of_utterance_token,
                    self.processor.global_image_token
                ]
            }
            phobert_tokenizer.add_special_tokens(tokens_to_add)
            self.processor.image_token_id = phobert_tokenizer.convert_tokens_to_ids(self.processor.image_token)
            self.processor.tokenizer = phobert_tokenizer

            image_token_id = phobert_tokenizer.convert_tokens_to_ids(self.processor.image_token)
            fake_image_token_id = phobert_tokenizer.convert_tokens_to_ids(self.processor.fake_image_token)
            end_of_utterance_token_id = phobert_tokenizer.convert_tokens_to_ids(self.processor.end_of_utterance_token)
            
            print(f"Added {self.processor.image_token} token to PhobertTokenizer with new ID: {image_token_id}")
            print(f"Added {self.processor.fake_image_token} token to PhobertTokenizer with new ID: {fake_image_token_id}")
            print(f"Added {self.processor.end_of_utterance_token} token to PhobertTokenizer with new ID: {end_of_utterance_token_id}")
            if isinstance(self.processor, SmolVLMProcessor):
                global_image_token_id = phobert_tokenizer.convert_tokens_to_ids(self.processor.global_image_token)
                print(f"Added {self.processor.global_image_token} token to PhobertTokenizer with new ID: {global_image_token_id}")


        self.split = split
        self.max_length = 512
    
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

        # assert len(answer_to_label) == len(label_to_answer), \
        #     f"answer_to_label and label_to_answer dicts mismatch {len(answer_to_label)} vs {len(label_to_answer)}!"

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

        print(f"ViVQASmolVLMDataset: Loaded {len(answer_to_label)} answer-to-label mappings from {file_path}.")
        return answer_to_label, label_to_answer

    def __getitem__(self, idx):
        item = self.data[idx]
        qid = item["id"]

        image_id = item["image"]
        image_path = self.image_dir + f"/{image_id}.jpg"
        image = Image.open(image_path).convert("RGB")

        question = "<image>" + item["question"]

        answer_str = segment_normalize(item["answer"])
        label = self.answer_to_label.get(answer_str, len(self.answer_to_label)-1) # Default to "UNKNOWN" if answer not in map
        if label == len(self.answer_to_label)-1:
            print(f"Warning: Answer '{answer_str}' for qid {qid} not found in answer map. Returning label {len(self.answer_to_label)-1}: {self.label_to_answer[len(self.answer_to_label)-1]}.")

        return {
            "image": image,
            "question": question,
            "labels": label,
            "qid": qid
        }

    def get_tokenizer(self):
        return self.processor.tokenizer
    
    def __len__(self):
        return len(self.data)
    

def create_smolvlm_dataloader(dataset, is_train, batch_size, num_workers, pin_mem, device, dist_eval=False):
    if is_train:
        sampler = torch.utils.data.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)
    
    # Use a partial function to pass the processor to the collate_fn
    from functools import partial
    collate_fn = partial(smolvlm_collate_fn, processor=dataset.processor)

    return torch.utils.data.DataLoader(
        dataset, sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=True, # Always drop last batch --> should set batch size that divides evenly the data.
        # collate_fn=utils.merge_batch_tensors_by_dict_key,
        collate_fn=collate_fn,
    )

def create_smolvlm_dataset_by_split(args, split, is_train=True, phobert_tokenizer=None, device=None):
    en_json_suffix = 'en'
    vi_json_suffix = 'vi_en_ans'

    json_path = args.data_path + f"/vqa/{split}_{en_json_suffix if phobert_tokenizer is None else vi_json_suffix}.json"
    print(f"Creating dataset '{split}' from data path {json_path}")
    if split=="train" or split=="val":
        image_dir = args.data_path + "/images/train"
    if split=="test":
        image_dir = args.data_path + "/images/test"

    dataset = ViVQASmolVLMDataset(
        args=args,
        json_path=json_path,
        image_dir=image_dir,
        split=split,
        smolvlm_model_id=SMOLVLM_ID,
        phobert_tokenizer=phobert_tokenizer
    )

    if is_train:
        batch_size = args.batch_size
    elif hasattr(args, "eval_batch_size") and args.eval_batch_size is not None:
        batch_size = args.eval_batch_size
    else:
        batch_size = int(args.batch_size * 1.5)

    data_loader = create_smolvlm_dataloader(
        dataset, is_train=is_train, batch_size=batch_size, 
        num_workers=args.num_workers, pin_mem=args.pin_mem, dist_eval=args.dist_eval, device=device,
    )
    
    return dataset, data_loader

def create_smolvlm_datasets(args, is_eval=False, phobert_tokenizer=None, device=None):
    if is_eval:
        return create_smolvlm_dataset_by_split(args, split="test", is_train=False, phobert_tokenizer=phobert_tokenizer, device=device)
    else:
        dataset_train, data_loader_train =create_smolvlm_dataset_by_split(
            args, split="train", is_train=True, phobert_tokenizer=phobert_tokenizer, device=device
        )
        dataset_val, data_loader_val = create_smolvlm_dataset_by_split(
            args, split="val", is_train=True, phobert_tokenizer=phobert_tokenizer, device=device
        )
        return dataset_train, data_loader_train, dataset_val, data_loader_val


import torch
from collections.abc import Mapping

def smolvlm_collate_fn(batch, processor):
    """
    The definitive collate function that processes an entire batch at once.
    """
    # Extract images and questions from the batch
    images = [[item['image']] for item in batch]
    questions = [item['question'] for item in batch]
    
    # Process the entire batch of images and questions at once
    inputs = processor(
        text=questions, 
        images=images, 
        return_tensors="pt", 
        padding=True, # Pad text to the longest in the batch
        truncation=False,
    )

    # Gather labels and qids
    inputs['labels'] = torch.tensor([item['labels'] for item in batch], dtype=torch.long)
    inputs['qid'] = [item['qid'] for item in batch]
    
    return inputs

def smolvlm_collate_fn_2(batch, processor):
    """
    The definitive collate function that processes an entire batch at once by
    separating text and vision processing. This is more robust and avoids
    issues with the main processor's batch handling and truncation logic.
    """
    # Extract images and questions from the batch
    images = [item['image'] for item in batch]
    questions = [item['question'] for item in batch]

    text_inputs = processor.tokenizer(
        text=questions,
        return_tensors="pt",
        padding="True",
        truncation=False,
        max_length=512
    )

    # 2. Process all image inputs in a single batch call.
    vision_inputs = processor.image_processor(
        images=images,
        return_tensors="pt"
    )

    inputs = {**text_inputs, **vision_inputs}

    inputs['labels'] = torch.tensor([item['labels'] for item in batch], dtype=torch.long)
    inputs['qid'] = [item['qid'] for item in batch]

    return inputs