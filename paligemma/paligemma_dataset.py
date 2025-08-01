import json
import os
import torch
import sys

from PIL import Image
from transformers import PaliGemmaForConditionalGeneration, PhobertTokenizer, PaliGemmaProcessor, GemmaTokenizer
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from transformers.tokenization_utils_base import AddedToken
from transformers.models.paligemma.processing_paligemma import IMAGE_TOKEN, EXTRA_TOKENS

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import utils
from glossary import segment_normalize


def remove_under_score(text: str):
    return text.replace("_", " ")

class ViVQAPaligemmaProcessor(PaliGemmaProcessor):
    tokenizer_class = ("GemmaTokenizer", "GemmaTokenizerFast", "PhobertTokenizer")

    def __init__(self, image_processor, tokenizer, **kwargs):
        super().__init__(image_processor, tokenizer, **kwargs)

class ViVQAPaligemmaDataset(torch.utils.data.Dataset):
    def __init__(self,
                 args,
                 json_path, image_dir, split,
                 paligemma_model_id="google/paligemma2-3b-pt-224",
                 phobert_tokenizer: PhobertTokenizer = None):
        
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        assert args.answer2label is not None, "ViVQAPaligemmaDataset: answer2label.txt path is None!"
        self.answer_to_label, self.label_to_answer = self._load_answer_mappings(args.answer2label)

        self.image_dir = image_dir

        self.processor = PaliGemmaProcessor.from_pretrained(paligemma_model_id, use_fast=True)

        if args.phobert and phobert_tokenizer is not None:
            if not hasattr(phobert_tokenizer, "image_token"):
                image_token = AddedToken(IMAGE_TOKEN, normalized=False, special=True)
                tokens_to_add = {"additional_special_tokens": [image_token]}
                phobert_tokenizer.add_special_tokens(tokens_to_add)
                self.processor.image_token_id = phobert_tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
                self.processor.image_token = IMAGE_TOKEN

                phobert_tokenizer.add_tokens(EXTRA_TOKENS)
                phobert_tokenizer.add_bos_token = False
                phobert_tokenizer.add_eos_token = False

                self.processor.tokenizer = phobert_tokenizer

        self.split = split
        # self.max_length = 290
        self.max_length = 384 # Phobert tokenizes up to 299 tokens

        self.dummy_suffix = None
        self.dummy_suffix = self.processor.tokenizer.decode(self.processor.tokenizer.pad_token_id)
        print(f"Paligemma Processor's dummy suffix: {self.dummy_suffix}")
    
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

        print(f"ViVQAPaligemmaDataset: Loaded {len(answer_to_label)} answer-to-label mappings from {file_path}.")
        return answer_to_label, label_to_answer

    def __getitem__(self, idx):
        item = self.data[idx]
        qid = item["id"]

        image_id = item["image"]
        image_path = self.image_dir + f"/{image_id}.jpg"
        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(
            text = "<image>\n" + item['question'],
            images = image,
            return_tensors = "pt",
            padding = 'max_length',
            truncation=False,
            max_length=self.max_length,
            suffix = segment_normalize(item["answer"]) if self.dummy_suffix is None else self.dummy_suffix
        )

        # print(self.processor.tokenizer.pad_token_id)
        # print(self.processor.tokenizer.decode(self.processor.tokenizer.pad_token_id))

        pixel_values = inputs['pixel_values'].squeeze(0)
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)
        token_type_ids = inputs['token_type_ids'].squeeze(0)
        paligemma_labels = inputs['labels'].squeeze(0)

        answer_str = segment_normalize(item["answer"])
        label = self.answer_to_label.get(answer_str, len(self.answer_to_label)-1) # Default to "UNKNOWN" if answer not in map
        if label == len(self.answer_to_label)-1:
            print(f"Warning: Answer '{answer_str}' for qid {qid} not found in answer map. Returning label {len(self.answer_to_label)-1}: {self.label_to_answer[len(self.answer_to_label)-1]}.")

        return {
            "qid": qid,
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "paligemma_labels": paligemma_labels,
            "labels": torch.tensor(label, dtype=torch.long),
        }

    def get_tokenizer(self):
        return self.processor.tokenizer
    
    def __len__(self):
        return len(self.data)
    

def create_paligemma_dataloader(dataset, is_train, batch_size, num_workers, pin_mem, dist_eval=False):
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

def create_paligemma_dataset_by_split(args, split, is_train=True, phobert_tokenizer=None):
    en_json_suffix = 'en'
    vi_json_suffix = 'vi_en_ans'

    json_path = args.data_path + f"/vqa/{split}_{en_json_suffix if phobert_tokenizer is None else vi_json_suffix}.json"
    print(f"Creating dataset '{split}' from data path {json_path}")
    if split=="train" or split=="val":
        image_dir = args.data_path + "/images/train"
    if split=="test":
        image_dir = args.data_path + "/images/test"

    dataset = ViVQAPaligemmaDataset(
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

    data_loader = create_paligemma_dataloader(
        dataset, is_train=is_train, batch_size=batch_size, 
        num_workers=args.num_workers, pin_mem=args.pin_mem, dist_eval=args.dist_eval, 
    )
    
    return dataset, data_loader

def create_paligemma_datasets(args, is_eval=False, phobert_tokenizer=None):
    if is_eval:
        return create_paligemma_dataset_by_split(args, split="test", is_train=False, phobert_tokenizer=phobert_tokenizer)
    else:
        dataset_train, data_loader_train =create_paligemma_dataset_by_split(
            args, split="train", is_train=True, phobert_tokenizer=phobert_tokenizer
        )
        dataset_val, data_loader_val = create_paligemma_dataset_by_split(
            args, split="val", is_train=True, phobert_tokenizer=phobert_tokenizer
        )
        return dataset_train, data_loader_train, dataset_val, data_loader_val