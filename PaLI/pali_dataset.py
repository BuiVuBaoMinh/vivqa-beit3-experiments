import json
import os
import torch

from PIL import Image
from transformers import MT5Tokenizer, ViTImageProcessor
from torch.utils.data import DataLoader

import utils


class ViVQAPaLIDataset(torch.utils.data.Dataset):
    def __init__(self,
                 args,
                 json_path, image_dir, split,
                 vit_model_name="google/vit-base-patch16-224", mt5_model_name="google/mt5-base",
                 phobert_tokenizer=None):
        
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.image_dir = image_dir

        self.image_processor = ViTImageProcessor.from_pretrained(vit_model_name)

        if args.phobert and phobert_tokenizer is not None:
            self.tokenizer = phobert_tokenizer
        else:
            self.tokenizer = MT5Tokenizer.from_pretrained(mt5_model_name)

        self.input_max_length = 50
        self.label_max_length = 5
        self.split = split

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


def create_pali_dataloader(dataset, is_train, batch_size, num_workers, pin_mem, dist_eval=False):
    if is_train or dist_eval:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()

        if not is_train and dist_eval and len(dataset) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')

        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=is_train
        )
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)
    
    return torch.utils.data.DataLoader(
        dataset, sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=is_train,
        collate_fn=utils.merge_batch_tensors_by_dict_key, # TODO: Reivew this params in case of bug
    )

def create_pali_dataset_by_split(args, split, is_train=True, phobert_tokenizer=None):
    json_path = args.data_path + f"/vqa/{split}_{"en" if phobert_tokenizer is None else "vi"}.json"
    if split=="train" or split=="val":
        image_dir = args.data_path + "/images/train"
    if split=="test":
        image_dir = args.data_path + "/images/test"

    dataset = ViVQAPaLIDataset(
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

    data_loader = create_pali_dataloader(
        dataset, is_train=is_train, batch_size=batch_size, 
        num_workers=args.num_workers, pin_mem=args.pin_mem, dist_eval=args.dist_eval, 
    )
    
    return dataset, data_loader

def create_pali_datasets(args, is_eval=False, phobert_tokenizer=None):
    if is_eval:
        return create_pali_dataset_by_split(args, split="test", is_train=False, phobert_tokenizer=phobert_tokenizer)
    else:
        return \
            create_pali_dataset_by_split(args, split="train", is_train=True, phobert_tokenizer=phobert_tokenizer), \
            create_pali_dataset_by_split(args, split="val", is_train=True, phobert_tokenizer=phobert_tokenizer)


