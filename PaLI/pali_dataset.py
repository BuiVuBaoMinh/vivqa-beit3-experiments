import json
import os
import torch
import sys

from PIL import Image
from transformers import MT5Tokenizer, ViTImageProcessor
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils
from randaug import RandomAugment

class ViVQAPaLIDataset(torch.utils.data.Dataset):
    def __init__(self,
                 args,
                 json_path, image_dir, split,
                 vit_model_name="google/vit-base-patch16-224", mt5_model_name="google/mt5-base",
                 phobert_tokenizer=None):
        
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.image_dir = image_dir

        # self.image_processor = ViTImageProcessor.from_pretrained(vit_model_name)

        if split == "train" or split == "val":
            self.image_processor = build_vqa_transform(is_train=True)
        else:
            self.image_processor = build_vqa_transform(is_train=False)

        if args.phobert and phobert_tokenizer is not None:
            self.tokenizer = phobert_tokenizer
        else:
            self.tokenizer = MT5Tokenizer.from_pretrained(mt5_model_name)

        self.input_max_length = 50
        self.label_max_length = 4
        self.split = split

    def __getitem__(self, idx):
        qid = self.data[idx]["id"]

        image_id = self.data[idx]["image"]
        image_path = self.image_dir + f"/{image_id}.jpg"
        image = Image.open(image_path).convert("RGB")

        # pixel_values = self.image_processor(images=image, return_tensors="pt").pixel_values.squeeze(0)

        image = self.image_processor(image)  # This returns a tensor normalized for ViT

        pixel_values = image  # Already a tensor, shape [3, H, W], no need for .squeeze()

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
    if is_train:
        sampler = torch.utils.data.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)
    
    return torch.utils.data.DataLoader(
        dataset, sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=is_train,
        collate_fn=utils.merge_batch_tensors_by_dict_key,
    )

def create_pali_dataset_by_split(args, split, is_train=True, phobert_tokenizer=None):
    json_path = args.data_path + f"/vqa/{split}_{'en' if phobert_tokenizer is None else 'vi'}.json"
    if split=="train" or split=="val":
        image_dir = args.data_path + "/images/train"
    if split=="test":
        image_dir = args.data_path + "/images/test"

    dataset = ViVQAPaLIDataset(
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

    data_loader = create_pali_dataloader(
        dataset, is_train=is_train, batch_size=batch_size, 
        num_workers=args.num_workers, pin_mem=args.pin_mem, dist_eval=args.dist_eval, 
    )
    
    return dataset, data_loader

def create_pali_datasets(args, is_eval=False, phobert_tokenizer=None):
    if is_eval:
        return create_pali_dataset_by_split(args, split="test", is_train=False, phobert_tokenizer=phobert_tokenizer)
    else:
        dataset_train, data_loader_train =create_pali_dataset_by_split(
            args, split="train", is_train=True, phobert_tokenizer=phobert_tokenizer
        )
        dataset_val, data_loader_val = create_pali_dataset_by_split(
            args, split="val", is_train=True, phobert_tokenizer=phobert_tokenizer
        )
        return dataset_train, data_loader_train, dataset_val, data_loader_val
            


def build_vqa_transform(is_train=True):
    if is_train:
        t = [
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
        ]
        t.append(
                RandomAugment(
                    2, 7, isPIL=True,
                    augs=[
                        'Identity','AutoContrast','Equalize','Brightness','Sharpness',
                        'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate',
                    ]
                )
            )
        t += [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
        ]
    else:
        t = [
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
        ]

    return transforms.Compose(t)