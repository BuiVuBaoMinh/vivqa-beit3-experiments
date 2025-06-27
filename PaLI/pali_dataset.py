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
from glossary import segment_normalize


def remove_under_score(text: str):
    return text.replace("_", " ")

class ViVQAPaLIDataset(torch.utils.data.Dataset):
    def __init__(self,
                 args,
                 json_path, image_dir, split,
                 vit_model_name="google/vit-base-patch16-224", mt5_model_name="google/mt5-base",
                 phobert_tokenizer=None):
        
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.image_dir = image_dir

        if split == "train":
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

class ViVQAPaLIClassificationDataset(torch.utils.data.Dataset):
    def __init__(self,
                 args,
                 json_path, image_dir, split,
                 vit_model_name="google/vit-base-patch16-224", mt5_model_name="google/mt5-base",
                 phobert_tokenizer=None):
        
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        assert args.answer2label is not None, "ViVQAPaLIClassificationDataset: answer2label.txt path is None!"
        self.answer_to_label, self.label_to_answer = self._load_answer_mappings(args.answer2label)

        self.image_dir = image_dir

        if split == "train":
            self.image_processor = build_vqa_transform(is_train=True)
        else:
            self.image_processor = build_vqa_transform(is_train=False)

        if args.phobert and phobert_tokenizer is not None:
            self.tokenizer = phobert_tokenizer
        else:
            self.tokenizer = MT5Tokenizer.from_pretrained(mt5_model_name)

        self.input_max_length = 50
        self.split = split

    def _load_answer_mappings(self, file_path):
        """
        Loads the answer to label mappings from the provided JSONL file.
        Returns:
            - A dictionary mapping answer strings to integer labels.
            - A dictionary mapping integer labels to answer strings.
        """
        answer_to_label = {} # there are same answers with different labels in translated en!
        label_to_answer = {}
        seen_answers = set()
        unique_answers = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                answer = remove_under_score(item['answer'])
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

        print(f"ViVQAPaLIClassificationDataset: Loaded {len(answer_to_label)} answer-to-label mappings from {file_path}.")
        return answer_to_label, label_to_answer

    def __getitem__(self, idx):
        item = self.data[idx]
        qid = item["id"]

        image_id = item["image"]
        image_path = self.image_dir + f"/{image_id}.jpg"
        image = Image.open(image_path).convert("RGB")

        image = self.image_processor(image)  # This returns a tensor normalized for ViT

        pixel_values = image  # Already a tensor, shape [3, H, W], no need for .squeeze()

        tokenized_text = self.tokenizer(item["question"], return_tensors="pt",
                                        padding="max_length", max_length=self.input_max_length, truncation=True)
        input_ids = tokenized_text.input_ids.squeeze(0)
        attention_mask = tokenized_text.attention_mask.squeeze(0)

        answer_str = segment_normalize(item["answer"])
        label = self.answer_to_label.get(answer_str, len(self.answer_to_label)-1) # Default to "UNKNOWN" if answer not in map
        if label == len(self.answer_to_label)-1:
            print(f"Warning: Answer '{answer_str}' for qid {qid} not found in answer map. Returning label {len(self.answer_to_label)-1}: {self.label_to_answer[len(self.answer_to_label)-1]}.")

        # label = self.answer_to_label.get(answer_str, 0) # Default to "0" if answer not in map
        # if label == 0 and answer_str != "bowl":
        #     print(f"Warning: Answer '{answer_str}' for qid {qid} not found in answer map. Returning label 0.")

        return {
            "qid": qid,
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label, dtype=torch.long),
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
        drop_last=True, # Always drop last batch --> should set batch size that divides evenly the data.
        collate_fn=utils.merge_batch_tensors_by_dict_key,
    )

def create_pali_dataset_by_split(args, split, is_train=True, phobert_tokenizer=None):
    json_path = args.data_path + f"/vqa/{split}_{'en' if phobert_tokenizer is None else 'vi'}.json"
    print(f"Creating dataset '{split}' from data path {json_path}")
    if split=="train" or split=="val":
        image_dir = args.data_path + "/images/train"
    if split=="test":
        image_dir = args.data_path + "/images/test"

    if args.pali_class == 'pali_classification':
        dataset = ViVQAPaLIClassificationDataset(
            args=args,
            json_path=json_path,
            image_dir=image_dir,
            split=split,
            phobert_tokenizer=phobert_tokenizer
        )
    else:
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