import json
import os
import torch
from PIL import Image
from transformers import Blip2Processor, PhobertTokenizer
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from my_blip2 import BLIP2_MODEL_ID
from glossary import segment_normalize

class ViVQABlip2Dataset(torch.utils.data.Dataset):
    """
    Dataset for VQA with the BLIP-2 model.
    """
    def __init__(self,
                 args,
                 json_path: str,
                 image_dir: str,
                 split: str,
                 blip2_model_id: str = BLIP2_MODEL_ID,
                 phobert_tokenizer: PhobertTokenizer = None):
        
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        assert args.answer2label is not None, "ViVQABlip2Dataset: answer2label path is required!"
        self.answer_to_label, self.label_to_answer, self.label_to_vi_answer = self._load_answer_mappings(args.answer2label)

        self.image_dir = image_dir
        self.split = split
        
        # If using PhoBERT, the processor will use the provided tokenizer
        if args.phobert and phobert_tokenizer:
            print("Initializing Blip2Processor with custom PhoBERT tokenizer.")
            self.processor = Blip2Processor.from_pretrained(blip2_model_id, tokenizer=phobert_tokenizer, use_fast=True)
        else:
            print("Initializing standard Blip2Processor Processor.")
            self.processor = Blip2Processor.from_pretrained(blip2_model_id, use_fast=True)

    def _load_answer_mappings(self, file_path):
        """Loads answer-to-label mappings from a file."""
        answer_to_label = {}
        label_to_answer = {}
        label_to_vi_answer = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                answer = segment_normalize(item['answer'])
                label = item['label']
                answer_to_label[answer] = label
                label_to_answer[label] = answer

                vi_answer = segment_normalize(item['vi_answer'])
                label_to_vi_answer[label] = vi_answer
        
        if "UNKNOWN" not in answer_to_label:
            new_label = len(answer_to_label)
            answer_to_label["UNKNOWN"] = new_label
            label_to_answer[new_label] = "UNKNOWN"

        print(f"ViVQABlip2Dataset: Loaded {len(answer_to_label)} mappings from {file_path}.")
        return answer_to_label, label_to_answer, label_to_vi_answer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        qid = item["id"]

        image_path = os.path.join(self.image_dir, f"{item['image']}.jpg")
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"Warning: Image not found at {image_path}.")
            return None # Will be filtered in collate_fn

        # BLIP-2 expects a specific prompt format for VQA.
        question = f"Question: {item['question']} Answer:"

        answer_str = segment_normalize(item["answer"])
        label = self.answer_to_label.get(answer_str, self.answer_to_label["UNKNOWN"])

        if isinstance(self.processor.tokenizer, PhobertTokenizer):
            vi_answer_str = self.label_to_vi_answer.get(label, "UNKNOWN")
            answer_str = vi_answer_str

        return {
            "image": image,
            "question": question,
            "labels": label,
            "answer_str": answer_str,
            "qid": qid
        }

    def get_tokenizer(self):
        return self.processor.tokenizer

def blip2_collate_fn(batch, processor):
    """
    Collate function to process a batch of samples with the Blip2Processor.
    """
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    images = [item['image'] for item in batch]
    questions = [item['question'] for item in batch]
    
    # Process the batch of images and text together
    # Use padding=True to handle variable length sequences
    inputs = processor(
        images=images, 
        text=questions, 
        return_tensors="pt", 
        padding=True,
        truncation=False,
    )

    inputs['labels'] = torch.tensor([item['labels'] for item in batch], dtype=torch.long)
    inputs['qid'] = [item['qid'] for item in batch]
    
    answer_str = [item['answer_str'] for item in batch]

    # Use the tokenizer to prepare decoder inputs
    # The tokenizer is accessed via processor.tokenizer
    # decoder_inputs = processor.tokenizer(
    #     answer_str,
    #     padding="longest",
    #     truncation=False,
    #     return_tensors="pt"
    # )

    # # These will be passed to the 'decoder_input_ids' and 'decoder_attention_mask'
    # # arguments of the model's forward pass.
    # inputs['decoder_input_ids'] = decoder_inputs['input_ids']
    # inputs['decoder_attention_mask'] = decoder_inputs['attention_mask']

    
    # Dummy decoder input ids since we are NOT causal generating
    decoder_input_ids = torch.full(
        (len(batch), 1),
        fill_value=processor.tokenizer.pad_token_id,
        dtype=torch.long
    )
    decoder_attention_mask = torch.ones_like(decoder_input_ids, dtype=torch.long)
    inputs['decoder_input_ids'] = decoder_input_ids
    inputs['decoder_attention_mask'] = decoder_attention_mask

    return inputs

def create_blip2_dataloader(dataset, is_train, batch_size, num_workers, pin_mem):
    sampler = torch.utils.data.RandomSampler(dataset) if is_train else torch.utils.data.SequentialSampler(dataset)
    
    from functools import partial
    collate_fn = partial(blip2_collate_fn, processor=dataset.processor)

    return torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=is_train,
        collate_fn=collate_fn,
    )

def create_blip2_dataset_by_split(args, split, is_train=True, phobert_tokenizer=None):
    json_suffix = 'vi_en_ans' if args.phobert else 'en'
    json_path = os.path.join(args.data_path, "vqa", f"{split}_{json_suffix}.json")
    print(f"Creating dataset '{split}' from data path {json_path}")
    
    image_dir = os.path.join(args.data_path, "images", "train" if split in ["train", "val"] else "test")

    dataset = ViVQABlip2Dataset(
        args=args,
        json_path=json_path,
        image_dir=image_dir,
        split=split,
        blip2_model_id=BLIP2_MODEL_ID,
        phobert_tokenizer=phobert_tokenizer
    )

    batch_size = args.batch_size if is_train else args.eval_batch_size or int(args.batch_size * 1.5)

    data_loader = create_blip2_dataloader(
        dataset,
        is_train=is_train,
        batch_size=batch_size, 
        num_workers=args.num_workers,
        pin_mem=args.pin_mem
    )
    
    return dataset, data_loader

def create_blip2_datasets(args, is_eval=False, phobert_tokenizer=None):
    if is_eval:
        return create_blip2_dataset_by_split(args, split="test", is_train=False, phobert_tokenizer=phobert_tokenizer)
    else:
        dataset_train, data_loader_train = create_blip2_dataset_by_split(
            args, split="train", is_train=True, phobert_tokenizer=phobert_tokenizer
        )
        dataset_val, data_loader_val = create_blip2_dataset_by_split(
            args, split="val", is_train=False, phobert_tokenizer=phobert_tokenizer
        )
        return dataset_train, data_loader_train, dataset_val, data_loader_val
