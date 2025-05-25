from datasets import OpenViVQADataset, ViVQADataset
from transformers import XLMRobertaTokenizer

import torch
from transformers import AutoModel, AutoTokenizer

# tokenizer = XLMRobertaTokenizer("text-tokenizers/beit3.spm")
# tokenizer = XLMRobertaTokenizer("/home/lenovo/exp1/beit3-model-n-ckpts/beit3-model/beit3.spm") # for gcp vm

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2") # using PhoBERT tokenizer

VIVIQA_DATASET_PATH = "data/vivqa"

data_path = VIVIQA_DATASET_PATH
annotation_data_path = VIVIQA_DATASET_PATH + "/vqa"

ViVQADataset.make_dataset_index(
    data_path=data_path,
    tokenizer=tokenizer,
    annotation_data_path=annotation_data_path,
    # phobert=False,
    phobert=True
)