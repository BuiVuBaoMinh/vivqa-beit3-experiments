from datasets import OpenViVQADataset, ViVQADataset
from transformers import XLMRobertaTokenizer

import torch
from transformers import AutoModel, AutoTokenizer

tokenizer_spm = XLMRobertaTokenizer("/home/21khac.dd/bm/beit-3-checkpoints/beit3.spm")
# tokenizer = XLMRobertaTokenizer("/home/lenovo/exp1/beit3-model-n-ckpts/beit3-model/beit3.spm") # for gcp vm

tokenizer_phobert = AutoTokenizer.from_pretrained("vinai/phobert-base-v2") # using PhoBERT tokenizer

print(tokenizer_phobert.decode([82, 152, 12737, 1812, 1875, 25, 602]))
import sys
sys.exit(0)

VIVIQA_DATASET_PATH = "data/vivqa"

data_path = VIVIQA_DATASET_PATH
annotation_data_path = VIVIQA_DATASET_PATH + "/vqa"
predefined_dict_path = "/home/21khac.dd/bm/data/vivqa/annotations/dicts/answer2label.txt"

ViVQADataset.make_dataset_index(
    data_path=data_path,
    tokenizer_spm=tokenizer_spm,
    tokenizer_phobert=tokenizer_phobert,
    annotation_data_path=annotation_data_path,
    predefined_dict_path=predefined_dict_path
)