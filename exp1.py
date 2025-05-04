from datasets import OpenViVQADataset, ViVQADataset
from transformers import XLMRobertaTokenizer

# tokenizer = XLMRobertaTokenizer("text-tokenizers/beit3.spm")
tokenizer = XLMRobertaTokenizer("/home/lenovo/exp1/beit3-model-n-ckpts/beit3-model/beit3.spm") # for gcp vm

VIVIQA_DATASET_PATH = "data/vivqa"

data_path = VIVIQA_DATASET_PATH
annotation_data_path = VIVIQA_DATASET_PATH + "/vqa"

ViVQADataset.make_dataset_index(
    data_path=data_path,
    tokenizer=tokenizer,
    annotation_data_path=annotation_data_path,
)