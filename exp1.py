from datasets import OpenViVQADataset
from transformers import XLMRobertaTokenizer

tokenizer = XLMRobertaTokenizer("/home/lenovo/exp1/beit3-model-n-ckpts/beit3-model/beit3.spm")

data_path = "/home/lenovo/exp1/data/openvivqa"
annotation_data_path = "/home/lenovo/exp1/data/openvivqa/vqa"

OpenViVQADataset.make_dataset_index(
    data_path=data_path,
    tokenizer=tokenizer,
    annotation_data_path=annotation_data_path,
)