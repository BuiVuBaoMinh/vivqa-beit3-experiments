from datasets import OpenViVQADataset
from transformers import XLMRobertaTokenizer

tokenizer = XLMRobertaTokenizer("/root/projects/exp1/beit3.spm")

data_path = "data/openvivqa"
annotation_data_path = "data/openvivqa"

OpenViVQADataset.make_dataset_index(
    data_path=data_path,
    tokenizer=tokenizer,
    annotation_data_path=annotation_data_path,
)