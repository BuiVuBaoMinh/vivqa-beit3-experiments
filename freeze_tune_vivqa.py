import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os

from pathlib import Path

from timm.data.mixup import Mixup
from timm.models import create_model
from timm.utils import ModelEma
from optim_factory import create_optimizer, get_parameter_groups, \
    LayerDecayValueAssigner, get_is_head_flag_for_vit

from engine_for_finetuning import train_one_epoch, get_handler, evaluate
from datasets import create_downstream_dataset
from utils import NativeScalerWithGradNormCount as NativeScaler
import utils
import modeling_finetune


from transformers import AutoTokenizer, AutoModel

model_config = "beit3_base_patch16_480_vivqa"
drop_path = 0.1
vocab_size = 64010
checkpoint_activations = False

model =     model = create_model(
        model_config,
        pretrained=False,
        drop_path_rate=drop_path,
        vocab_size=vocab_size,
        checkpoint_activations=checkpoint_activations,
    )

phobert_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2", use_fast=False)
phobert_model = AutoModel.from_pretrained("vinai/phobert-base-v2")

print("Phobert token size: ", phobert_model.embeddings.word_embeddings)
print("Beit3 token size: ", model.beit3.text_embed)

model.beit3.text_embed = phobert_model.embeddings.word_embeddings

# Print the class of phobert tokenizer
print("Phobert tokenizer class: ", type(phobert_tokenizer))
print("Beit3 tokenizer class: ", type(model.beit3.text_embed))

# Example: freeze all except text embed and text (B) experts
for name, param in model.named_parameters():
    # Freeze vision-embedding and A-expert parameters (do not train)
    if name.startswith("beit3.vision_embed") or ".A." in name:
        param.requires_grad = False
    # Allow training of text embedding and B-expert parameters
    elif name.startswith("beit3.text_embed") or ".B." in name:
        param.requires_grad = True
    else:
        # For any other parameters (e.g. classification head), you may freeze or unfreeze as needed
        param.requires_grad = False

# Check the frozen parameters
for name, param in model.named_parameters():
    if not param.requires_grad:
        print(f"Frozen parameter: {name}")
    else:
        print(f"Trainable parameter: {name}")