# --------------------------------------------------------
# Image as a Foreign Language: BEiT Pretraining for Vision and Vision-Language Tasks (https://arxiv.org/abs/2208.10442)
# Github source: https://github.com/microsoft/unilm/tree/master/beit3
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------'

import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
import sys
import copy
import re

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

import torch
from transformers import AutoModel, AutoTokenizer

import loralib as lora

def verify_model_weights(model, epoch_str=""):
    """Prints checksums of key model weights for debugging."""
    print(f"\n--- Verifying weights for epoch: {epoch_str} ---")
    try:
        # 1. A LoRA weight (should be trained)
        lora_weight = model.beit3.encoder.layers[0].self_attn.q_proj.A.lora_A
        print(f"LORA Weight Checksum (q_proj.A.lora_A): {torch.sum(lora_weight).item()}")

        # 2. An original, frozen weight (should be from the --finetune checkpoint)
        original_weight = model.beit3.encoder.layers[0].self_attn.q_proj.A.weight
        print(f"Base Weight Checksum (q_proj.A.weight): {torch.sum(original_weight).item()}")

        # 3. The transplanted PhoBERT embedding (should be trained)
        phobert_embed = model.beit3.text_embed.weight
        print(f"PhoBERT Embed Checksum: {torch.sum(phobert_embed).item()}")

        # 4. The classification head (should be trained)
        head_weight = model.head[3].weight
        print(f"Head Weight Checksum: {torch.sum(head_weight).item()}")

    except AttributeError as e:
        print(f"Could not access a weight for verification: {e}")
    print("--- Verification complete ---\n")

def debug_data_batch(data_loader, batch_name=""):
    """Pulls one batch and prints its tensor statistics for debugging."""
    print(f"\n--- Debugging one batch from: {batch_name} ---")
    try:
        # Get the first batch from the data loader
        data = next(iter(data_loader))

        # Check language tokens
        lang_tokens = data['language_tokens']
        print(f"Language Tokens Shape: {lang_tokens.shape}")
        print(f"Language Tokens (first sample, 15 tokens): \n{lang_tokens[0, :15]}")

        # Check image tensor stats
        image = data['image']
        print(f"Image Shape: {image.shape}")
        print(f"Image Mean: {image.mean().item():.4f}")
        print(f"Image Std Dev: {image.std().item():.4f}")

    except Exception as e:
        print(f"Error during data batch debugging: {e}")
    print("--- Debug complete ---\n")


def get_args():
    parser = argparse.ArgumentParser('BEiT fine-tuning and evaluation script for image classification', add_help=False)

    # Model parameters
    parser.add_argument('--model', default='beit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--task', type=str, required=True, 
                        choices=['nlvr2', 'vqav2', 'flickr30k', 'coco_retrieval', 'coco_captioning', 'nocaps', 'imagenet', 'openvivqa', 'vivqa'], 
                        help='Name of task to fine-tuning')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--checkpoint_activations', action='store_true', default=None, 
                        help='Enable checkpointing to save your memory.')
    parser.add_argument('--sentencepiece_model', type=str, required=True, 
                        help='Sentencepiece model path for the pretrained model.')
    parser.add_argument('--vocab_size', type=int, default=64010)
    parser.add_argument('--num_max_bpe_tokens', type=int, default=64)

    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=[0.9, 0.999], type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: 0.9, 0.999, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--layer_decay', type=float, default=0.9)
    parser.add_argument('--task_head_lr_weight', type=float, default=0)

    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--eval_batch_size', default=None, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=5, type=int)

    # Augmentation parameters
    parser.add_argument('--randaug', action='store_true', default=False)
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # parameter for dump predictions (VQA, COCO captioning, NoCaps)
    parser.add_argument('--task_cache_path', default=None, type=str)

    # parameter for imagenet finetuning
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    
    # augmentation parameters for imagenet finetuning
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # evaluation parameters for imagenet
    parser.add_argument('--crop_pct', type=float, default=None)

    # random Erase params for imagenet finetuning
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # parameter for captioning finetuning
    parser.add_argument('--captioning_mask_prob', type=float, default=0.6)
    parser.add_argument('--drop_worst_ratio', type=float, default=0.2)
    parser.add_argument('--drop_worst_after', type=int, default=12000)
    parser.add_argument('--num_beams', type=int, default=3)
    parser.add_argument('--length_penalty', type=float, default=0.6)

    # label smoothing for imagenet and captioning
    parser.add_argument('--label_smoothing', type=float, default=0.1)

    # deepspeed parameters
    parser.add_argument('--enable_deepspeed', action='store_true', default=False)
    parser.add_argument('--initial_scale_power', type=int, default=16)
    parser.add_argument('--zero_stage', default=0, type=int,
                        help='ZeRO optimizer stage (default: 0)')

    # For Phobert Vivqa experiments
    parser.add_argument('--phobert', action='store_true', default=False,
                        help="Specify whether replacing the phobert's tokenizer and embedding layers or not")
    parser.add_argument('--patience', type=int, default=5,
                        help='Number of training epochs without improvements.')
    parser.add_argument('--early_stopping', type=str, default='val_score',
                        help="Determine the early stopping criteria. Supports val_score and val_loss")


    # --- LORA ---
    parser.add_argument('--enable_lora', action='store_true', default=False,
                        help='Enable LoRA for parameter-efficient finetuning.')
    parser.add_argument('--lora_r', type=int, default=8,
                        help='LoRA rank.')
    parser.add_argument('--lora_alpha', type=int, default=16,
                        help='LoRA alpha scaling factor.')
    parser.add_argument('--lora_dropout', type=float, default=0.05,
                        help='LoRA dropout.')
    parser.add_argument('--lora_target_modules', type=str, nargs='+',
                        default=['q_proj', 'v_proj'],
                        help='List of module names to apply LoRA to.')
    
    known_args, _ = parser.parse_known_args()

    if known_args.enable_deepspeed:
        try:
            import deepspeed
            from deepspeed import DeepSpeedConfig
            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except Exception as e:
            print("Failed to import deepspeed:")
            print(e)
            import traceback
            traceback.print_exc()
            exit(1)
        # except:
        #     print("Please 'pip install deepspeed==0.4.0'")
        #     exit(0)
    else:
        ds_init = None

    return parser.parse_args(), ds_init

def log_model_arch(args, model, config_name):
    with open(args.output_dir + "/" + config_name + ".txt", "w") as f:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                f.write(f"Frozen {param.shape}: {name}\n")
            else:
                f.write(f"Trainable {param.shape}: {name}\n")

        total_params = sum(p.numel() for p in model.parameters())
        frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

        f.write(f"\n\nTotal params: {total_params}\n")
        f.write(f"Frozen params: {frozen_params}\n")
        f.write(f"Trainable params: {n_parameters}\n")

        if args.enable_lora:
            f.write(f"\n\n LoRA SETTINGS\n")
            f.write(f"\n\n - Rank: {args.lora_r}\n")
            f.write(f"\n\n - Alpha: {args.lora_alpha}\n")
            f.write(f"\n\n - Dropout: {args.lora_dropout}\n")
            f.write(f"\n\n - Target Modeuls: {args.lora_target_modules}\n")


def main(args, ds_init):
    utils.init_distributed_mode(args)

    if ds_init is not None:
        utils.create_ds_config(args)

    if args.task_cache_path is None:
        args.task_cache_path = args.output_dir

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    if utils.get_rank() == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    phobert_model = None
    phobert_tokenizer = None
    if args.task == 'vivqa' and args.phobert:
        phobert_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2") # using PhoBERT tokenizer
        phobert_model = AutoModel.from_pretrained("vinai/phobert-base-v2") # Get word_embedding from phobert to replace text_embed

    data_loader_train, data_loader_val, data_loader_train_eval = create_downstream_dataset(args, phobert_tokenizer=phobert_tokenizer)

    if not args.model.endswith(args.task):
        if args.task in ("flickr30k", "coco_retrieval"):
            model_config = "%s_retrieval" % args.model
        elif args.task in ("coco_captioning", "nocaps"):
            model_config = "%s_captioning" % args.model
        elif args.task in ("imagenet"):
            model_config = "%s_imageclassification" % args.model
        elif args.task in ("openvivqa"):
            model_config = args.model
        else:
            model_config = "%s_%s" % (args.model, args.task)
    else:
        model_config = args.model
    print("model_config = %s" % model_config)

    # --- 1. CREATE THE BASE MODEL ---
    model = create_model(
        model_config,
        pretrained=False,
        drop_path_rate=args.drop_path,
        vocab_size=args.vocab_size,
        checkpoint_activations=args.checkpoint_activations,
    )

    # --- 2. LOAD ORIGINAL PRE-TRAINED WEIGHTS (if starting a new run) ---
    # This is for the very first run, to load the base BEiT-3 weights.
    if args.finetune:
        utils.load_model_and_may_interpolate(args.finetune, model, args.model_key, args.model_prefix)

    log_model_arch(args, model, "base_BEiT3")

    # --- 3. INJECT LoRA AND SWAP EMBEDDINGS ---
    # --- LORA-START ---
    # if args.enable_lora:
    #     if lora is None:
    #         raise RuntimeError("loralib is not installed, cannot enable LoRA.")
    #     print(f"Applying LoRA with r={args.lora_r}, alpha={args.lora_alpha}")
    #     print(f"Target modules: {args.lora_target_modules}")
        
    #     # This regex will match the parent MultiwayNetwork module names (e.g., "q_proj", "v_proj")
    #     pattern = r".*\.(" + "|".join(args.lora_target_modules) + r")$"
        
    #     replaced_count = 0
    #     # Iterate through the model's modules to find the MultiwayNetworks
    #     for name, module in model.named_modules():
    #         if re.match(pattern, name):
    #             # Check if this module is a MultiwayNetwork containing Linear layers 'A' and 'B'
    #             if hasattr(module, 'A') and hasattr(module, 'B') and \
    #             isinstance(module.A, torch.nn.Linear) and isinstance(module.B, torch.nn.Linear):
                    
    #                 # --- Replace the internal 'A' linear layer ---
    #                 old_linear_A = module.A
    #                 new_linear_A = lora.Linear(
    #                     in_features=old_linear_A.in_features,
    #                     out_features=old_linear_A.out_features,
    #                     bias=old_linear_A.bias is not None,
    #                     r=args.lora_r,
    #                     lora_alpha=args.lora_alpha,
    #                     lora_dropout=args.lora_dropout,
    #                 )
    #                 module.A = new_linear_A # Replace in-place
                    
    #                 # --- Replace the internal 'B' linear layer ---
    #                 old_linear_B = module.B
    #                 new_linear_B = lora.Linear(
    #                     in_features=old_linear_B.in_features,
    #                     out_features=old_linear_B.out_features,
    #                     bias=old_linear_B.bias is not None,
    #                     r=args.lora_r,
    #                     lora_alpha=args.lora_alpha,
    #                     lora_dropout=args.lora_dropout,
    #                 )
    #                 module.B = new_linear_B # Replace in-place
                    
    #                 # We replaced two linear layers (A and B) for each target
    #                 replaced_count += 2

    #     if replaced_count == 0:
    #         raise ValueError(
    #             f"Could not find any 'MultiwayNetwork' modules with Linear submodules 'A' and 'B' "
    #             f"matching the target patterns: {args.lora_target_modules}. "
    #             "This script is now specifically tailored for BEiT-3's Multiway attention structure."
    #         )

    #     print(f"Successfully replaced {replaced_count} internal linear layers with LoRA layers.")

    #     log_model_arch(args, model, "BEiT-3-LoRA")
    # # --- LORA-END ---

    # --- LORA-START ---
    if args.enable_lora:
        if lora is None:
            raise RuntimeError("loralib is not installed, cannot enable LoRA.")
        print(f"Applying LoRA with r={args.lora_r}, alpha={args.lora_alpha}")
        print(f"Target modules: {args.lora_target_modules}")
        
        # Find all nn.Linear layers in the model that should be replaced
        modules_to_replace = []
        for name, module in model.named_modules():
            # The name of a linear layer submodule could be, e.g., 'beit3.encoder.layers.0.self_attn.q_proj.A'
            # We check if the target keyword (e.g., 'q_proj') is part of its full name.
            if isinstance(module, torch.nn.Linear) and any(target in name for target in args.lora_target_modules):
                # To replace the submodule, we need to get its parent module and its name (the last part of the path)
                path_parts = name.split('.')
                parent_name = '.'.join(path_parts[:-1])
                child_name = path_parts[-1]
                parent_module = model.get_submodule(parent_name)
                
                modules_to_replace.append((parent_module, child_name, module))

        if not modules_to_replace:
            raise ValueError(
                f"Could not find any torch.nn.Linear modules whose names contain "
                f"the target keywords: {args.lora_target_modules}."
            )

        # Now, perform the actual replacement
        for parent_module, child_name, old_linear in modules_to_replace:
            new_linear = lora.Linear(
                in_features=old_linear.in_features,
                out_features=old_linear.out_features,
                bias=old_linear.bias is not None,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
            )
            # Set the new lora.Linear layer on the parent module
            setattr(parent_module, child_name, new_linear)
            
        print(f"Successfully replaced {len(modules_to_replace)} internal linear layers with LoRA layers.")
        log_model_arch(args, model, "BEiT-3-LoRA")
    # --- LORA-END ---

    if args.enable_lora:
        print("Marking only LoRA parameters as trainable.")
        lora.mark_only_lora_as_trainable(model)

    if args.task == 'vivqa' and args.phobert:

        new_layer = phobert_model.embeddings.word_embeddings
        # new_layer = phobert_model.embeddings
        # Replace tokenizer
        print("Replacing original beit3 text_embed w/ PhoBERT's...\n")

        model.beit3.text_embed = new_layer

    # --- 5. SET TRAINABLE PARAMETERS ---
    for param in model.beit3.text_embed.parameters():
        param.requires_grad = True
    for param in model.head.parameters():
        param.requires_grad = True
    for param in model.pooler.parameters():
        param.requires_grad = True
    for param in model.beit3.vision_embed.parameters():
        param.requires_grad = True

    # Check the frozen parameters
    if args.phobert:
        log_model_arch(args, model, "BEiT-3-LoRA-XPho")
    else:
        log_model_arch(args, model, "BEiT-3-XPho")

    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)

    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(data_loader_train.dataset) // total_batch_size
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(data_loader_train.dataset))
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)

    # --- LORA-START ---
    # Layer decay is not meaningful for LoRA, as we are not fine-tuning the original layers.
    # Disable it to avoid potential conflicts and simplify the optimizer setup.
    if args.enable_lora:
        print("LoRA is enabled. Disabling Layer Decay.")
        assigner = None
    else:
        num_layers = model_without_ddp.get_num_layers()
        if args.layer_decay < 1.0:
            lrs = list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2))
            assigner = LayerDecayValueAssigner(lrs)
        elif args.task_head_lr_weight > 1:
            assigner = LayerDecayValueAssigner([1.0, args.task_head_lr_weight], scale_handler=get_is_head_flag_for_vit)
        else:
            assigner = None
    # --- LORA-END ---

    # if assigner is not None:
    #     print("Assigned values = %s" % str(assigner.values))

    skip_weight_decay_list = model.no_weight_decay()

    if args.distributed:
        torch.distributed.barrier()
    if args.enable_deepspeed:
        loss_scaler = None
        optimizer_params = get_parameter_groups(
            model, args.weight_decay, skip_weight_decay_list,
            assigner.get_layer_id if assigner is not None else None,
            assigner.get_scale if assigner is not None else None)
        model, optimizer, _, _ = ds_init(
            args=args, model=model, model_parameters=optimizer_params,
            dist_init_required=not args.distributed,
        )

        print("model.gradient_accumulation_steps() = %d" % model.gradient_accumulation_steps())
        assert model.gradient_accumulation_steps() == args.update_freq
    else:
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            model._set_static_graph()
            model_without_ddp = model.module

        optimizer = create_optimizer(
            args, model_without_ddp, skip_list=skip_weight_decay_list,
            get_num_layer=assigner.get_layer_id if assigner is not None else None, 
            get_layer_scale=assigner.get_scale if assigner is not None else None)
        loss_scaler = NativeScaler()

    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

    task_handler = get_handler(args)

    # mixup for imagenet
    mixup_fn = None
    if args.task in ["imagenet", "in1k"]:
        mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
        if mixup_active:
            print("Mixup is activated!")
            mixup_fn = Mixup(
                mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                label_smoothing=args.label_smoothing, num_classes=args.nb_classes)
   

    if args.eval:
        data_loader_test = create_downstream_dataset(args, is_eval=True, phobert_tokenizer=phobert_tokenizer)

        if args.task == "vivqa":
            result, _ = evaluate(data_loader_test, model, device, task_handler)
            utils.dump_predictions(args, result, "vivqa_test")

            import torch.distributed as dist
            if args.distributed:
                dist.destroy_process_group()
            exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    max_accuracy, min_val_loss, epochs_without_improvements = utils.get_best_meters_and_no_improvement_streak(args.output_dir, args.early_stopping)
    print(f"This section's initial max_accuracy: {max_accuracy}")
    print(f"This section's initial epochs_without_improvements: {epochs_without_improvements}")
    patience = args.patience

    data_loader_test = create_downstream_dataset(args, is_eval=True, phobert_tokenizer=phobert_tokenizer)

    for epoch in range(args.start_epoch, args.epochs):

        epoch_start_time = time.time()

        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        train_stats = train_one_epoch(
            model, data_loader_train, optimizer, device, task_handler, epoch, 
            epoch * num_training_steps_per_epoch, lr_schedule_values, loss_scaler, 
            args.clip_grad, args.update_freq, model_ema, log_writer, args.task, mixup_fn,
        )

        # Get epoch's train_score
        # train_eval_stats, _ = evaluate(data_loader_train_eval, model, device, safe_eval_handler)
        # train_stats["score"] = train_eval_stats["score"]  

        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema)
                
        if data_loader_val is not None:
            verify_model_weights(model, epoch_str=str(epoch))     

            if args.task not in ["coco_captioning", "nocaps"]:
                test_stats, task_key = evaluate(data_loader_val, model, device, task_handler)
            else:
                predictions, _ = evaluate(data_loader_val, model, device, task_handler)
                prediction_file = utils.dump_predictions(args, predictions, f"{args.task}_val_e{epoch}")
                result_file = os.path.join(args.output_dir, f"{args.task}_result_val_e{epoch}.json")
                task_key = "CIDEr"
                if utils.is_main_process():
                    test_stats = utils.coco_caption_eval(args.output_dir, prediction_file, "{}_val".format(args.task))
                    utils.write_result_to_jsonl(test_stats, result_file)
                torch.distributed.barrier()
                if not utils.is_main_process():
                    test_stats = utils.read_result_from_jsonl(result_file)

            print(f"Performance of the network on the {len(data_loader_val.dataset)} val images: {test_stats[task_key]:.1f}%")
            if max_accuracy < test_stats[task_key]:
                max_accuracy = test_stats[task_key]
                epochs_without_improvements = 0 # Reset patience

                result, _ = evaluate(data_loader_test, model, device, task_handler)
                utils.dump_predictions(args, result, "vivqa_test")

                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)    
                
            else:
                epochs_without_improvements += 1

            print(f'Max performance: {max_accuracy:.2f}%')
            if log_writer is not None:
                log_writer.update(acc=test_stats[task_key], head="perf", step=epoch)
            
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'val_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters,
                        'time': time.time() - epoch_start_time}
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters,
                         'time': time.time() - epoch_start_time}

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        if epochs_without_improvements >= patience:
            print(f"No improvement in {patience} consecutive epochs. Early stopping.")
            break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))



if __name__ == '__main__':
    opts, ds_init = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts, ds_init)