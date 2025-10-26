import argparse
import time
import numpy as np
import os
import sys
import json
from pathlib import Path
import copy
import math

from tqdm import tqdm

from torch.utils.data import DataLoader
import torch
from torch.optim import AdamW
from transformers import PhobertTokenizer, BitsAndBytesConfig 
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from paligemma import get_vivqa_paligemma, get_vivqa_paligemma_phobert_with_adapter, get_vivqa_paligemma_phobert_with_adapter_dev, PaligemmaForVQAClassification
from paligemma_dataset import create_paligemma_datasets
from my_utils import TrainingF1Score, my_dump_predictions, my_save_model, my_auto_resume
from paligemma_engine_for_finetuning import PaligemmaHandler, paligemma_evaluate

from GemmaFitPhobertTokenizer import GemmaFitPhobertTokenizer

import utils
from optim_factory import create_optimizer, LayerDecayValueAssigner, get_is_head_flag_for_vit


def get_args():
    parser = argparse.ArgumentParser('Paligemma fine-tuning and evaluation script for image classification', add_help=False)

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
    
    parser.add_argument('--lr_sched_type', type=str, default='linear', choices=['cos', 'linear', 'none'],
                        help='Learning rate scheduler type (default: linear)')

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--eval_batch_size', default=None, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=5, type=int)

     # Dataset parameters
    parser.add_argument('--data_path', default='/root/projects/exp1/data/vivqa', type=str,
                        help='dataset path')
    parser.add_argument('--answer2label', default=None, type=str,
                        help='answer2label.txt path')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--checkpoint_dir', default='',
                        help='path where to save checkpoints') # Move to disk due to reduced max storage from server admin
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
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

    # parameter for dump predictions (VQA, COCO captioning, NoCaps)
    parser.add_argument('--task_cache_path', default=None, type=str)

    # For Phobert Vivqa experiments
    parser.add_argument('--phobert', action='store_true', default=False,
                        help="Specify whether replacing the phobert's tokenizer and embedding layers or not")
    parser.add_argument('--patience', type=int, default=5,
                        help='Number of training epochs without improvements.')
    parser.add_argument('--early_stopping', type=str, default='val_score',
                        help="Determine the early stopping criteria. Supports val_score and val_loss")
    parser.add_argument('--staged_training', action='store_true', default=False,
                        help="Enable 3-stage gradual unfreezing training.")
    parser.add_argument('--freeze_embed_tokens', action='store_true', default=False,
                        help="Freeze Paligemma language model's embed_tokens.")
    
    # Custom resume args
    parser.add_argument('--no_resume_optimizer', action="store_true", default=False,
                        help="This parameter prevents auto loading optimizer from checkpoint")
    

    known_args, _ = parser.parse_known_args()
    
    return parser.parse_args()

# <<< START HELPER FUNCTIONS FOR STAGED TRAINING >>>
def log_model_architecture(model, output_dir, suffix):
    """Saves a file detailing which parameters are trainable/frozen for a given stage."""
    filepath = os.path.join(output_dir, f"Model_Architecture_{suffix}.txt")

    with open(filepath, "w") as f:
        f.write(f"--- Model Architecture {suffix}: ---\n\n")
        
        trainable_params = []
        frozen_params = []

        total_trainable = 0
        total_frozen = 0
        
        for name, param in model.named_parameters():
            num_params = param.numel()
            # Format: Parameter Name | Dimensions | Total Elements
            param_info = f"{name:<120} | shape: {str(param.shape):<25} | params: {num_params}"
            
            if param.requires_grad:
                trainable_params.append(f"Trainable: {param_info}")
                total_trainable += num_params
            else:
                frozen_params.append(f"Frozen:    {param_info}")
                total_frozen += num_params

        f.write("--- Trainable Parameters ---\n")
        for line in trainable_params:
            f.write(f"{line}\n")
            
        f.write("\n--- Frozen Parameters ---\n")
        for line in frozen_params:
            f.write(f"{line}\n")
            
        f.write("\n--- Summary ---\n")
        total_params = total_trainable + total_frozen
        f.write(f"Total Parameter Blocks: {len(trainable_params) + len(frozen_params)}\n")
        f.write(f"Trainable Parameter Blocks: {len(trainable_params)}\n")
        f.write(f"Frozen Parameter Blocks: {len(frozen_params)}\n\n")

        f.write(f"Total params:     {total_params:,}\n")
        f.write(f"Trainable params: {total_trainable:,} ({100 * total_trainable / total_params:.4f}%)\n")
        f.write(f"Frozen params:    {total_frozen:,}\n")



def log_training_config(output_dir, stage_config):
    filepath = os.path.join(output_dir, f"Stage_Config:{stage_config['name']}.txt")
    with open(filepath, "w") as f:
        f.write("\n--- Stage Config ---\n")
        json.dump(stage_config, f, ensure_ascii=False, indent=4)

def get_resume_stage_index(global_epoch, stages):
    print(f"{global_epoch}")
    cnt = 0
    for i, stage_config in enumerate(stages):
        cnt += stage_config["epochs"]
        if cnt > global_epoch:
            return i  
        
def log_paligemma_model_config(model: PaligemmaForVQAClassification, output_dir):
    outfile = os.path.join(output_dir, f"model_config.txt")

     # --- Define a custom JSON encoder to handle torch.dtype ---
    class DtypeJSONEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, torch.dtype):
                return str(obj)  # Convert dtype to its string representation
            return super().default(obj)
        
    with open(outfile, "w") as f:
        f.write(f"--- Paligemma Language Model Config ---\n")
        # --- Use the custom encoder in the json.dump call ---
        json.dump(model.paligemma.language_model.config.to_dict(), f, indent=4, cls=DtypeJSONEncoder)
        
        f.write("\n\n--- Paligemma Vision Tower Config ---\n\n")
        json.dump(model.paligemma.vision_tower.config.to_dict(), f, indent=4, cls=DtypeJSONEncoder)

        f.write("\n\n--- Paligemma Model Config ---\n\n")
        json.dump(model.paligemma.config.to_dict(), f, indent=4, cls=DtypeJSONEncoder)

        f.write("\n\n--- Classifier head Dropout ---\n\n")
        f.write(f"nn.DropOut: {model.dropout.p}")

        if hasattr(model, 'phobert_embedding_adapter') and model.phobert_embedding_adapter is not None:
            f.write("\n\n--- Phobert Adapter Shape ---\n\n")
            for name, params in model.phobert_embedding_adapter.named_parameters():
                f.write(f"{name}: {params.shape}\n")

def setup_model_for_stage(model: PaligemmaForVQAClassification, stage_config):
    """Freezes/unfreezes BLIP model parameters based on the stage configuration."""
    print(f"--- Configuring model for stage: {stage_config['name']} ---")

    # This logic assumes a "start trainable and freeze some" approach
    # which is simpler. First, set everything to trainable.
    for param in model.parameters():
        if param.dtype in [torch.float, torch.float16, torch.bfloat16]:
            param.requires_grad = True

    if stage_config.get('freeze_paligemma_vision_tower', True):
        print("Paligemma's vision tower is FROZEN.")
        for param in model.paligemma.vision_tower.parameters():
            param.requires_grad = False

    if stage_config.get('freeze_paligemma_language_model', True):
        print("Paligemma's language model (excluding embed-tokens) is FROZEN.")
        for name, param in model.paligemma.language_model.named_parameters():
            if 'embed-tokens' not in name:
                param.requires_grad = False

    if stage_config.get('freeze_embed_tokens', True):
        print("PhoBERT transplanted word embeddings (paligemma's embed_tokens) are FROZEN.")
        for name, param in model.paligemma.language_model.embed_tokens.named_parameters():
            if "word_embeddings" not in name:
                param.requires_grad = False

    if stage_config.get('freeze_classifier', True):
        print("Classifier head is FROZEN.")
        for param in model.classifier.parameters():
            param.requires_grad = False

    if model.phobert_embedding_adapter and not stage_config.get('freeze_phobert_adapter', True):
        print("Unfreezing: PhoBERT Embedding Adapter")
        for param in model.phobert_embedding_adapter.parameters():
            param.requires_grad = True

def create_stage_optimizer(model: PaligemmaForVQAClassification, stage_config, args):
    """
    Creates a staged optimizer for the Paligemma model, correctly handling both
    differential learning rates and selective weight decay.
    """
    print(f"--- Creating optimizer for stage: {stage_config['name']} ---")
    
    no_decay_list = model.no_weight_decay()
    print(f"Substrings for no weight decay: {no_decay_list}")

    # This will be the final list of parameter groups passed to the optimizer
    optimizer_groups = []
    
    # Define the components of your model and their learning rates
    lrs = stage_config['lrs']
    components = {
        'paligemma_vision_tower': {'params': [], 'lr': lrs['paligemma_vision_tower']},
        'paligemma_language_model': {'params': [], 'lr': lrs['paligemma_language_model']},
        'embed_tokens': {'params': [], 'lr': lrs['embed_tokens']},
        'classifier': {'params': [], 'lr': lrs['classifier']},
    }

    if hasattr(model, 'phobert_embedding_adapter') and model.phobert_embedding_adapter is not None:
        components['phobert_adapter'] = {
            'params': [],
            'lr': lrs['phobert_adapter']
        }

    # 1. First, assign each parameter to its component group
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if 'vision_tower' in name:
            components['paligemma_vision_tower']['params'].append((name, param))
        elif 'embed_tokens' in name:
            components['embed_tokens']['params'].append((name, param))
        elif 'language_model' in name:
            components['paligemma_language_model']['params'].append((name, param))
        elif 'classifier' in name:
            components['classifier']['params'].append((name, param))
        elif 'phobert_embedding_adapter' in name:
            components['phobert_adapter']['params'].append((name, param))

    # 2. For each component, create final groups with and without weight decay
    for group_name, component_data in components.items():
        if not component_data['params']:
            continue # Skip if a component is completely frozen

        decay_params = []
        no_decay_params = []
        
        # Separate the component's parameters into decay/no_decay lists
        for name, param in component_data['params']:
            if name.endswith(".bias") or "LayerNorm.weight" in name or any(nd in name for nd in no_decay_list):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        # Create a group for parameters with weight decay
        if decay_params:
            optimizer_groups.append({
                'params': decay_params,
                'lr': component_data['lr'],
                'weight_decay': args.weight_decay,
                'group_base_lr': component_data['lr']
            })
            print(f"  - Group '{group_name}_decay' -> lr: {component_data['lr']:.1e}, wd: {args.weight_decay}")

        # Create a group for parameters without weight decay
        if no_decay_params:
            optimizer_groups.append({
                'params': no_decay_params,
                'lr': component_data['lr'],
                'weight_decay': 0.0,
                'group_base_lr': component_data['lr']
            })
            print(f"  - Group '{group_name}_no_decay' -> lr: {component_data['lr']:.1e}, wd: 0.0")

    # return AdamW(optimizer_groups, eps=args.opt_eps, betas=args.opt_betas)

    print("Using 8-bit AdamW optimizer from bitsandbytes.")
    # Replace the original AdamW with the 8-bit version
    return bnb.optim.AdamW8bit(optimizer_groups, eps=args.opt_eps, betas=args.opt_betas)

def train_stage(stage_config, global_epoch_start, model,
                data_loader_train, dataset_train, 
                data_loader_val, dataset_val,
                optimizer, device, task_handler: PaligemmaForVQAClassification, args,
                # Pass metric trackers by reference (as a dict) to modify them
                metric_trackers,
                stage_index):

    stage_epochs = stage_config['epochs']
    print(f"\nTraining for {stage_epochs} epochs...")

    # Re-create the LR scheduler for this specific stage
    total_batch_size = args.batch_size * args.update_freq
    num_training_steps_per_epoch = len(data_loader_train.dataset) // total_batch_size
    
    # Use the stage's main LR as the base for the scheduler
    stage_base_lr = max(stage_config['lrs'].values())

    if (args.lr_sched_type != 'none'):
        lr_schedule_values = utils.cosine_scheduler(
            stage_base_lr, args.min_lr, stage_epochs, num_training_steps_per_epoch,
            warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
            sched_type=args.lr_sched_type,
        )
    else:
        lr_schedule_values = None
    
    for epoch_in_stage in range(stage_epochs):
        epoch = global_epoch_start + epoch_in_stage
        epoch_start_time = time.time()

        print(f"\nGlobal Epoch {epoch} (Stage '{stage_config['name']}', Epoch {epoch_in_stage + 1}/{stage_epochs})")

        total_loss = 0.0
        total_score = 0.0
        num_batches = 0

        min_lr = 10.
        max_lr = 0.

        for step, batch in enumerate(tqdm(data_loader_train, desc=f"Epoch {epoch}", leave=False)):
            optimizer.zero_grad()

            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            paligemma_labels = batch["paligemma_labels"].to(device)
            labels = batch["labels"].to(device)
            qid = batch["qid"]

            # Scheduler step calculation needs to be relative to the stage
            step_in_epoch = step // args.update_freq
            global_step_in_stage = epoch_in_stage * num_training_steps_per_epoch + step_in_epoch

            if lr_schedule_values is not None and (epoch * len(data_loader_train) + step) % args.update_freq == 0:
                # Apply scheduler to each group
                for i, param_group in enumerate(optimizer.param_groups):
                    # The LR for a group is its base LR scaled by the scheduler value
                    group_base_lr = param_group['group_base_lr']
                    lr_scale = lr_schedule_values[global_step_in_stage] / stage_base_lr
                    param_group["lr"] = group_base_lr * lr_scale

            train_stats = task_handler.train_batch(
                model = model,
                input_ids = input_ids,
                pixel_values = pixel_values,
                attention_mask = attention_mask,
                token_type_ids = token_type_ids,
                paligemma_labels = paligemma_labels, # Pass to Paligemma
                labels = labels, # Use for our classification
                qid=None
            )
            
            loss = train_stats["loss"]
            batch_score = train_stats["score"]

            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()))
                sys.exit(1)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Accumulate for epoch loss
            total_loss += loss.item()
            total_score += batch_score.item()
            num_batches += 1

            for group in optimizer.param_groups:
                min_lr = min(min_lr, group["lr"])
                max_lr = max(max_lr, group["lr"])

            if (step + 1) % 10 == 0:
                running_avg_loss = total_loss / num_batches
                running_avg_score = total_score / num_batches
                print(f"Epoch {epoch} Step {step + 1}: loss: {running_avg_loss:.4f}, score {running_avg_score:.4f}, " +
                      f"min_lr: {min_lr:.6f}, max_lr: {max_lr:.6f}")
        
        average_loss = total_loss / num_batches
        average_score = total_score / num_batches
        epoch_training_time = time.time() - epoch_start_time
        print(f"Epoch {epoch} completed in {(epoch_training_time):.2f}s - Average Loss: {average_loss:.4f} - Average Score: {average_score:.4f}")

        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or (epoch + 1) == args.epochs:
                my_save_model(args=args, epoch=epoch, model=model, optimizer=optimizer)
                
        if data_loader_val is not None:
            val_stats = paligemma_evaluate(
                data_loader_val, model, dataset_val.processor.tokenizer, device, task_handler, return_preds=False
            )
            print(f"Performance of the network on the {len(data_loader_val)} val batches: {val_stats['score']:.1f}%")

            if metric_trackers['max_accuracy'] < val_stats['score']:
                metric_trackers['max_accuracy'] = val_stats['score']
                if args.output_dir and args.save_ckpt:
                    print(f"*** New best score! Saving model for epoch {epoch}. ***")
                    my_save_model(args=args, epoch=f"best-{epoch}", model=model, optimizer=optimizer)
            
            current_val_score = val_stats['score']
            current_val_loss = val_stats['loss']

            if args.early_stopping == "val_score":
                if metric_trackers['max_accuracy_stop'] < current_val_score:
                    metric_trackers['max_accuracy_stop'] = current_val_score
                    metric_trackers['epochs_without_improvements'] = 0 # Reset patience
                else:
                    metric_trackers['epochs_without_improvements'] += 1
                if metric_trackers['min_val_loss'] > current_val_loss:
                    metric_trackers['min_val_loss'] = current_val_loss
            elif args.early_stopping == "val_loss":
                if metric_trackers['min_val_loss'] > current_val_loss:
                    metric_trackers['min_val_loss'] = current_val_loss
                    metric_trackers['epochs_without_improvements'] = 0 # Reset patience
                else:
                    metric_trackers['epochs_without_improvements'] += 1
                if metric_trackers['max_accuracy_stop'] < current_val_score:
                    metric_trackers['max_accuracy_stop'] = current_val_score
            
            print(f"Current epochs without improvements: {metric_trackers['epochs_without_improvements']}")
            print(f'Max val score so far: {metric_trackers["max_accuracy"]:.2f}%')
            print(f'Min val loss so far: {metric_trackers["min_val_loss"]}')
            
            log_stats = {'train_loss': average_loss, 
                         'train_score': average_score, 
                         'train_min_lr': min_lr, 
                         'train_max_lr': max_lr,
                         **{f'val_{k}': v.item() if isinstance(v, torch.Tensor) else v for k, v in val_stats.items() if k != "prediction"},
                         'epoch': epoch, 
                         'n_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad), 
                         'time': epoch_training_time}
        else: # No validation set
            log_stats = {'train_loss': average_loss, 
                         'train_score': average_score, 
                         'train_min_lr': min_lr, 
                         'train_max_lr': max_lr,
                         'epoch': epoch, 
                         'n_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad), 
                         'time': epoch_training_time}

        if args.output_dir:
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        if metric_trackers['epochs_without_improvements'] >= args.patience: # and stage_index >= 2: # and average_score >= 85:
            print(f"No improvement in {args.patience} consecutive epochs. Early stopping stage.")
            return True # Signal to stop all training
    
    return False # Signal to continue to next stage

# <<< END HELPER FUNCTIONS FOR STAGED TRAINING >>>

def main(args):

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    if args.task_cache_path is None:
        args.task_cache_path = args.output_dir

    device = torch.device(args.device)

    dtype = torch.bfloat16

    phobert_tokenizer = None

    if args.phobert:
        phobert_tokenizer = GemmaFitPhobertTokenizer.from_pretrained("vinai/phobert-base-v2", use_fast=True)


    if args.phobert:
        # model = get_vivqa_paligemma_phobert(device=device, answer2label_path=args.answer2label).to(device, non_blocking=True, dtype=dtype)
        # model = get_vivqa_paligemma_phobert_with_adapter_dev(
        #     device=device, answer2label_path=args.answer2label
        # ).to(device, non_blocking=True, dtype=dtype)
        model = get_vivqa_paligemma_phobert_with_adapter(
            device=device, answer2label_path=args.answer2label
        ).to(device, non_blocking=True, dtype=dtype)
    else:
        model = get_vivqa_paligemma(device = device, answer2label_path=args.answer2label).to(device, non_blocking=True, dtype=dtype)

    log_model_architecture(model, args.output_dir, "BASE_PALIGEMMA")

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # --- ADD THIS DEBUGGING CODE ---
    # print("--- Finding available module names for LoRA ---")
    # for name, module in model.named_modules():
    #     if isinstance(module, torch.nn.Linear): # Optional: Filter for Linear layers
    #         print(name)
    # print("---------------------------------------------")

    # sys.exit(0)

    # --- Define the LoRA Configuration ---
    # Target modules for both vision and language models
    # These are common linear layers in transformer architectures
    lora_target_modules = (
        r"paligemma\.language_model\.layers\.\d+\."
        # r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"
        r"(self_attn\.(q_proj|v_proj))"
    )


    # lora_target_modules = ["q_proj", "v_proj"]

    config = LoraConfig(
        r=64,  # LoRA rank
        lora_alpha=128,  # LoRA scaling, convention is to set it to 2 * r.
        target_modules=lora_target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=['phobert_embedding_adapter', 'classifier']
    )

    # Wrap the model with PEFT
    model = get_peft_model(model, config)

    # Print the trainable parameters to verify
    # model.print_trainable_parameters()

    dataset_train, data_loader_train, dataset_val, data_loader_val = create_paligemma_datasets(
        args,
        phobert_tokenizer=phobert_tokenizer
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('Number of params:', n_parameters)

    task_handler = PaligemmaHandler()

    if args.eval:
        my_auto_resume(args, model=model, optimizer=None, device=device) # Load best checkpoint for eval
        dataset_test, data_loader_test = create_paligemma_datasets(
            args,
            is_eval = True,
            phobert_tokenizer = phobert_tokenizer 
        )
        result = paligemma_evaluate(
            data_loader=data_loader_test,
            model=model,
            tokenizer=dataset_test.processor.tokenizer,
            device=device,
            handler=task_handler,
            return_preds=True
        )
        my_dump_predictions(args, result["prediction"], "vivqa_paligemma_test")
        exit(0)

    start_time = time.time()

    log_paligemma_model_config(model, args.output_dir)

    if args.staged_training and args.phobert:
        
        print("--- Staged Training Enabled (2 Stages) ---")

        STAGES = [
            # Add stages when use staged_training
        ]
        
        global_epoch = args.start_epoch
        metric_trackers = {
            'max_accuracy': 0.0,
            'min_val_loss': float('inf'),
            'epochs_without_improvements': 0,
            'max_accuracy_stop': 0.0
        }

        metric_trackers['max_accuracy'], \
        metric_trackers['min_val_loss'], \
        metric_trackers['epochs_without_improvements'] = utils.get_best_meters_and_no_improvement_streak(
            args.output_dir, early_stopping_metric=args.early_stopping
        )
        metric_trackers["max_accuracy_stop"] = metric_trackers["max_accuracy"]
        print(f"Initial max_accuracy: {metric_trackers['max_accuracy']}")
        print(f"Initial min_val_loss: {metric_trackers['min_val_loss']}")

        resume_stage_index = -1
        if args.resume != '':
            print("Creating temporary old_optimizer.")
            old_optimizer = create_stage_optimizer(model, STAGES[0], args)
            model, old_optimizer, global_epoch = my_auto_resume(
                args, model, old_optimizer, device
            )

            resume_stage_index = get_resume_stage_index(global_epoch, STAGES)
            print(f"Resuming from Stage {resume_stage_index+1}")

        for i, stage_config in enumerate(STAGES):
            if args.resume != '' and resume_stage_index > i:
                continue
                    
            stage_config['name'] = f"Stage_{i+1}_{stage_config['name']}" # Add number to name
            setup_model_for_stage(model, stage_config)
            
            # log_model_architecture(model, args.output_dir, stage_config)
            
            if args.resume != '':
                optimizer = old_optimizer
            else:
                optimizer = create_stage_optimizer(model, stage_config, args)
            
            should_stop = train_stage(
                stage_config, global_epoch, model,
                data_loader_train, dataset_train,
                data_loader_val, dataset_val,
                optimizer, device, task_handler, args,
                metric_trackers=metric_trackers,
                stage_index = i
            )

            global_epoch += stage_config['epochs']
            if should_stop:
                print("Early stopping criteria met. Halting all training.")
                break

    else:
        # --- ORIGINAL TRAINING LOGIC ---
        print("--- Standard End-to-End Training Enabled ---")
        num_layers = model.get_num_layers()
        if args.layer_decay < 1.0:
            lrs = list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2))
            assigner = LayerDecayValueAssigner(lrs)
        elif args.task_head_lr_weight > 1:
            assigner = LayerDecayValueAssigner([1.0, args.task_head_lr_weight], scale_handler=get_is_head_flag_for_vit)
        else:
            assigner = None
        
        if assigner is not None:
            print("Assigned values = %s" % str(assigner.values))

        # Encapsulate the original loop logic into a single stage config
        stage_config = {
                'name': 'Standard End-to-End Training',
                'epochs': args.epochs,
                'freeze_paligemma_vision_tower': False,  # args.phobert,
                'freeze_paligemma_language_model': False,
                'freeze_embed_tokens': args.freeze_embed_tokens,
                'freeze_classifier': False,
                'freeze_phobert_adapter': False,
                'lrs': {
                    'paligemma_vision_tower': 2e-5,
                    'paligemma_language_model': 2e-5,
                    'classifier': 5e-5,
                    'embed_tokens': 1e-5, # <<< Use a very small LR
                    'phobert_adapter': 5e-5
                }
            }

        optimizer = create_stage_optimizer(model, stage_config, args)
        # setup_model_for_stage(model, stage_config)
        if args.resume != '':
            model, optimizer, args.start_epoch = my_auto_resume(args, model=model, optimizer=optimizer, device=device)

        log_model_architecture(model, args.output_dir, "PALIGEMMA_LORA") # Log architecture for standard training too
        log_training_config(args.output_dir, stage_config)

        metric_trackers = {
            'max_accuracy': 0.0,
            'min_val_loss': float('inf'),
            'epochs_without_improvements': 0,
            'max_accuracy_stop': 0.0
        }

        metric_trackers['max_accuracy'], \
        metric_trackers['min_val_loss'], \
        metric_trackers['epochs_without_improvements'] = utils.get_best_meters_and_no_improvement_streak(
            args.output_dir, early_stopping_metric=args.early_stopping
        )
        print(f"Initial max_accuracy: {metric_trackers['max_accuracy']}")
        print(f"Initial min_val_loss: {metric_trackers['min_val_loss']}")

        train_stage(
            stage_config, args.start_epoch, model,
            data_loader_train, dataset_train,
            data_loader_val, dataset_val,
            optimizer, device, task_handler, args,
            metric_trackers=metric_trackers,
            stage_index=0
        )

    # <<< END MODIFIED TRAINING LOGIC >>>

    total_time = time.time() - start_time
    total_time_str = str(time.strftime('%H hours, %M minutes, %S seconds', time.gmtime(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == "__main__":
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)