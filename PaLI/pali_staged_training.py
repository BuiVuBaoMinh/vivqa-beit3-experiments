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
from transformers import AutoModel, AutoTokenizer, PhobertTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pali_dataset import ViVQAPaLIDataset, create_pali_datasets
from pali import PaLI, PaLI_PhoBERT, PaLI_Classification, PaLI_Classification_PhoBERT
from pali_engine_for_finetuning import PaLIHandler, PaLIClassificationHandler, pali_evaluate
from pali_utils import pali_dump_predictions, pali_save_model, pali_auto_resume

import utils
from optim_factory import create_optimizer, get_parameter_groups, \
    LayerDecayValueAssigner, get_is_head_flag_for_vit


def get_args():
    parser = argparse.ArgumentParser('PaLI fine-tuning and evaluation script for image classification', add_help=False)

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
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true') # For PaLI, resumes the last epoch
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
    parser.add_argument('--eval_num_beams', type=int, default=1,
                        help="Number of beam used for beam search in PaLI.generate()")
    parser.add_argument('--staged_training', action='store_true', default=False,
                        help="Enable 3-stage gradual unfreezing training.")
    parser.add_argument('--pali_class', default="pali_generative", choices = ["pali_classification", "pali_generative"],
                        help="Enable 3-stage gradual unfreezing training.")

    # Custom resume args
    parser.add_argument('--no_resume_optimizer', action="store_true", default=False,
                        help="This parameter prevents auto loading optimizer from checkpoint")
    

    known_args, _ = parser.parse_known_args()

    ds_init = None
    
    return parser.parse_args(), ds_init


# <<< START HELPER FUNCTIONS FOR STAGED TRAINING >>>

def get_resume_stage_index(global_epoch, stages):
    print(f"{global_epoch}")
    cnt = 0
    for i, stage_config in enumerate(stages):
        cnt += stage_config["epochs"]
        if cnt > global_epoch:
            return i            

def log_model_architecture(model, output_dir, stage_config):
    """Saves a file detailing which parameters are trainable/frozen for a given stage."""
    filepath = os.path.join(output_dir, f"Model_Architecture_{stage_config['name']}.txt")
    print(f"Logging model architecture for stage '{stage_config['name']}' to {filepath}")

    with open(filepath, "w") as f:
        f.write(f"--- Model Architecture for Stage: {stage_config['name']} ---\n\n")
        
        trainable_params = []
        frozen_params = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_params.append(name)
            else:
                frozen_params.append(name)

        f.write("--- Trainable Parameters ---\n")
        for name in trainable_params:
            f.write(f"Trainable: {name}\n")
            
        f.write("\n--- Frozen Parameters ---\n")
        for name in frozen_params:
            f.write(f"Frozen: {name}\n")
            
        f.write("\n--- Summary ---\n")
        total_params = len(trainable_params) + len(frozen_params)
        f.write(f"Total Parameter Blocks: {total_params}\n")
        f.write(f"Trainable Parameter Blocks: {len(trainable_params)}\n")
        f.write(f"Frozen Parameter Blocks: {len(frozen_params)}\n")

        f.write(f"Total params: {sum(p.numel() for p in model.parameters())}\n")
        f.write(f"Frozen params: {sum(p.numel() for p in model.parameters() if not p.requires_grad)}\n")
        f.write(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n")

        f.write("\n--- Stage Config ---\n")
        json.dump(stage_config, f, ensure_ascii=False)

def setup_model_for_stage(model, stage_config):
    """Freezes/unfreezes model parameters based on the stage configuration."""
    print(f"--- Configuring model for stage: {stage_config['name']} ---")
    
    # First, freeze everything to be safe
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze bridge layers (always trainable in this setup)
    for param in model.vision_proj.parameters():
        param.requires_grad = True
    for param in model.vision_layernorm.parameters():
        param.requires_grad = True
    
    print("Bridge layers (vision_proj, vision_layernorm) are TRAINABLE.")

    # Unfreeze ViT based on config
    if not stage_config.get('freeze_vit', True):
        for param in model.vit.parameters():
            param.requires_grad = True
        print("ViT backbone is TRAINABLE.")
    else:
        print("ViT backbone is FROZEN.")

    # Unfreeze mT5 based on config
    if not stage_config.get('freeze_mt5', True):
        for param in model.mt5.parameters():
            param.requires_grad = True
        print("Entire mT5 backbone is TRAINABLE.")
    else:
        # Handle partial unfreezing of mT5 decoder
        unfreeze_layers = stage_config.get('unfreeze_mt5_decoder_layers', 0)
        if unfreeze_layers > 0:
            print(f"Unfreezing the top {unfreeze_layers} layers of the mT5 DECODER.")
            if isinstance(model, (PaLI, PaLI_PhoBERT)):
                for layer in model.mt5.decoder.block[-unfreeze_layers:]:
                    for param in layer.parameters():
                        param.requires_grad = True
            elif isinstance(model, (PaLI_Classification, PaLI_Classification_PhoBERT)):
                for layer in model.mt5.transformer.decoder.block[-unfreeze_layers:]:
                    for param in layer.parameters():
                        param.requires_grad = True
            print("Rest of mT5 is FROZEN.")
        else:
            print("Entire mT5 backbone is FROZEN.")

    # Handle word embedding layer (mt5.shared), only for PhoBERT integration
    if stage_config.get('freeze_word_embeddings', True):
        if isinstance(model, (PaLI_Classification_PhoBERT)):
            for layer in model.mt5.transformer.shared.parameters():
                param.requires_grad = False

    # Always train the classification head
    if not stage_config.get('freeze_mt5_head', True):
        if isinstance(model, (PaLI_Classification, PaLI_Classification_PhoBERT)):
            for name, param in model.named_parameters():
                if "classification_head" in name:
                    param.requires_grad = True

def create_stage_optimizer(model, stage_config, args):
    """Creates an optimizer with differential learning rates for a specific stage."""
    print(f"--- Creating optimizer for stage: {stage_config['name']} ---")
    
    param_groups = []

    # Group 1: Bridge Layers
    bridge_params = [p for p in model.vision_proj.parameters() if p.requires_grad] + \
                    [p for p in model.vision_layernorm.parameters() if p.requires_grad]
    if bridge_params:
        base_lr = stage_config['lrs']['bridge']
        param_groups.append({'params': bridge_params, 'lr': base_lr, 'group_base_lr': base_lr})
        print(f"Bridge layers LR: {stage_config['lrs']['bridge']:.1e}")

    # Group 2: ViT Backbone
    vit_params = [p for p in model.vit.parameters() if p.requires_grad]
    if vit_params:
        base_lr = stage_config['lrs']['vit']
        param_groups.append({'params': vit_params, 'lr': base_lr, 'group_base_lr': base_lr})
        print(f"ViT backbone LR: {stage_config['lrs']['vit']:.1e}")

    # Group 3: mT5 Backbone
    mt5_params = []
    for name, param in model.mt5.named_parameters():
        if not "classification_head" in name and param.requires_grad:
            mt5_params.append(param)
    if mt5_params:
        base_lr = stage_config['lrs']['mt5']
        param_groups.append({'params': mt5_params, 'lr': base_lr, 'group_base_lr': base_lr})
        print(f"mT5 backbone LR: {stage_config['lrs']['mt5']:.1e}")

    # Group 4: mT5 classification head
    if isinstance(model, (PaLI_Classification, PaLI_Classification_PhoBERT)):
        mt5_head = [p for p in model.mt5.classification_head.parameters() if p.requires_grad]
        if mt5_head:
            base_lr = stage_config['lrs']['mt5_head']
            param_groups.append({'params': mt5_head, 'lr': base_lr, 'group_base_lr': base_lr})
            print(f"mT5 classification head LR: {stage_config['lrs']['mt5_head']:.1e}")
        
    return AdamW(param_groups, eps=args.opt_eps, betas=args.opt_betas, weight_decay=args.weight_decay)

def log_model_config(model, output_dir):
    outfile = os.path.join(output_dir, f"model_config.txt")
    with open(outfile, "w") as f:
        f.write(f"--- ViT Config ---")
        json.dump(model.vit.config.to_dict(), f,  indent = 4)
        
        f.write("\n\n--- mT5 Config ---\n\n")
        json.dump(model.mt5_config.to_dict(), f, indent = 4)

        f.write("\n\n--- Vision Projection Dropout ---\n\n")
        f.write(f"Vision Proj dropout rate: {model.vision_dropout.p}")
def train_stage(stage_config, global_epoch_start, model,
                data_loader_train, dataset_train, 
                data_loader_val, dataset_val,
                optimizer, device, task_handler, args,
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
            
            if isinstance(task_handler, PaLIClassificationHandler):
                train_stats = task_handler.train_batch(
                    model, args.batch_size, pixel_values, input_ids, attention_mask, labels
                )
            elif isinstance(task_handler, PaLIHandler):
                train_stats = task_handler.train_batch(
                    model, dataset_train.tokenizer, pixel_values, input_ids, attention_mask, labels, qid
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
                pali_save_model(args=args, epoch=epoch, model=model, optimizer=optimizer)
                
        if data_loader_val is not None:
            test_stats = pali_evaluate(
                data_loader_val, model, dataset_val.tokenizer, device, task_handler, batch_size=args.batch_size
            )
            print(f"Performance of the network on the {len(data_loader_val)} val batches: {test_stats['score']:.1f}%")

            if metric_trackers['max_accuracy'] < test_stats['score']:
                metric_trackers['max_accuracy'] = test_stats['score']
                if args.output_dir and args.save_ckpt:
                    print(f"*** New best score! Saving model for epoch {epoch}. ***")
                    pali_save_model(args=args, epoch=f"best-{epoch}", model=model, optimizer=optimizer)
            
            current_val_score = test_stats['score']
            current_val_loss = test_stats['loss']

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
                         **{f'val_{k}': v.item() if isinstance(v, torch.Tensor) else v for k, v in test_stats.items() if k != "prediction"},
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

        if metric_trackers['epochs_without_improvements'] >= args.patience and stage_index >= 2: # and average_score >= 85:
            print(f"No improvement in {args.patience} consecutive epochs. Early stopping stage.")
            return True # Signal to stop all training
    
    return False # Signal to continue to next stage

# <<< END HELPER FUNCTIONS FOR STAGED TRAINING >>>


def main(args, ds_init):

    if args.task_cache_path is None:
        args.task_cache_path = args.output_dir

    device = torch.device(args.device)

    phobert_tokenizer = None

    if args.phobert:
        phobert_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

    if args.pali_class == "pali_classification":
        if args.phobert:
            model = PaLI_Classification_PhoBERT(
                answer2label_path=args.answer2label,
                device=device
            ).to(device, non_blocking=True)
        else:
            model = PaLI_Classification(
                answer2label_path=args.answer2label,
                device=device
            ).to(device, non_blocking=True)
    else:
        if args.phobert:
            model = PaLI_PhoBERT(device=device).to(device)
        else:
            model = PaLI(device=device).to(device, non_blocking=True)

    dataset_train, data_loader_train, dataset_val, data_loader_val = create_pali_datasets(
        args,
        phobert_tokenizer=phobert_tokenizer
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('Number of params:', n_parameters)

    if args.pali_class == "pali_classification":
        task_handler = PaLIClassificationHandler()
    else:
        task_handler = PaLIHandler()

    log_model_config(model, args.output_dir)

    if args.eval:
        pali_auto_resume(args, model=model, optimizer=None, device=device) # Load best checkpoint for eval
        dataset_test, data_loader_test = create_pali_datasets(
            args,
            is_eval = True,
            phobert_tokenizer = phobert_tokenizer 
        )
        if isinstance(model, (PaLI_Classification, PaLI_Classification_PhoBERT)):
            result = pali_evaluate(
                data_loader_test,
                model,
                dataset_test.tokenizer,
                device, 
                task_handler,
                True,
                batch_size=args.batch_size)
        elif isinstance(model, (PaLI, PaLI_PhoBERT)):
            result = pali_evaluate(
                data_loader_test,
                model,
                dataset_test.tokenizer,
                device, task_handler,
                True,
                args.eval_num_beams
            )
        pali_dump_predictions(args, result["prediction"], "vivqa_pali_test")
        exit(0)
    
    start_time = time.time()
    
    # <<< START MODIFIED TRAINING LOGIC >>>

    if args.staged_training:
        print("--- Staged Training Enabled ---")
        
        STAGES = [
            {
                'name': 'Stage 1 Train Bridge',
                'epochs': 5,
                'freeze_vit': True,
                'freeze_mt5': True,
                'freeze_mt5_head': True,
                'freeze_word_embeddings': True,
                'unfreeze_mt5_decoder_layers': 0,
                'lrs': {'bridge': 1e-4, 'mt5': 0, 'vit': 0, 'mt5_head': 1e-4},
            },
            {
                'name': 'Stage 2 Adapt Decoder Head',
                'epochs': 5,
                'freeze_vit': True,
                'freeze_mt5': True,
                'freeze_mt5_head': False,
                'freeze_word_embeddings': True,
                'unfreeze_mt5_decoder_layers': 2, # Unfreeze top 2 layers of mT5 decoder
                'lrs': {'bridge': 5e-5, 'mt5': 2e-5, 'vit': 0, 'mt5_head': 1e-4}
            },
            {
                'name': 'Stage 3 Full Fine-Tuning',
                'epochs': 20,
                'freeze_vit': False,
                'freeze_mt5': False,
                'freeze_mt5_head': False,
                'freeze_word_embeddings': True,
                'unfreeze_mt5_decoder_layers': 0, # ignored as freeze_mt5 is False
                'lrs': {'bridge': 2e-5, 'mt5': 5e-6, 'vit': 5e-6, 'mt5_head': 2e-5}
            },
            {
                'name': 'Stage 4 Full Fine-Tuning',
                'epochs': 20,
                'freeze_vit': False,
                'freeze_mt5': False,
                'freeze_mt5_head': False,
                'freeze_word_embeddings': False,
                'unfreeze_mt5_decoder_layers': 0, # ignored as freeze_mt5 is False
                'lrs': {'bridge': 2e-5, 'mt5': 5e-6, 'vit': 5e-6, 'mt5_head': 2e-5}
            }
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
            model, old_optimizer, global_epoch = pali_auto_resume(
                args, model, old_optimizer, device
            )

            resume_stage_index = get_resume_stage_index(global_epoch, STAGES)
            print(f"Resuming from Stage {resume_stage_index+1}")

        for i, stage_config in enumerate(STAGES):
            if args.resume != '' and resume_stage_index > i:
                continue
                    
            stage_config['name'] = f"Stage_{i+1}_{stage_config['name']}" # Add number to name
            setup_model_for_stage(model, stage_config)
            
            log_model_architecture(model, args.output_dir, stage_config)
            
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
        
        skip_weight_decay_list = model.no_weight_decay()
        optimizer = create_optimizer(args, model, skip_list=skip_weight_decay_list)
        model, optimizer, args.start_epoch = pali_auto_resume(args, model=model, optimizer=optimizer, device=device)
        
        log_model_architecture(model, args.output_dir, "End_to_End") # Log architecture for standard training too

        # Encapsulate the original loop logic into a single stage config
        stage_config = {
            'name': 'End-to-End Training',
            'epochs': args.epochs - args.start_epoch,
            'lrs': {'bridge': args.lr, 'mt5': args.lr, 'vit': args.lr} # Use a single LR for scheduler
        }

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
            metric_trackers=metric_trackers
        )

    # <<< END MODIFIED TRAINING LOGIC >>>

    total_time = time.time() - start_time
    total_time_str = str(time.strftime('%H hours, %M minutes, %S seconds', time.gmtime(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == "__main__":
    opts, ds_init = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts, ds_init)