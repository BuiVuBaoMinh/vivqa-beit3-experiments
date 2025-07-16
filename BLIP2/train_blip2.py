import argparse
import time
import os
import sys
import json
from pathlib import Path
import math

from tqdm import tqdm
import torch
from transformers import PhobertTokenizer
import bitsandbytes as bnb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from my_blip2 import (
    get_vivqa_blip2, 
    get_vivqa_blip2_phobert, 
    get_vivqa_blip2_phobert_with_adapter, 
    Blip2ForVQAClassification, 
    PHOBERT_MODEL_ID
)
from blip2_dataset import create_blip2_datasets
from blip2_engine_for_finetuning import Blip2VQAHandler, blip2_evaluate
import utils
from optim_factory import LayerDecayValueAssigner
from my_utils import my_dump_predictions, my_save_model, my_auto_resume

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
    parser.add_argument('--staged_training', action='store_true', default=False,
                        help="Enable 3-stage gradual unfreezing training.")
    parser.add_argument('--freeze_embed_tokens', action='store_true', default=False,
                        help="Freeze SmolVLM's text_model(LLama)'s embed_tokens.")
    
    # Custom resume args
    parser.add_argument('--no_resume_optimizer', action="store_true", default=False,
                        help="This parameter prevents auto loading optimizer from checkpoint")
    

    known_args, _ = parser.parse_known_args()
    
    return parser.parse_args()

# --- Staged Training Helpers for BLIP-2 ---

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
        json.dump(stage_config, f, ensure_ascii=False, indent=4)

def get_resume_stage_index(global_epoch, stages):
    print(f"{global_epoch}")
    cnt = 0
    for i, stage_config in enumerate(stages):
        cnt += stage_config["epochs"]
        if cnt > global_epoch:
            return i 
        
def setup_model_for_stage(model: Blip2ForVQAClassification, stage_config):
    """Freezes/unfreezes BLIP-2 model parameters based on the stage configuration."""
    print(f"--- Configuring model for stage: {stage_config['name']} ---")
    
    # Freeze everything by default
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze based on config
    if not stage_config.get('freeze_vision_model', True):
        print("Unfreezing: Vision Model")
        for param in model.blip2.vision_model.parameters():
            param.requires_grad = True

    if not stage_config.get('freeze_qformer', True):
        print("Unfreezing: Q-Former")
        for param in model.blip2.qformer.parameters():
            param.requires_grad = True

         # --- ADDED: Unfreeze related components ---
        print("Unfreezing: Query Tokens")
        model.blip2.query_tokens.requires_grad = True

        print("Unfreezing: Language Projection Layer")
        for param in model.blip2.language_projection.parameters():
            param.requires_grad = True

    if not stage_config.get('freeze_language_model', True):
        print("Unfreezing: Language Model (excluding embeddings)")
        for name, param in model.blip2.language_model.named_parameters():
            if 'embed_tokens' not in name:
                param.requires_grad = True
    
    if not stage_config.get('freeze_embed_tokens', True):
        print("Unfreezing: Embed Tokens")
        for name, param in model.blip2.language_model.get_input_embeddings().named_parameters():
            print(name)
            param.requires_grad = True

    if not stage_config.get('freeze_classifier', True):
        print("Unfreezing: Classifier Head")
        for param in model.classifier.parameters():
            param.requires_grad = True

    if hasattr(model, 'phobert_embedding_adapter') and not stage_config.get('freeze_phobert_adapter', True):
        print("Unfreezing: PhoBERT Embedding Adapter")
        for param in model.phobert_embedding_adapter.parameters():
            param.requires_grad = True

def create_stage_optimizer(model: Blip2ForVQAClassification, stage_config, args):
    """Creates a staged optimizer for BLIP-2."""
    print(f"--- Creating optimizer for stage: {stage_config['name']} ---")
    
    no_decay_list = model.no_weight_decay()
    optimizer_groups = []
    lrs = stage_config['lrs']
    
    components = {
        'vision_model': (model.blip2.vision_model.parameters(), lrs.get('vision_model', 0)),
        'qformer': (model.blip2.qformer.parameters(), lrs.get('qformer', 0)),
        'language_model': ((p for n, p in model.blip2.language_model.named_parameters() if 'embed' not in n), lrs.get('language_model', 0)),
        'embed_tokens': (model.blip2.language_model.get_input_embeddings().parameters(), lrs.get('embed_tokens', 0)),
        'classifier': (model.classifier.parameters(), lrs.get('classifier', 0)),
    }

    # --- Assign LRs to the bridge components ---
    # We will give them the same LR as the Q-Former since they are functionally related.
    qformer_lr = lrs.get('qformer', 0)
    components['query_tokens'] = ([model.blip2.query_tokens], qformer_lr)
    components['language_projection'] = (model.blip2.language_projection.parameters(), qformer_lr)

     # Add the adapter to the components dictionary if it exists
    if hasattr(model, 'phobert_embedding_adapter'):
        components['phobert_adapter'] = (model.phobert_embedding_adapter.parameters(), lrs.get('phobert_adapter', 0))

    for name, (params, lr) in components.items():
        if lr == 0: continue
        
        trainable_params = [p for p in params if p.requires_grad]
        if not trainable_params: continue

        # Simple split for decay/no_decay
        decay_params = [p for p in trainable_params if p.dim() >= 2]
        no_decay_params = [p for p in trainable_params if p.dim() < 2]

        if decay_params:
            optimizer_groups.append({'params': decay_params, 'lr': lr, 'weight_decay': args.weight_decay, 'group_base_lr': lr})
        if no_decay_params:
            optimizer_groups.append({'params': no_decay_params, 'lr': lr, 'weight_decay': 0.0, 'group_base_lr': lr})
        print(f"  - Group '{name}' -> lr: {lr:.1e}")

    return bnb.optim.AdamW8bit(optimizer_groups, eps=args.opt_eps, betas=args.opt_betas)

def log_blip2_model_config(model: Blip2ForVQAClassification, output_dir):
    outfile = os.path.join(output_dir, f"model_config.txt")
    with open(outfile, "w") as f:
        f.write("\n\n--- Blip2 Model Config ---")
        json.dump(model.blip2.config.to_dict(), f, indent = 4)

        f.write("\n\n--- Classifier head Dropout ---\n\n")
        f.write(f"nn.DropOut: {model.dropout.p}")

def train_stage(stage_config, global_epoch_start, model,
                data_loader_train, dataset_train, 
                data_loader_val, dataset_val,
                optimizer, device, task_handler: Blip2ForVQAClassification, args,
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

            pixel_values = batch["pixel_values"].to(device=device, dtype=torch.bfloat16)
            input_ids = batch["input_ids"].to(device=device)
            attention_mask = batch["attention_mask"].to(device=device)
            labels = batch["labels"].to(device=device)
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
                labels = labels, # Use for our classification
                qid = None
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
            val_stats = blip2_evaluate(
                args, data_loader_val, model, device, task_handler, return_preds=False
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
            
            del val_stats
            torch.cuda.empty_cache()
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

def main(args):

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    if args.task_cache_path is None:
        args.task_cache_path = args.output_dir

    device = torch.device(args.device)

    dtype = torch.bfloat16

    phobert_tokenizer = None

    if args.phobert:
        phobert_tokenizer = PhobertTokenizer.from_pretrained("vinai/phobert-base-v2", use_fast=True)

    if args.phobert:
        # model = get_vivqa_blip2_phobert(device=device, answer2label_path=args.answer2label).to(device, non_blocking=True, dtype=dtype)
        model = get_vivqa_blip2_phobert_with_adapter(
            device=device, answer2label_path=args.answer2label
        ).to(device, non_blocking=True, dtype=dtype)
    else:
        model = get_vivqa_blip2(
            device = device, answer2label_path=args.answer2label
        ).to(device, non_blocking=True, dtype=dtype)

    model.gradient_checkpointing_enable()

    dataset_train, data_loader_train, dataset_val, data_loader_val = create_blip2_datasets(
        args,
        phobert_tokenizer=phobert_tokenizer,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('Number of params:', n_parameters)

    task_handler = Blip2VQAHandler()

    if args.eval:
        my_auto_resume(args, model=model, optimizer=None, device=device) # Load best checkpoint for eval
        dataset_test, data_loader_test = create_blip2_datasets(
            args,
            is_eval = True,
            phobert_tokenizer = phobert_tokenizer 
        )
        result = blip2_evaluate(
            args,
            data_loader=data_loader_test,
            model=model,
            device=device,
            handler=task_handler,
            return_preds=True
        )
        my_dump_predictions(args, result["prediction"], "vivqa_blip2_test")
        exit(0)

    start_time = time.time()

    log_blip2_model_config(model, args.output_dir)

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
        else:
            assigner = None
        
        if assigner is not None:
            print("Assigned values = %s" % str(assigner.values))

        # Encapsulate the original loop logic into a single stage config
        stage_config = {
            'name': 'Full_Finetune_End_to_End',
            'epochs': args.epochs, 
            'freeze_vision_model': True,
            'freeze_language_model': False, 
            'freeze_embed_tokens': args.freeze_embed_tokens, 
            'freeze_qformer': False, 
            'freeze_classifier': False, 
            'freeze_phobert_adapter': False, # Ensure the adapter is trainable
            'lrs': {
                'vision_model': 2e-5, 
                'qformer': 2e-5, 
                'language_model': 2e-5, 
                'embed_tokens': 5e-6, 
                'classifier': 5e-5,
                'phobert_adapter': 5e-5
            }
        }

        optimizer = create_stage_optimizer(model, stage_config, args)
        setup_model_for_stage(model, stage_config)
        if args.resume != '':
            model, optimizer, args.start_epoch = my_auto_resume(args, model=model, optimizer=optimizer, device=device)

        log_model_architecture(model, args.output_dir, stage_config) # Log architecture for standard training too

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
    Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)
