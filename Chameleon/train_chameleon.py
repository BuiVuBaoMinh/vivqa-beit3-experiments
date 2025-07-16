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

from myChameleon import get_vivqa_chameleon, get_vivqa_chameleon_phobert, ChameleonForVQAClassification, PHOBERT_MODEL_ID
from chameleon_dataset import create_chameleon_datasets
from chameleon_engine_for_finetuning import ChameleonVQAHandler, chameleon_evaluate
import utils
from my_utils import my_dump_predictions, my_save_model, my_auto_resume

def get_args():
    parser = argparse.ArgumentParser('Chameleon fine-tuning and evaluation script for vivqa classification', add_help=False)

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

# --- Staged Training Helper Functions ---

def setup_model_for_stage(model: ChameleonForVQAClassification, stage_config):
    """Freezes/unfreezes model parameters based on the stage configuration."""
    print(f"--- Configuring model for stage: {stage_config['name']} ---")
    
    # Default to all frozen, then unfreeze based on config
    for param in model.parameters():
        param.requires_grad = False

    if not stage_config.get('freeze_vision_tower', True):
        print("Unfreezing: Vision Tower")
        for param in model.model.vision_tower.parameters():
            param.requires_grad = True

    if not stage_config.get('freeze_language_model', True):
        print("Unfreezing: Language Model (excluding embeddings)")
        for name, param in model.model.language_model.named_parameters():
            if 'embed_tokens' not in name:
                param.requires_grad = True

    if not stage_config.get('freeze_embed_tokens', True):
        print("Unfreezing: Language Model Embeddings")
        for param in model.model.language_model.embed_tokens.parameters():
            param.requires_grad = True

    if not stage_config.get('freeze_classifier', True):
        print("Unfreezing: Classifier Head")
        for param in model.classifier.parameters():
            param.requires_grad = True

def create_stage_optimizer(model: ChameleonForVQAClassification, stage_config, args):
    """Creates a staged optimizer with differential learning rates and weight decay."""
    print(f"--- Creating optimizer for stage: {stage_config['name']} ---")
    
    no_decay_list = model.no_weight_decay()
    optimizer_groups = []
    lrs = stage_config['lrs']
    
    # Define model components and their parameters
    components = {
        'vision_tower': (model.model.vision_tower.parameters(), lrs.get('vision_tower', 0)),
        'language_model': (
            (p for n, p in model.model.language_model.named_parameters() if 'embed_tokens' not in n),
            lrs.get('language_model', 0)
        ),
        'embed_tokens': (model.model.language_model.embed_tokens.parameters(), lrs.get('embed_tokens', 0)),
        'classifier': (model.classifier.parameters(), lrs.get('classifier', 0)),
    }

    for name, (params, lr) in components.items():
        if lr == 0: continue # Skip frozen components

        trainable_params = [p for p in params if p.requires_grad]
        if not trainable_params: continue

        decay_params = [p for p in trainable_params if not any(nd in n for nd in no_decay_list for n,p_ in model.named_parameters() if p_ is p)]
        no_decay_params = [p for p in trainable_params if any(nd in n for nd in no_decay_list for n,p_ in model.named_parameters() if p_ is p)]

        if decay_params:
            optimizer_groups.append({
                'params': decay_params, 'lr': lr, 'weight_decay': args.weight_decay, 'group_base_lr': lr
            })
        if no_decay_params:
            optimizer_groups.append({
                'params': no_decay_params, 'lr': lr, 'weight_decay': 0.0, 'group_base_lr': lr
            })
        print(f"  - Group '{name}' -> lr: {lr:.1e}, wd: {args.weight_decay} / 0.0")

    return bnb.optim.AdamW8bit(optimizer_groups, eps=args.opt_eps, betas=args.opt_betas)

# --- Main Training Logic ---

def train_one_epoch(model, data_loader, optimizer, device, epoch, handler, lr_scheduler, args):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)
    
    for i, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        if batch is None: continue

        if lr_scheduler is not None:
            step = i + epoch * len(data_loader)
            utils.adjust_learning_rate(optimizer, lr_scheduler, step)

        pixel_values = batch["pixel_values"].to(device, non_blocking=True, dtype=torch.bfloat16)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad()
        
        train_stats = handler.train_batch(
            model=model,
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        loss = train_stats["loss"]
        loss_value = loss.item()
        score_value = train_stats["score"].item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        loss.backward()
        if args.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        optimizer.step()

        torch.cuda.synchronize()
        
        metric_logger.update(loss=loss_value)
        metric_logger.update(score=score_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def main(args):
    device = torch.device(args.device)
    dtype = torch.bfloat16

    # --- Dataset and Model Setup ---
    phobert_tokenizer = PhobertTokenizer.from_pretrained(PHOBERT_MODEL_ID) if args.phobert else None
    
    if args.phobert:
        model = get_vivqa_chameleon_phobert(device=device, answer2label_path=args.answer2label).to(device, dtype=dtype)
    else:
        model = get_vivqa_chameleon(device=device, answer2label_path=args.answer2label).to(device, dtype=dtype)

    dataset_train, data_loader_train, dataset_val, data_loader_val = create_chameleon_datasets(
        args, phobert_tokenizer=phobert_tokenizer
    )

    handler = ChameleonVQAHandler()

    if args.eval:
        my_auto_resume(args, model=model, optimizer=None, device=device)
        dataset_test, data_loader_test = create_chameleon_datasets(args, is_eval=True, phobert_tokenizer=phobert_tokenizer)
        result = chameleon_evaluate(args, data_loader_test, model, device, handler, return_preds=True)
        my_dump_predictions(args, result["prediction"], "vivqa_chameleon_test")
        return

    # --- Staged Training ---
    if args.staged_training:
        print("--- Staged Training Enabled ---")
        STAGES = [
            {
                'name': 'Classifier_Only', 'epochs': 2,
                'freeze_vision_tower': True, 'freeze_language_model': True, 'freeze_embed_tokens': True, 'freeze_classifier': False,
                'lrs': {'classifier': 1e-4}
            },
            {
                'name': 'Full_Finetune', 'epochs': 8,
                'freeze_vision_tower': False, 'freeze_language_model': False, 'freeze_embed_tokens': False, 'freeze_classifier': False,
                'lrs': {'vision_tower': 1e-5, 'language_model': 1e-5, 'embed_tokens': 1e-6, 'classifier': 5e-5}
            }
        ]
        
        global_epoch = args.start_epoch
        max_accuracy = 0.0

        for stage_config in STAGES:
            setup_model_for_stage(model, stage_config)
            optimizer = create_stage_optimizer(model, stage_config, args)

            # Create LR scheduler for the stage
            num_training_steps_per_epoch = len(data_loader_train)
            lr_scheduler = utils.cosine_scheduler(
                base_value=max(stage_config['lrs'].values()),
                final_value=args.min_lr,
                epochs=stage_config['epochs'],
                niter_per_ep=num_training_steps_per_epoch,
                warmup_epochs=args.warmup_epochs
            )
            
            print(f"\n--- Starting Stage: {stage_config['name']} for {stage_config['epochs']} epochs ---")
            for epoch_in_stage in range(stage_config['epochs']):
                epoch = global_epoch + epoch_in_stage
                
                train_stats = train_one_epoch(model, data_loader_train, optimizer, device, epoch, handler, lr_scheduler, args)
                
                val_stats = chameleon_evaluate(args, data_loader_val, model, device, handler, return_preds=False)
                print(f"Validation Score: {val_stats['score']:.2f}%")

                if val_stats['score'] > max_accuracy:
                    max_accuracy = val_stats['score']
                    my_save_model(args, epoch=f"best-{epoch}", model=model, optimizer=optimizer)
                
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'val_{k}': v for k, v in val_stats.items()},
                             'epoch': epoch}
                
                if args.output_dir:
                    with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                        f.write(json.dumps(log_stats) + "\n")

            global_epoch += stage_config['epochs']
    else:
        # --- Standard End-to-End Training ---
        print("--- Standard End-to-End Training ---")
        # Your original training loop logic would go here
        # For simplicity, this is left as an exercise.
        # You would create one optimizer and one scheduler for all epochs.
        pass

if __name__ == "__main__":
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)
