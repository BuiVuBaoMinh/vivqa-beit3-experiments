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
from pali import PaLI
from pali_engine_for_finetuning import PaLIHandler, pali_evaluate
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
    
    parser.add_argument('--lr_sched_type', type=str, default='linear', choices=['cos', 'linear'],
                        help='Learning rate scheduler type (default: linear)')

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--eval_batch_size', default=None, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=5, type=int)

     # Dataset parameters
    parser.add_argument('--data_path', default='/root/projects/exp1/data/vivqa', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint') # For PaLI, only accepts resume=="best"
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

    # Custom resume args
    parser.add_argument('--no_resume_optimizer', action="store_true", default=False,
                        help="This parameter prevents auto loading optimizer from checkpoint")
    

    known_args, _ = parser.parse_known_args()

    ds_init = None
    
    return parser.parse_args(), ds_init

def main(args, ds_init):

    if args.task_cache_path is None:
        args.task_cache_path = args.output_dir

    device = torch.device(args.device)

    phobert_model = None
    phobert_tokenizer = None
    if args.phobert:
        phobert_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2") # using PhoBERT tokenizer
        phobert_model = AutoModel.from_pretrained("vinai/phobert-base-v2") # Get word_embedding from phobert to replace text_embed

    dataset_train, data_loader_train, dataset_val, data_loader_val = create_pali_datasets(
        args,
        phobert_tokenizer=phobert_tokenizer
    )

    model =PaLI(device=device).to(device, non_blocking=True)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print("Model = %s" % str(model))
    print('number of params:', n_parameters)

    total_batch_size = args.batch_size * args.update_freq
    num_training_steps_per_epoch = len(data_loader_train.dataset) // total_batch_size

    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % args.batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(data_loader_train))
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)

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

    # for name, param in model.named_parameters():
    #     if name.startswith("vit"):
    #         param.requires_grad = False
    #     if name.startswith("mt5"):
    #         if "encoder" in name:
    #             param.requires_grad = False

    # Check the frozen parameters
    with open(args.output_dir + "/Model_Architecture.txt", "w") as f:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                # print(f"Frozen parameter block: {name}")
                f.write(f"Frozen parameter block: {name}\n")
            else:
                # print(f"Trainable parameter block: {name}")
                f.write(f"Trainable parameter block: {name}\n")

    with open(args.output_dir + "/Paramaters.txt", "w") as f:
        print(f"Checking trainable params.\n")
        total_params = sum(p.numel() for p in model.parameters())
        frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Total params: {total_params}\n")
        print(f"Frozen params: {frozen_params}\n")
        print(f"Trainable params: {n_parameters}\n")

        f.write(f"Total params: {total_params}\n")
        f.write(f"Frozen params: {frozen_params}\n")
        f.write(f"Trainable params: {n_parameters}\n")

    skip_weight_decay_list = model.no_weight_decay()

    optimizer = create_optimizer(
        args, model, skip_list=skip_weight_decay_list
    )

    task_handler = PaLIHandler()

    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
        sched_type=args.lr_sched_type,
    )

    model, optimizer, args.start_epoch = pali_auto_resume(
        args,
        model=model,
        optimizer=optimizer,
        device=device
    )

    if args.eval:
        dataset_test, data_loader_test = create_pali_datasets(
            args,
            is_eval = True,
            phobert_tokenizer = phobert_tokenizer 
        )
        result = pali_evaluate(data_loader_test, model, dataset_test.tokenizer, device, task_handler)
        pali_dump_predictions(args, result["prediction"], "vivqa_pali_test")
        exit(0)
    
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    max_accuracy, min_val_loss, epochs_without_improvements = utils.get_best_meters_and_no_improvement_streak(
        args.output_dir,
        early_stopping_metric=args.early_stopping
    )
    print(f"This section's initial max_accuracy: {max_accuracy}")
    print(f"This section's initial min_val_loss: {min_val_loss}")
    print(f"Using early stopping: {args.early_stopping} with patience = {args.patience}")
    print(f"This section's initial epochs_without_improvements: {epochs_without_improvements}")
    patience = args.patience

    max_accuracy_stop = max_accuracy # Used for early stopping logic
    for epoch in range(args.start_epoch, args.epochs):

        epoch_start_time = time.time()

        print(f"\nEpoch {epoch}/{args.epochs}")

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

            data_iter_step = epoch * len(data_loader_train) + step
            step_in_update  = data_iter_step // args.update_freq
            start_step = epoch * num_training_steps_per_epoch
            global_step = start_step + step_in_update

            if lr_schedule_values is not None and data_iter_step % args.update_freq == 0:
                for i, param_group in enumerate(optimizer.param_groups):
                    if lr_schedule_values is not None:
                        param_group["lr"] = lr_schedule_values[global_step] * param_group.get("lr_scale", 1.0)

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
            total_score += batch_score
            num_batches += 1

            for group in optimizer.param_groups:
                min_lr = min(min_lr, group["lr"])
                max_lr = max(max_lr, group["lr"])

            # Print running average loss every 10 steps
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
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                pali_save_model(args=args, epoch=epoch, model=model, optimizer=optimizer)
                
        if data_loader_val is not None:
            test_stats = pali_evaluate(data_loader_val, model, dataset_val.tokenizer, device, task_handler)

            print(f"Performance of the network on the {len(data_loader_val)} val batches: {test_stats['score']:.1f}%")

            if max_accuracy < test_stats['score']:
                max_accuracy = test_stats['score']
                if args.output_dir and args.save_ckpt:
                    pali_save_model(args=args, epoch=f"best-{epoch}", model=model, optimizer=optimizer)
                
            if args.early_stopping == "val_score":
                if max_accuracy_stop < test_stats['score']:
                    max_accuracy_stop = test_stats['score']
                    epochs_without_improvements = 0 # Reset patience
                else:
                    epochs_without_improvements += 1
            elif args.early_stopping == "val_loss":
                if min_val_loss > test_stats['loss']:
                    min_val_loss = test_stats['loss']
                    epochs_without_improvements = 0 # Reset patience
                else:
                    epochs_without_improvements += 1

            print(f'Max performance: {max_accuracy:.2f}%')
            print(f'Min loss: {min_val_loss}')
            
            log_stats = {**{f'train_{k}': v.item() for k, v in train_stats.items() if k != "prediction"},
                        'train_min_lr': min_lr,
                        'train_max_lr': max_lr,
                        **{f'val_{k}': v for k, v in test_stats.items() if k != "prediction"},
                        'epoch': epoch,
                        'n_parameters': n_parameters,
                        'time': epoch_training_time}
        else:
            log_stats = {**{f'train_{k}': v.item() for k, v in train_stats.items() if k != "prediction"},
                         **{f'test_{k}': v for k, v in test_stats.items() if k != "prediction"},
                         'train_min_lr': min_lr,
                         'train_max_lr': max_lr,
                         'epoch': epoch,
                         'n_parameters': n_parameters,
                         'time': epoch_training_time}

        if args.output_dir:
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        if epochs_without_improvements >= patience:
            print(f"No improvement in {patience} consecutive epochs. Early stopping.")
            break

if __name__ == "__main__":
    opts, ds_init = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts, ds_init)