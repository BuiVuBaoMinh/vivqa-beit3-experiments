import argparse
import time
import numpy as np
import os
import sys
import json
from pathlib import Path
import copy

from tqdm import tqdm

from torch.utils.data import DataLoader
import torch
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer, PhobertTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pali_dataset import ViVQAPaLIDataset, create_pali_datasets
from pali import PaLI
from pali_engine_for_finetuning import PaLIHandler, pali_evaluate
from pali_utils import pali_dump_predictions

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

    # For Phobert Vivqa experiments
    parser.add_argument('--phobert', action='store_true', default=False,
                        help="Specify whether replacing the phobert's tokenizer and embedding layers or not")
    parser.add_argument('--patience', type=int, default=5,
                        help='Number of training epochs without improvements.')

    # deepspeed parameters
    parser.add_argument('--enable_deepspeed', action='store_true', default=False)
    parser.add_argument('--initial_scale_power', type=int, default=16)
    parser.add_argument('--zero_stage', default=0, type=int,
                        help='ZeRO optimizer stage (default: 0)')

    known_args, _ = parser.parse_known_args()

    if known_args.enable_deepspeed:
        try:
            import deepspeed
            from deepspeed import DeepSpeedConfig
            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except:
            print("Please 'pip install deepspeed==0.4.0'")
            exit(0)
    else:
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

    print("Model = %s" % str(model))
    print('number of params:', n_parameters)

    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % args.batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(data_loader_train))

    # utils.auto_load_model(
    #     args=args, model=model, model_without_ddp=model,
    #     optimizer=optimizer)

    for name, param in model.named_parameters():
        if name.startswith("vit"):
            param.requires_grad = False
        if name.startswith("mt5"):
            if "encoder" in name:
                param.requires_grad = False

    # Check the frozen parameters
    with open(args.output_dir + "/Model_Architecture.txt", "w") as f:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                print(f"Frozen parameter block: {name}")
                f.write(f"Frozen parameter block: {name}\n")
            else:
                print(f"Trainable parameter block: {name}")
                f.write(f"Trainable parameter block: {name}\n")

    with open(args.output_dir + "/Paramaters.txt", "w") as f:
        print(f"Replaced beit3 text_embed w/ PhoBERT's. Checking trainable params.\n")
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

    if args.eval:
        dataset_test, data_loader_test = create_pali_datasets(
            args,
            is_eval = True,
            phobert_tokenizer = phobert_tokenizer 
        )
        result, _ = pali_evaluate(data_loader_test, model, dataset_test.tokenizer, device, task_handler)
        pali_dump_predictions(args, result, "vivqa_pali_test")
        exit(0)
    
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    max_accuracy, epochs_without_improvements = utils.get_max_accuracy_and_no_improvement_streak(args.output_dir)
    print(f"This section's initial max_accuracy: {max_accuracy}")
    print(f"This section's initial epochs_without_improvements: {epochs_without_improvements}")
    patience = args.patience
    for epoch in range(args.start_epoch, args.epochs):

        epoch_start_time = time.time()

        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        total_loss = 0.0
        num_batches = 0

        for step, batch in enumerate(tqdm(data_loader_train, desc=f"Epoch {epoch+1}", leave=False)):
            optimizer.zero_grad()

            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            qid = batch["qid"]

            train_stats = task_handler.train_batch(
                model, pixel_values, input_ids, attention_mask, labels, qid
            )
            
            loss = train_stats["loss"]

            loss.backward()
            optimizer.step()

            # Accumulate for epoch loss
            total_loss += loss.item()
            num_batches += 1

            # Print running average loss every 10 steps
            if (step + 1) % 10 == 0:
                running_avg_loss = total_loss / num_batches
                print(f"Epoch {epoch + 1} Step {step + 1}: loss: {running_avg_loss:.4f}, ")
        
        average_loss = total_loss / num_batches
        epoch_training_time = time.time() - epoch_start_time
        print(f"Epoch {epoch + 1} completed in {(epoch_training_time):.2f}s - Average Loss: {average_loss:.4f}")
        



        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)
                
        if data_loader_val is not None:
            test_stats, task_key = pali_evaluate(data_loader_val, model, device, task_handler)

            print(f"Performance of the network on the {len(data_loader_val.dataset)} val images: {test_stats[task_key]:.1f}%")
            if max_accuracy < test_stats[task_key]:
                max_accuracy = test_stats[task_key]
                epochs_without_improvements = 0 # Reset patience
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch="best")
            else:
                epochs_without_improvements += 1

            print(f'Max performance: {max_accuracy:.2f}%')
            
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

BATCH_SIZE = 1
LEARNING_RATE = 2e-15
EPOCHS = 50

train_dataset = ViVQAPaLIDataset(
    json_path="/home/lenovo/exp1/data/vivqa/vqa/train_en.json",
    image_dir="/home/lenovo/exp1/data/vivqa/images/train",
)

test_dataset = ViVQAPaLIDataset(
    json_path="/home/lenovo/exp1/data/vivqa/vqa/test_en.json",
    image_dir="/home/lenovo/exp1/data/vivqa/images/test",
)

train_data_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
)

test_data_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model =PaLI(device=device).to(device, non_blocking=True)

model.train()

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    print(f"training epoch {epoch}")
    for batch in train_data_loader:
        optimizer.zero_grad()

        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        output = model(pixel_values, input_ids, attention_mask, labels)
        loss = output.loss
        print(f"Loss: {loss}")

        loss.backward()
        optimizer.step()
        
