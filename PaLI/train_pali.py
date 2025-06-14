import argparse
import time
import numpy as np
import os
from pathlib import Path

from torch.utils.data import DataLoader
import torch
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer

from pali_dataset import ViVQAPaLIDataset
from pali import PaLI

import utils


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
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
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

    utils.init_distributed_mode(args)

    if ds_init is not None:
        utils.create_ds_config(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

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

    data_loader_train, data_loader_val = create_downstream_dataset(
        args,
        phobert_tokenizer=phobert_tokenizer
    )
    
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    max_accuracy, epochs_without_improvements = utils.get_max_accuracy_and_no_improvement_streak(args.output_dir)
    print(f"This section's initial max_accuracy: {max_accuracy}")
    print(f"This section's initial epochs_without_improvements: {epochs_without_improvements}")
    patience = args.patience
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
        
