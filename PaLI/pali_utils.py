import torch
import sys
import os
import json
from pathlib import Path
import re

from torchmetrics import Metric

from utils import get_rank, get_world_size

class PaLIF1Score(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, predictions, references):
        """
        :param predictions: list of predicted strings
        :param references: list of ground truth strings
        """
        assert len(predictions) == len(references)

        for pred, ref in zip(predictions, references):
            pred_tokens = pred.strip().lower().split()
            ref_tokens = ref.strip().lower().split()

            # Edge case: one or both are empty
            if len(pred_tokens) == 0 or len(ref_tokens) == 0:
                f1 = 1.0 if pred_tokens == ref_tokens else 0.0
            else:
                common = set(pred_tokens) & set(ref_tokens)
                if len(common) == 0:
                    f1 = 0.0
                else:
                    prec = len(common) / len(pred_tokens)
                    rec = len(common) / len(ref_tokens)
                    f1 = 2 * (prec * rec) / (prec + rec)

            self.score += f1
            self.total += 1


    def compute(self):
        return self.score / self.total if self.total > 0 else torch.tensor(0.0)



def pali_dump_predictions(args, result, file_suffix):
    global_rank = get_rank()
    jsons = None
    if global_rank >= 0:
        output_file = os.path.join(args.task_cache_path, f"submit_{global_rank}_{file_suffix}.json")
        with open(output_file, "w") as fp:
            json.dump(result, fp, indent=2)
        torch.distributed.barrier()

        if global_rank == 0:
            world_size = get_world_size()
            jsons = []
            for i in range(world_size):
                each_file = os.path.join(args.task_cache_path, f"submit_{i}_{file_suffix}.json")
                with open(each_file, "r") as fp:
                    jsons += json.load(fp)
            
            new_jsons = []

            sys.exit(0)
            res_dict = dict()
            qid_key = "question_id"
            for item in jsons:
                if item[qid_key] in res_dict:
                    continue
                new_jsons.append(item)
                res_dict[item[qid_key]] = item
            jsons = new_jsons

        torch.distributed.barrier()
        os.remove(output_file)
    else:
        jsons = result
    
    result_file = os.path.join(args.output_dir, f"submit_{file_suffix}.json")
    if jsons is not None:
        with open(result_file, "w") as fp:
            json.dump(jsons, fp, indent=2, ensure_ascii=False)
        print("Infer %d examples into %s" % (len(jsons), result_file))
    return result_file

def pali_save_model(
    args, epoch, model, optimizer
):
    """
    Save the PaLI model and optionally the optimizer and training state.
    Due to lack of disk space, we:
    - Keeps only 3 latest normal checkpoints by deleting the one with lowest epoch number.
    - Keeps only 1 best checkpoint at a time, deleting the older best.
    Max number of checkpoints-{int} and checkpoint-best-{int} can be modified in this function.

    Args:
        model (PaLI): PaLI model.
        optimizer (torch.optim.Optimizer): Optimizer, for resume in future.
        epoch (int): Current training epoch.
    """
    output_dir = Path(args.output_dir, "checkpoints")
    filename = "checkpoint-%s" % epoch
    full_path = output_dir / filename

    # Logic for deleting old "best" checkpoint
    if isinstance(epoch, str) and "best" in epoch:
        # Delete previous best-* checkpoint
        best_checkpoints = list(output_dir.glob("checkpoint-best-*"))
        for best_ckpt in best_checkpoints:
            print(f"🧹 Removing old best checkpoint: {best_ckpt}")
            best_ckpt.unlink()
            
    else:
        # Maintain only 3 latest integer checkpoints
        oldest_ckpt= 10000 # Assume never train up to 10k epochs.
        oldest_ckpt_path = None
        checkpoints = []
        import glob
        for ckpt in glob.glob(os.path.join(output_dir, 'checkpoint-*')):
            if "best" in ckpt:
                continue
            try:
                epoch_str = ckpt.split('-')[-1].split('.')[0]
                if epoch_str.isdigit():
                    epoch_int = int(epoch_str)
                    checkpoints.append(ckpt)
                    if epoch_int < oldest_ckpt:
                        oldest_ckpt = epoch_int
                        oldest_ckpt_path = ckpt
            except Exception as e:
                print(f"Error parsing checkpoint {ckpt}: {e}")
                continue

        # If already have 3 checkpoints, delete oldest one
        if len(checkpoints) >= 3:
            print(f"🧹 Removing old checkpoint to free space: {oldest_ckpt_path}")
            try:
                os.remove(oldest_ckpt_path)
            except Exception as e:
                print(f"Failed to delete {oldest_ckpt_path}: {e}")

    checkpoint = {
        'model': model.state_dict()
    }

    if optimizer is not None:
        checkpoint['optimizer'] = optimizer.state_dict()
    if epoch is not None:
        checkpoint['epoch'] = epoch

    torch.save(checkpoint, full_path)
    print(f"✅ Model saved to {full_path}")


def pali_auto_resume(args, model, optimizer=None, device='cuda'):
    """
    Automatically resume the latest checkpoint if available and args.auto_resume is True.

    Args:
        args: .
        model (nn.Module): The PaLI model instance.
        optimizer (torch.optim.Optimizer): The optimizer instance.
        device (str): The device to map the checkpoint to.

    Returns:
        model: Model with loaded weights.
        optimizer: Optimizer with loaded state if provided.
        epoch (int): The epoch to resume from.

    Sample usage (in case I forget):
    ```
    model = PaLI(...)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model, optimizer, start_epoch = pali_auto_resume(args, model, optimizer, device=args.device)
    ```
    """
    checkpoint_path = None
    ckpt_dir = Path(args.output_dir, "checkpoints")

    # Case 1: Resume from 'best'
    if hasattr(args, 'resume') and args.resume == 'best':
        best_checkpoints = sorted(ckpt_dir.glob('checkpoint-best-*'))

        if not best_checkpoints:
            raise FileNotFoundError(f"No best checkpoint found in {ckpt_dir}")

        # Optionally, choose the one with highest epoch
        def extract_epoch(ckpt):
            match = re.search(r'checkpoint-best-(\d+)', ckpt.name)
            return int(match.group(1)) if match else -1

        best_checkpoint = max(best_checkpoints, key=extract_epoch)
        checkpoint_path = best_checkpoint

    # Case 2: Resume from specific epoch
    elif hasattr(args, 'resume_epoch') and args.resume_epoch is not None:
        checkpoint_path = os.path.join(args.output_dir, f'checkpoint-{args.resume_epoch}')

    # Case 3: Auto resume from latest epoch
    elif getattr(args, 'auto_resume', False):
        import glob
        checkpoint_files = glob.glob(os.path.join(args.output_dir, 'checkpoint-*'))
        latest_epoch = -1
        for ckpt in checkpoint_files:
            ckpt_name = os.path.basename(ckpt)
            if 'checkpoint-best' in ckpt_name:
                continue
            try:
                epoch_str = ckpt.split('-')[-1].split('.')[0]
                if epoch_str.isdigit():
                    epoch = int(epoch_str)
                    if epoch > latest_epoch:
                        latest_epoch = epoch
                        checkpoint_path = ckpt
            except Exception:
                continue

    if checkpoint_path is not None and os.path.isfile(checkpoint_path):
        print(f"✅ Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])

        if optimizer is not None and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])

        # Attempt to extract epoch from file name
        match = re.search(r'checkpoint-(?:best-)?(\d+)', checkpoint_path.name)
        if match:
            start_epoch = int(match.group(1)) + 1  # resume from next epoch
        else:
            # Fallback: try to get it from checkpoint itself
            start_epoch = checkpoint.get('epoch', 0) + 1

        args.resume = checkpoint_path
        args.start_epoch = start_epoch if isinstance(start_epoch, int) else 0
        return model, optimizer, args.start_epoch

    print("⚠️ No valid checkpoint found to resume.")
    args.start_epoch = 0
    return model, optimizer, 0