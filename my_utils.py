import torch
import sys
import os
import json
from pathlib import Path
import re
import glob

from torchmetrics import Metric

class TrainingF1Score(Metric): # Same as F1 Score in vivqa-evaluation
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



def my_dump_predictions(args, result, file_suffix): # Reuse code from unilm beit3, changed name to recognize

    jsons = None
    jsons = result
    
    result_file = os.path.join(args.output_dir, f"submit_{file_suffix}.json")
    if jsons is not None:
        with open(result_file, "w") as fp:
            json.dump(jsons, fp, indent=4, ensure_ascii=False)
        print("Infer %d examples into %s" % (len(jsons), result_file))
    return result_file

def my_save_model(
    args, epoch, model, optimizer
):
    output_dir = Path(args.checkpoint_dir, "checkpoints")
    output_dir.mkdir(parents=True, exist_ok=True)
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

        # If already have 3 checkpoints, delete oldest one, adjust as needed
        if len(checkpoints) >= 1:
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


def my_auto_resume(args, model, optimizer=None, device='cuda'):

    checkpoint_path = None
    ckpt_dir = Path(args.checkpoint_dir, "checkpoints")

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
        checkpoint_path = os.path.join(ckpt_dir, best_checkpoint.name)


    # Case 2: Auto resume from latest epoch
    if hasattr(args, 'resume') and args.resume == 'latest':
        checkpoint_files = glob.glob(os.path.join(ckpt_dir, 'checkpoint-*'))
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
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'], strict=False)

        if optimizer is not None and 'optimizer' in checkpoint and not args.no_resume_optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])

        # Attempt to extract epoch from file name
        match = re.search(r'checkpoint-(?:best-)?(\d+)', checkpoint_path)
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