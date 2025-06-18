import os
import glob
from pathlib import Path

output = "/home/lenovo/exp1/output-dir"
output_dir = Path(output, "checkpoints")

checkpoint_path = None

# Maintain only 3 latest integer checkpoints
checkpoints = []

oldest_ckpt = 10000

for ckpt in glob.glob(os.path.join(output_dir, 'checkpoint-*')):
    try:
        epoch_str = ckpt.split('-')[-1].split('.')[0]
        if epoch_str.isdigit():
            epoch_int = int(epoch_str)
            checkpoints.append(ckpt)
            if epoch_int < oldest_ckpt:
                oldest_ckpt = epoch_int
                checkpoint_path = ckpt
    except Exception as e:
        print(f"Error parsing checkpoint {ckpt}: {e}")
        continue


# If more than 3 checkpoints, delete oldest ones
if len(checkpoints) > 3:
    print(f"🧹 Removing old checkpoint to free space: {checkpoint_path}")
    try:
        os.remove(checkpoint_path)
    except Exception as e:
        print(f"Failed to delete {checkpoint_path}: {e}")
