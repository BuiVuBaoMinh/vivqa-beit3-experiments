import os
import glob
import random

data_path = "data/vivqa"

for split in ["train", "val", "test"]:
    split_name = {
        "train": "train",
        "val": "train", # same folder as train images
        "test": "test",
    }[split]
    paths = list(glob.glob(f"{data_path}/images/{split_name}/*.jpg"))
    random.shuffle(paths)
    
    # annot_paths = [path for path in paths \
    #     if int(path.split("/")[-1].split("_")[-1][:-4]) in annot]


test_path = paths[0]
print(test_path)
print(test_path.split("/")[-1][:-4])
