import json
import random

train_en_full = "/home/21khac.dd/bm/data/vivqa/vqa/train_en_full.json"
train_vi_full = "/home/21khac.dd/bm/data/vivqa/vqa/train_vi_full.json"

with open(train_en_full, "r", encoding="utf-8") as en_f:
    train_en_full_data = json.load(en_f)
    train_en_path = train_en_full.replace("train_en_full.json", "train_en.json")
    val_en_path = train_en_full.replace("train_en_full.json", "val_en.json")

    random.shuffle(train_en_full_data)
    val_en_data = train_en_full_data[:1000]
    train_en_data = train_en_full_data[1000:]

    with open(train_en_path, "w", encoding="utf-8") as fwtrain:
        json.dump(train_en_data, fwtrain, ensure_ascii=False, indent=4)

    with open(val_en_path, "w", encoding="utf-8") as fwval:
        json.dump(val_en_data, fwval, ensure_ascii=False, indent=4)

    with open(train_en_full, "r", encoding="utf-8") as f1:
        with open(train_en_path, "r", encoding="utf-8") as f2:
            with open(val_en_path, "r", encoding="utf-8") as f3:
                count_full = len(json.load(f1))
                count_train = len(json.load(f2))
                count_val = len(json.load(f3))

    print(f"Full train: {count_full}")
    print(f"Used train: {count_train}")
    print(f"Val: {count_val}")

with open(train_vi_full, "r", encoding="utf-8") as vi_f:
    train_vi_full_data = json.load(vi_f)
    train_vi_path = train_vi_full.replace("train_vi_full.json", "train_vi.json")
    val_vi_path = train_vi_full.replace("train_vi_full.json", "val_vi.json")

    random.shuffle(train_vi_full_data)
    val_vi_data = train_vi_full_data[:1000]
    train_vi_data = train_vi_full_data[1000:]

    with open(train_vi_path, "w", encoding="utf-8") as fwtrain:
        json.dump(train_vi_data, fwtrain, ensure_ascii=False, indent=4)

    with open(val_vi_path, "w", encoding="utf-8") as fwval:
        json.dump(val_vi_data, fwval, ensure_ascii=False, indent=4)

    with open(train_vi_full, "r", encoding="utf-8") as f1:
        with open(train_vi_path, "r", encoding="utf-8") as f2:
            with open(val_vi_path, "r", encoding="utf-8") as f3:
                count_full = len(json.load(f1))
                count_train = len(json.load(f2))
                count_val = len(json.load(f3))


    print(f"Full train: {count_full}")
    print(f"Used train: {count_train}")
    print(f"Val: {count_val}")

    