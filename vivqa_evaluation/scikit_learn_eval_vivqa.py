import numpy as np
import json
from sklearn.metrics import f1_score

def main():
    gt_file = "/root/projects/exp1/data/vivqa/vqa/test_en.json"
    res_file = "/root/projects/exp1/vivqa_scoring/en/submit_vivqa_test_en_sorted.json"

    with open(gt_file, "r", encoding="utf-8") as gtf:
        gt_data = json.load(gtf)

    with open(res_file, "r", encoding="utf-8") as rsf:
        res_data = json.load(rsf)

    # Build dictionaries for fast access
    gts = {item["id"]: item["answer"].lower() for item in gt_data}
    res = {item["question_id"]: item["answer"].lower() for item in res_data}

    # Prepare y_true and y_pred as label arrays
    y_true = []
    y_pred = []

    for qid in res:
        pred = res[qid]
        gt = gts.get(qid, "")  # fallback to empty string if missing

        y_true.append(gt)
        y_pred.append(pred)

    # Convert to binary labels: 1 if match, 0 if not
    y_true_bin = [1] * len(y_true)
    y_pred_bin = [int(p == t) for p, t in zip(y_pred, y_true)]

    # Compute F1 score
    f1 = f1_score(y_true_bin, y_pred_bin)
    print(f"Exact Match F1 Score (as classification): {f1:.4f}")


if __name__ == "__main__":
    main()