import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from sklearn.metrics import accuracy_score
import numpy as np
import nltk
import sys

nltk.download('wordnet') # run this once

with open("/home/lenovo/exp1/data/vivqa/vqa/test.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# print(data[0])

gt_dict = {item["id"]: item["answer"] for item in data["annotations"]}


with open("/home/lenovo/exp1/output-dir/submit_vivqa_test.json", "r", encoding="utf-8") as f:
    predictions = json.load(f)

# print(predictions[0])
pred_dict = {item["question_id"]: item["answer"] for item in predictions}

print(f"GT: {len(gt_dict)} items")
print(f"Pred: {len(pred_dict)} items")

assert len(gt_dict) == len(pred_dict)

# qid, pred = pred_dict.items().__iter__().__next__()
# gt = gt_dict.get(qid)
# print(f"GT: {gt}")
# print(f"Pred: {pred}")

# sys.exit(0)

smoothie = SmoothingFunction().method4

bleu_scores = []
meteor_scores = []
exact_match = []

for qid, pred in pred_dict.items():
    gt = gt_dict.get(qid)
    if gt:
        gt_tokens = gt.split()
        pred_tokens = pred.split()

        bleu = sentence_bleu([gt_tokens], pred_tokens, smoothing_function=smoothie)
        meteor = meteor_score([gt_tokens], pred_tokens)
        exact = int(gt.strip().lower() == pred.strip().lower())
        # save pairs that are not exact match as json
        if exact == 0:
            with open("wrong_pairs.json", "a") as f:
                json.dump({"qid": qid, "gt": gt, "pred": pred}, f)
                f.write("\n")


        bleu_scores.append(bleu)
        meteor_scores.append(meteor)
        exact_match.append(exact)

print(f"BLEU Score: {np.mean(bleu_scores):.4f}")
print(f"METEOR Score: {np.mean(meteor_scores):.4f}")
print(f"Exact Match Accuracy: {np.mean(exact_match):.4f}")
