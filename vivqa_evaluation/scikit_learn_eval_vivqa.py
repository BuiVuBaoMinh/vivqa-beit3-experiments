import json
from sklearn.metrics import precision_score, recall_score, f1_score

def segment_normalize(text):
    normalized_text = text.lower()
    normalized_text = normalized_text.replace("_", " ")
    return normalized_text

def main():
    gt_file = "/root/projects/exp1/data/vivqa/vqa/test_vi.json"
    res_file = "/root/projects/exp1/vivqa_scoring/vi/submit_vivqa_test_vi_sorted.json"
    label_map_file = "/root/projects/exp1/data/vivqa/dicts/answer2label.txt"

    # Load label map
    answer_to_label = {}
    with open(label_map_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            answer_to_label[item["answer"].lower().strip()] = item["label"]

    # Load ground truth and results
    with open(gt_file, "r", encoding="utf-8") as gtf:
        gt_data = json.load(gtf)

    with open(res_file, "r", encoding="utf-8") as rsf:
        res_data = json.load(rsf)

    # Build dictionaries for fast access
    gts = {item["id"]: segment_normalize(item["answer"]) for item in gt_data}
    res = {item["question_id"]: segment_normalize(item["answer"]) for item in res_data}

    y_true = []
    y_pred = []

    for qid in res:
        gt_answer = gts.get(qid, "")
        pred_answer = res[qid]

        gt_label = answer_to_label.get(gt_answer, -1)
        pred_label = answer_to_label.get(pred_answer, -1)

        # Skip examples with unknown answers
        if gt_label == -1 or pred_label == -1:
            if (gt_label == -1):
                print(f"Unknown gt_label: {qid} - {gts[qid]}")
            if (pred_label == -1):
                print(f"Unknown pred_label: {qid} - {res[qid]}")
            continue

        y_true.append(gt_label)
        y_pred.append(pred_label)

    # Compute F1 score
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"F1 Score (macro): {f1:.4f}")

    # Compute Precision score
    p = precision_score(y_true, y_pred, average="macro")
    print(f"Precision Score (macro): {p:.4f}")

    # Compute Recall score
    r = recall_score(y_true, y_pred, average="macro")
    print(f"Recall Score (macro): {r:.4f}")


if __name__ == "__main__":
    main()