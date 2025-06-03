import json
import argparse
from sklearn.metrics import precision_score, recall_score, f1_score

def segment_normalize(text):
    normalized_text = text.lower()
    normalized_text = normalized_text.replace("_", " ")
    return normalized_text

def get_args():
    parser = argparse.ArgumentParser('sklearn eval P, R, F1 for vivqa.', add_help=False)

    parser.add_argument('--gt_file', required=True, type=str,
                        help="Path to ground truth file (e.g test.json)")
    parser.add_argument('--res_file', required=True, type=str,
                        help="Path to result/prediction file (e.g submit_vivqa_test.json)")
    parser.add_argument('--answer2label', required=True, type=str,
                        help="Path of answer2label.txt.")
    parser.add_argument('--average', default='macro', type=str,
                        help="Specifies the macro param in sklearn eval functions.")
    
    return parser.parse_args()

def main(args):
    gt_file = args.gt_file
    res_file = args.res_file
    label_map_file = args.answer2label
    average = args.average

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

    # gts = {item["id"]: (item["answer"]) for item in gt_data}
    # res = {item["question_id"]: (item["answer"]) for item in res_data}

    y_true = []
    y_pred = []

    for qid in res:
        gt_answer = gts.get(qid, "")
        pred_answer = res[qid]

        gt_label = answer_to_label.get(gt_answer, -1)
        pred_label = answer_to_label.get(pred_answer, -1)

        # Skip examples with unknown answers
        if gt_label == -1: #or pred_label == -1:
            if (gt_label == -1):
                print(f"Unknown gt_label: {qid} - {gts[qid]}")
            # if (pred_label == -1):
            #     print(f"Unknown pred_label: {qid} - {res[qid]}")
            continue

        y_true.append(gt_label)
        y_pred.append(pred_label)

    # Compute F1 score
    print(f"len(gt_label): {len(gts)}")
    print(f"len(pred_label): {len(res)}")
    print(f"len(y_true): {len(y_true)}")
    print(f"len(y_pred): {len(y_pred)}")

    f1 = f1_score(y_true, y_pred, average=average)
    print(f"F1 Score ({average}): {f1:.4f}")

    # Compute Precision score
    p = precision_score(y_true, y_pred, average=average)
    print(f"Precision Score ({average}): {p:.4f}")

    # Compute Recall score
    r = recall_score(y_true, y_pred, average=average)
    print(f"Recall Score ({average}): {r:.4f}")


if __name__ == "__main__":
    opts = get_args()
    main(opts)