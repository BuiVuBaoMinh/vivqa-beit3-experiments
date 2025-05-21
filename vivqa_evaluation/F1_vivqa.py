import numpy as np
import json
from glossary import normalize_word

class F1:
    def compute_score(self, gts, res):
        """
        Main function to compute F1 score
        :param  gts (dict) : dictionary with key <id> and value <ground truth answer>
                res (dict)  : dictionary with key <question_id> and value <predicted answer>
        :return: f1 (float) : computed F1 score for the corpus
        """
        res = {key: value for key, value in res.items()}
        scores = []
        for key in res:
            print(f"Scoring key {key}.")
            r = res[key]
            scores_per_res = []

            gt = gts[key]
            # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
            if len(r) == 0 or len(gt) == 0:
                scores_per_res.append(int(r == gt))
                print("no-answer")
            else:
                common_tokens = set(r) & set(gt)
                print(f"r: {r}")
                print(f"gt: {gt}")
                print(f"common: {common_tokens}")
                # if there are no common tokens then f1 = 0
                if len(common_tokens) == 0:
                    scores_per_res.append(0)
                else:
                    prec = len(common_tokens) / len(r)
                    rec = len(common_tokens) / len(gt)

                    scores_per_res.append(2*(prec*rec)/(prec+rec))
        
            print("\n")

            scores_per_res = np.array(scores_per_res).mean()
            scores.append(scores_per_res)

        scores = np.array(scores)


        return scores.mean(), scores

    def __str__(self) -> str:
        return "F1"

def main():
    gt_file = "/root/projects/exp1/data/vivqa/vqa/test_en.json"
    res_file = "/root/projects/exp1/vivqa_scoring/en/submit_vivqa_test_en_sorted.json"

    with open(gt_file, "r", encoding="utf-8") as gtf:
        gt_data = json.load(gtf)

    with open(res_file, "r", encoding="utf-8") as rsf:
        res_data = json.load(rsf)

    # Convert to dicts with id as keys
    gts = {}
    for item in gt_data:
        gts[item["id"]] = [item["answer"].lower()]

    res = {}
    for item in res_data:
        res[item["question_id"]] = [item["answer"].lower()]

    scorer = F1()
    score, details = scorer.compute_score(gts, res)
    print(f"F1 Score (Exact Match Accuracy): {score:.4f}")

if __name__ == "__main__":
    main()