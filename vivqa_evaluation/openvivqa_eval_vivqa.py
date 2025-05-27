import numpy as np
import json
import argparse
import py_vncorenlp
from transformers import XLMRobertaTokenizer, AutoModel, AutoTokenizer

class F1:
    def compute_score(self, gts, res):
        """
        Main function to compute F1 score
        :param  gts (dict) : dictionary with key <image> and value <tokenized hypothesis / candidate sentence>
                res (dict)  : dictionary with key <image> and value <tokenized reference sentence>
        :return: f1 (float) : computed F1 score for the corpus
        """
        res = {key: value[0].split() for key, value in res.items()}
        scores = []
        for key in res:
            r = res[key]
            scores_per_res = []
            for gt in gts[key]:
                gt = gt.split()
                # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
                if len(r) == 0 or len(gt) == 0:
                    scores_per_res.append(int(r == gt))
                    print("F1 no answer")
                else:
                    common_tokens = set(r) & set(gt)
                    # if there are no common tokens then f1 = 0
                    if len(common_tokens) == 0:
                        scores_per_res.append(0)
                    else:
                        prec = len(common_tokens) / len(r)
                        rec = len(common_tokens) / len(gt)

                        scores_per_res.append(2*(prec*rec)/(prec+rec))

            scores_per_res = np.array(scores_per_res).mean()
            scores.append(scores_per_res)

        scores = np.array(scores)

        return scores.mean(), scores

    def __str__(self) -> str:
        return "F1"

class Precision:
    def compute_score(self, gts, res):
        """
        Main function to compute precision score
        :param  gts (dict) : dictionary with key <image> and value <tokenized hypothesis / candidate sentence>
                res (dict)  : dictionary with key <image> and value <tokenized reference sentence>
        :return: accuracy (float) : computed Accuracy score for the corpus
        """
        res = {key: value[0].split() for key, value in res.items()}
        scores = []
        for key in res:
            r = res[key]
            scores_per_res = []
            for gt in gts[key]:
                gt = gt.split()
                # if either the prediction or the truth is no-answer then precision = 1 if they agree, 0 otherwise
                if len(r) == 0 or len(gt) == 0:
                    scores_per_res.append(int(r == gt))
                else:
                    common_tokens = set(r) & set(gt)
                    # if there are no common tokens then precision = 0
                    if len(common_tokens) == 0:
                        scores_per_res.append(0)
                    else:
                        scores_per_res.append(len(common_tokens)/len(r))

            scores_per_res = np.array(scores_per_res).mean()
            scores.append(scores_per_res)

        scores = np.array(scores)

        return scores.mean(), scores

    def __str__(self) -> str:
        return "Precision"

class Recall:
    def compute_score(self, gts, res):
        """
        Main function to compute recall score
        :param  gts (dict) : dictionary with key <image> and value <tokenized hypothesis / candidate sentence>
                res (dict)  : dictionary with key <image> and value <tokenized reference sentence>
        :return: accuracy (float) : computed Accuracy score for the corpus
        """
        res = {key: value[0].split() for key, value in res.items()}
        scores = []
        for key in res:
            r = res[key]
            scores_per_res = []
            for gt in gts[key]:
                gt = gt.split()
                # if either the prediction or the truth is no-answer then recall = 1 if they agree, 0 otherwise
                if len(r) == 0 or len(gt) == 0:
                    scores_per_res.append(int(r == gt))
                else:
                    common_tokens = set(r) & set(gt)
                    # if there are no common tokens then recall = 0
                    if len(common_tokens) == 0:
                        scores_per_res.append(0)
                    else:
                        scores_per_res.append(len(common_tokens)/len(gt))

            scores_per_res = np.array(scores_per_res).mean()
            scores.append(scores_per_res)

        scores = np.array(scores)

        return scores.mean(), scores

    def __str__(self) -> str:
        return "Recall"

def simple_tokenize(text):
    """
    Convert text to a list of tokens by removing white spaces, and lowercasing.
    :param  text(str): The text to be tokenized
    """
    return text.lower().strip().split()

class ExactMatch:
    def compute_score(self, gts, res):
        """
        Compute the Exact Match (EM) score between raw string answers.
        :param  gts (dict) : dictionary with key <image> and value <list of raw string reference answers>
                res (dict) : dictionary with key <image> and value <list containing raw string predicted answer>
        :return: em (float) : computed Exact Match score for the corpus
        """
        scores = []
        for key in res:
            pred = res[key][0]
            match_scores = [int(pred == gt) for gt in gts[key]]
            # if pred!=gts[key][0]:
            #     print(f"Wrong pairs id: {key}")
            #     print(f"pred: {pred}")
            #     print(f"gt: {gts[key][0]}\n")
            scores.append(np.mean(match_scores))  # average across references

        scores = np.array(scores)
        return scores.mean(), scores

    def __str__(self) -> str:
        return "ExactMatch"

def segment_normalize(text):
    normalized_text = text.lower()
    normalized_text = normalized_text.replace("_", " ")
    return normalized_text

def get_args():
    parser = argparse.ArgumentParser('OpenViVQA eval P, R, F1 for vivqa.', add_help=False)

    parser.add_argument('--gt_file', required=True, type=str,
                        help="Path to ground truth file (e.g test.json)")
    parser.add_argument('--res_file', required=True, type=str,
                        help="Path to result/prediction file (e.g submit_vivqa_test.json)")
    parser.add_argument('--phobert', action='store_true', default=False,
                        help="Use PhoBERT's tokenizer.")
    parser.add_argument('--simple_tokenize', action='store_true', default=False,
                        help='Use simple tokenizing method, only lowercase and whitespace split.')
    
    return parser.parse_args()

def main(args):

    if not args.simple_tokenize:
        if args.phobert:
            tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2") # using PhoBERT tokenizer
        else:
            tokenizer = XLMRobertaTokenizer("/root/projects/exp1/text-tokenizers/beit3.spm")
            # tokenizer = XLMRobertaTokenizer("/home/lenovo/exp1/beit3-model-n-ckpts/beit3-model/beit3.spm") # for gcp vm

    gt_file = args.gt_file
    res_file = args.res_file

    with open(gt_file, "r", encoding="utf-8") as gtf:
        gt_data = json.load(gtf)

    with open(res_file, "r", encoding="utf-8") as rsf:
        res_data = json.load(rsf)


    gts = {}
    res = {}
    if not args.simple_tokenize:
        if args.phobert:
            for item in gt_data:
                gts[item["id"]] = tokenizer.tokenize(item["answer"].lower())

            rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir='/root/projects/exp1/vncorenlp')

            # for item in gt_data:
            #     segmented_ans = rdrsegmenter.word_segment(item["answer"])
            #     gts[item["id"]] = tokenizer.tokenize(segmented_ans[0])
            
            for item in res_data:
                segmented_ans = rdrsegmenter.word_segment(item["answer"])
                res[item["question_id"]] = tokenizer.tokenize(segmented_ans[0])
        else:
            for item in gt_data:
                gts[item["id"]] = tokenizer.tokenize(item["answer"].lower())

            for item in res_data:
                res[item["question_id"]] = tokenizer.tokenize(item["answer"].lower())
    else:
        for item in gt_data:
            gts[item["id"]] = simple_tokenize(item["answer"])

        for item in res_data:
            res[item["question_id"]] = simple_tokenize(item["answer"])

    scorer_p = Precision()
    score_p, details_p = scorer_p.compute_score(gts, res)
    print(f"Token-based Precision Score (P): {score_p:.4f}")

    scorer_r = Recall()
    score_r, details_r = scorer_r.compute_score(gts, res)
    print(f"Token-based Recall Score (R): {score_r:.4f}")

    scorer_f1 = F1()
    score_f1, details_f1 = scorer_f1.compute_score(gts, res)
    print(f"Token-based F1 Score (F1): {score_f1:.4f}")

    # Raw string ground truth and result (for Exact Match)
    gts_raw = {}
    for item in gt_data:
        gts_raw[item["id"]] = [segment_normalize(item["answer"])]

    res_raw = {}
    for item in res_data:
        res_raw[item["question_id"]] = [segment_normalize(item["answer"])]

    # Compute exact match
    scorer_em = ExactMatch()
    score_em, details_em = scorer_em.compute_score(gts_raw, res_raw)
    print(f"Exact Match Score (EM): {score_em:.4f}")


if __name__ == "__main__":
    """
    Examples:
    python openvivqa_eval_vivqa.py \
    --gt_file /root/projects/exp1/data/vivqa/vqa/test_en.json \
    --res_file /root/projects/exp1/vivqa_scoring/en/submit_vivqa_test_en_sorted.json
    

    python openvivqa_eval_vivqa.py \
    --phobert \
    --gt_file /root/projects/exp1/data/vivqa/vqa/test_vi.json \
    --res_file /root/projects/exp1/vivqa_scoring/vi/submit_vivqa_test_vi_sorted.json
    """
    opts = get_args()
    main(opts)