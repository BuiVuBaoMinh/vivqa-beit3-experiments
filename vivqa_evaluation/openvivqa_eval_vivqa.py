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

def get_args():
    parser = argparse.ArgumentParser('OpenViVQA eval P, R, F1 for vivqa.', add_help=False)

    parser.add_argument('--gt_file', required=True, type=str,
                        help="Path to ground truth file (e.g test.json)")
    parser.add_argument('--res_file', required=True, type=str,
                        help="Path to result/prediction file (e.g submit_vivqa_test.json)")
    parser.add_argument('--phobert', action='store_true', default=False,
                        help="Use PhoBERT's tokenizer.")
    
    return parser.parse_args()

def main(args):

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

    # Convert to dicts with id as keys
    if args.phobert:
        py_vncorenlp.download_model(save_dir='/root/projects/exp1/vncorenlp')
        rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir='/root/projects/exp1/vncorenlp')
        # TODO: finish this

    gts = {}
    for item in gt_data:
        gts[item["id"]] = tokenizer.tokenize(item["answer"].lower())

    res = {}
    for item in res_data:
        res[item["question_id"]] = tokenizer.tokenize(item["answer"].lower())

    scorer = Precision()
    score, details = scorer.compute_score(gts, res)
    print(f"Token-based Precision Score (P): {score:.4f}")

    scorer = Recall()
    score, details = scorer.compute_score(gts, res)
    print(f"Token-based Recall Score (R): {score:.4f}")

    scorer = F1()
    score, details = scorer.compute_score(gts, res)
    print(f"Token-based F1 Score (F1): {score:.4f}")


if __name__ == "__main__":
    opts = get_args()
    main(opts)