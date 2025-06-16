import torch
import sys
import os
import json

from torchmetrics import Metric

from utils import get_rank, get_world_size

class PaLIF1Score(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, predictions, references):
        """
        :param predictions: list of predicted strings
        :param references: list of ground truth strings
        """
        assert len(predictions) == len(references)

        for pred, ref in zip(predictions, references):
            pred_tokens = pred.strip().lower().split()
            ref_tokens = ref.strip().lower().split()

            # Edge case: one or both are empty
            if len(pred_tokens) == 0 or len(ref_tokens) == 0:
                f1 = 1.0 if pred_tokens == ref_tokens else 0.0
            else:
                common = set(pred_tokens) & set(ref_tokens)
                if len(common) == 0:
                    f1 = 0.0
                else:
                    prec = len(common) / len(pred_tokens)
                    rec = len(common) / len(ref_tokens)
                    f1 = 2 * (prec * rec) / (prec + rec)

            self.score += f1
            self.total += 1


    def compute(self):
        return self.score / self.total if self.total > 0 else torch.tensor(0.0)



def pali_dump_predictions(args, result, file_suffix):
    global_rank = get_rank()
    jsons = None
    if global_rank >= 0:
        output_file = os.path.join(args.task_cache_path, f"submit_{global_rank}_{file_suffix}.json")
        with open(output_file, "w") as fp:
            json.dump(result, fp, indent=2)
        torch.distributed.barrier()

        if global_rank == 0:
            world_size = get_world_size()
            jsons = []
            for i in range(world_size):
                each_file = os.path.join(args.task_cache_path, f"submit_{i}_{file_suffix}.json")
                with open(each_file, "r") as fp:
                    jsons += json.load(fp)
            
            new_jsons = []

            sys.exit(0)
            res_dict = dict()
            qid_key = "question_id"
            for item in jsons:
                if item[qid_key] in res_dict:
                    continue
                new_jsons.append(item)
                res_dict[item[qid_key]] = item
            jsons = new_jsons

        torch.distributed.barrier()
        os.remove(output_file)
    else:
        jsons = result
    
    result_file = os.path.join(args.output_dir, f"submit_{file_suffix}.json")
    if jsons is not None:
        with open(result_file, "w") as fp:
            json.dump(jsons, fp, indent=2, ensure_ascii=False)
        print("Infer %d examples into %s" % (len(jsons), result_file))
    return result_file