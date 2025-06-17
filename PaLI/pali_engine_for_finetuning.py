import math
import sys
import json
from typing import Iterable, Optional

import torch
import torch.nn as nn


from timm.utils import ModelEma

import utils

from pali_utils import PaLIF1Score

class PaLIHandler(object):
    def __init__(self) -> None:
        super().__init__()
        self.predictions = []
        self.label2ans = None

    def train_batch(self, model, pixel_values, input_ids, attention_mask, labels, qid=None):
        outputs = model(
            pixel_values = pixel_values,
            input_ids = input_ids,
            attention_mask = attention_mask,
            labels = labels
        )

        logits = outputs.logits
        loss = outputs.loss

        batch_size = input_ids.shape[0]
        # scores = PaLIF1Score()(logits, labels) * 100.0

        return {
            # "loss": self.criterion(input=logits.float(), target=labels.float()) * labels.shape[1], 
            "loss": loss,
        }

    def before_eval(self, metric_logger, data_loader, **kwargs):
        self.predictions.clear()
        self.metric_logger = metric_logger
        # self.label2ans = data_loader.dataset.label2ans

    def eval_batch(self, model, tokenizer, pixel_values, input_ids, attention_mask, labels, qid=None):
        outputs = model(
            pixel_values = pixel_values,
            input_ids = input_ids,
            attention_mask = attention_mask,
            labels = labels
        )
        logits = outputs.logits
        batch_size = input_ids.shape[0]

        loss = outputs.loss
        print(loss)
        sys.exit(0)

        generated_ids = model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=3,
            num_beams=1, # increase for better generation at cost of speed
        )

        # Decode predictions and labels
        pred_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # if labels is not None:
        scores = PaLIF1Score()(pred_texts, label_texts) * 100.0
        self.metric_logger.meters['score'].update(scores.item(), n=batch_size)

        loss = outputs.loss
        self.metric_logger.meters['loss'].update(loss.item(), n=batch_size)

        for question_id, pred in zip(qid, pred_texts):
            self.predictions.append({
                "question_id": question_id.item(),
                "answer": pred
            })



    def after_eval(self, **kwargs):
        if len(self.predictions) == 0:
            print('* Score {score.global_avg:.3f}'.format(score=self.metric_logger.score))
            return {k: meter.global_avg for k, meter in self.metric_logger.meters.items()}, "score"
        else:
            return self.predictions, "prediction"

@torch.no_grad()
def pali_evaluate(data_loader, model, tokenizer, device, handler: PaLIHandler):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    handler.before_eval(metric_logger=metric_logger, data_loader=data_loader)

    for data in metric_logger.log_every(data_loader, 10, header):
        for tensor_key in data.keys():
            data[tensor_key] = data[tensor_key].to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            handler.eval_batch(model=model, tokenizer=tokenizer, **data)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    return handler.after_eval()