import math
import sys
import json
from typing import Iterable, Optional

import torch
import torch.nn as nn

from tqdm import tqdm


from timm.utils import ModelEma

import utils

from pali_utils import PaLIF1Score

class PaLIHandler(object):
    def __init__(self) -> None:
        super().__init__()
        self.predictions = []
        self.label2ans = None

    def train_batch(self, model, tokenizer, pixel_values, input_ids, attention_mask, labels, qid=None):
        outputs = model(
            pixel_values = pixel_values,
            input_ids = input_ids,
            attention_mask = attention_mask,
            labels = labels
        )

        logits = outputs.logits
        loss = outputs.loss

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

        batch_size = input_ids.shape[0]

        scores = PaLIF1Score()(pred_texts, label_texts) * 100.0

        return {
            # "loss": self.criterion(input=logits.float(), target=labels.float()) * labels.shape[1], 
            "loss": loss,
            "score": scores
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

        scores = PaLIF1Score()(pred_texts, label_texts) * 100.0
        self.metric_logger.meters['score'].update(scores.item(), n=batch_size)

        loss = outputs.loss
        self.metric_logger.meters['loss'].update(loss.item(), n=batch_size)

        for question_id, pred in zip(qid, pred_texts):
            self.predictions.append({
                "question_id": question_id.item(),
                "answer": pred
            })

    def after_eval(self, return_preds: bool = True, **kwargs):
        # if len(self.predictions) == 0:
        #     print('* Score {score.global_avg:.3f}'.format(score=self.metric_logger.score))
        #     return {k: meter.global_avg for k, meter in self.metric_logger.meters.items()}, "score"
        # else:
        #     return self.predictions, "prediction"

        if return_preds:
            return {
                "prediction": self.predictions,
                "score": self.metric_logger.score.global_avg,
                "meters": {k: meter.global_avg for k, meter in self.metric_logger.meters.items()},
            }
        else:
            return {
                "score": self.metric_logger.score.global_avg,
                "meters": {k: meter.global_avg for k, meter in self.metric_logger.meters.items()},
            }

@torch.no_grad()
def pali_evaluate(data_loader, model, tokenizer, device, handler: PaLIHandler, return_preds: bool = True):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    handler.before_eval(metric_logger=metric_logger, data_loader=data_loader)

    for step, batch in enumerate(tqdm(data_loader, desc=f"Test: ", leave=False)):
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        qid = batch["qid"]

        handler.eval_batch(
            model=model,
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            qid=qid
        )

    return handler.after_eval(return_preds = return_preds)