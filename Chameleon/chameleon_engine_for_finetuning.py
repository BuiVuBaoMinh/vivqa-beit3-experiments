import math
import sys
from typing import Optional

import torch
from tqdm import tqdm

import utils

class ChameleonVQAHandler(object):
    """
    Handles training and evaluation logic for the Chameleon VQA classification model.
    """
    def __init__(self) -> None:
        self.predictions = []
        self.id2label = None
        self.metric_logger = None

    def train_batch(
        self,
        model,
        input_ids,
        pixel_values,
        attention_mask,
        labels,
    ):
        """Performs a single training step."""
        try:
            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )

            loss = outputs.loss
            if not torch.isfinite(loss):
                raise ValueError("Loss is NaN or Inf")
            
            pred_ids = torch.argmax(outputs.logits, dim=-1)
            accuracy = (pred_ids == labels).float().mean() * 100.0

            return {"loss": loss, "score": accuracy}
        except Exception as e:
            print(f"Exception caught in train_batch: {e}")
            raise

    def before_eval(self, metric_logger, data_loader, **kwargs):
        """Prepares for an evaluation run."""
        self.predictions.clear()
        self.metric_logger = metric_logger
        self.id2label = data_loader.dataset.label_to_answer

    def eval_batch(
        self, 
        model, 
        pixel_values, 
        input_ids, 
        attention_mask, 
        labels, 
        qid
    ):
        """Performs a single evaluation step for one batch."""
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        
        loss = outputs.loss
        batch_size = input_ids.shape[0]
        pred_ids = torch.argmax(outputs.logits, dim=-1)
        accuracy = (pred_ids == labels).float().mean() * 100.0

        self.metric_logger.meters['score'].update(accuracy.item(), n=batch_size)
        self.metric_logger.meters['loss'].update(loss.item(), n=batch_size)

        for question_id, predicted_id in zip(qid, pred_ids):
            pred_answer = self.id2label.get(predicted_id.item(), "[UNK]")
            self.predictions.append({
                "question_id": question_id,
                "answer": pred_answer
            })
            
    def after_eval(self, return_preds: bool = True, **kwargs):
        """Finalizes the evaluation run and returns results."""
        print(f'* Score {self.metric_logger.score.global_avg:.3f} | Loss {self.metric_logger.loss.global_avg:.3f}')
        
        results = {
            "score": self.metric_logger.score.global_avg,
            "loss": self.metric_logger.loss.global_avg
        }
        if return_preds:
            results["prediction"] = self.predictions
        
        return results
    
@torch.no_grad()
def chameleon_evaluate(args, data_loader, model, device, handler: ChameleonVQAHandler, return_preds: bool = True):
    metric_logger = utils.MetricLogger(delimiter="  ")
    model.eval()
    handler.before_eval(metric_logger=metric_logger, data_loader=data_loader)

    for batch in tqdm(data_loader, desc="Evaluating", leave=False):
        if batch is None: continue # Skip bad batches
        
        pixel_values = batch["pixel_values"].to(device=device, dtype=torch.bfloat16)
        input_ids = batch["input_ids"].to(device=device)
        attention_mask = batch["attention_mask"].to(device=device)
        labels = batch["labels"].to(device)
        qid = batch["qid"]

        handler.eval_batch(
            model=model,
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            qid=qid
        )

    return handler.after_eval(return_preds=return_preds)
