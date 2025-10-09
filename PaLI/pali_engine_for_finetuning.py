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
            max_length=4,
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
            max_length=4,
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
                # "meters": {k: meter.global_avg for k, meter in self.metric_logger.meters.items()},
                "loss": self.metric_logger.loss.global_avg
            }
        else:
            return {
                "score": self.metric_logger.score.global_avg,
                # "meters": {k: meter.global_avg for k, meter in self.metric_logger.meters.items()},
                "loss": self.metric_logger.loss.global_avg
            }

class PaLIClassificationHandler(object):
    """
    Handles training and evaluation for a PaLI-based classification model.
    Uses Accuracy as the primary metric and avoids slow text generation during evaluation.
    """
    def __init__(self) -> None:
        super().__init__()
        self.predictions = []
        self.id2label = None # Will be populated with the mapping from the dataset

    def train_batch(self, model, batch_size, pixel_values, input_ids, attention_mask, labels, qid=None):
        """
        Performs a single training step.
        """

        if torch.any(labels < 0) or torch.any(labels >= len(model.label2answer)-1):
            print(f"Invalid of UNKNOWN labels detected WHILE TRAINING! Min: {labels.min().item()}, Max: {labels.max().item()}")
            print(f"Labels: {labels.cpu().tolist()}")
            print(f"Model labels: 0..{len(model.label2answer)}")
            raise ValueError("Label out of bounds for logits")


        # Check shapes and types before forward pass
        # print(">>> Debug: batch_size =", batch_size)
        # print(">>> Debug: pixel_values shape =", pixel_values.shape)
        # print(">>> Debug: input_ids shape =", input_ids.shape)
        # print(">>> Debug: attention_mask shape =", attention_mask.shape)
        # print(">>> Debug: labels shape =", labels.shape)
        # print(">>> Debug: labels dtype =", labels.dtype)

        try:
            # Get model output from a single forward pass
            outputs = model(
                batch_size=batch_size,
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss

            # Check for NaNs or Infs in the loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(">>> Warning: NaN or Inf in loss!")
                raise ValueError("Loss is NaN or Inf")

            # Check for logits shape
            # print(">>> Debug: logits shape =", outputs.logits.shape)

            # Get predictions by finding the class with the highest logit
            pred_ids = torch.argmax(outputs.logits, dim=-1)

            # Check if label values are in a valid range (e.g., for classification)
            if torch.any(labels < 0) or torch.any(labels >= outputs.logits.shape[-1]):
                print(f">>> ERROR: Label values out of range! Label max = {labels.max()}, num_classes = {outputs.logits.shape[-1]}")
                raise ValueError("Invalid label value")

            accuracy = (pred_ids == labels).float().mean() * 100.0

            return {
                "loss": loss,
                "score": accuracy
            }
        except Exception as e:
            print(f">>> Exception caught in train_batch: {str(e)}")
            print(">>> Labels:", labels)
            print(">>> Logits (sample):", outputs.logits[0] if hasattr(outputs, 'logits') else "N/A")
            raise

    def before_eval(self, metric_logger, data_loader, **kwargs):
        """Prepares for an evaluation run."""
        self.predictions.clear()
        self.metric_logger = metric_logger
        # Get the id->label mapping to decode predictions for qualitative analysis
        self.id2label = data_loader.dataset.label_to_answer


    def eval_batch(self, model, batch_size, pixel_values, input_ids, attention_mask, labels, qid):
        """
        Performs a single evaluation step for one batch.
        """

        # Check shapes and types before forward pass
        # print(">>> Debug: batch_size =", batch_size)
        # print(">>> Debug: pixel_values shape =", pixel_values.shape)
        # print(">>> Debug: input_ids shape =", input_ids.shape)
        # print(">>> Debug: attention_mask shape =", attention_mask.shape)
        # print(">>> Debug: labels shape =", labels.shape)
        # print(">>> Debug: labels dtype =", labels.dtype)

        
        if torch.any(labels < 0) or torch.any(labels > len(model.label2answer)-1):
            print(f"Invalid or UNKNOWN labels detected WHILE EVALUATING! Min: {labels.min().item()}, Max: {labels.max().item()}")
            print(f"Labels: {labels.cpu().tolist()}")
            print(f"Model labels: 0..{len(model.label2answer)}")
            raise ValueError("Label out of bounds for logits")

        outputs = model(
            batch_size=batch_size,
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        batch_size = input_ids.shape[0]

        # Get predictions via argmax
        pred_ids = torch.argmax(outputs.logits, dim=-1)
        
        # Calculate accuracy
        accuracy = (pred_ids == labels).float().mean() * 100.0

        # Update metric loggers
        self.metric_logger.meters['score'].update(accuracy.item(), n=batch_size)
        self.metric_logger.meters['loss'].update(loss.item(), n=batch_size)

        # We convert the predicted ID back to a string answer for readability.
        for question_id, predicted_id in zip(qid, pred_ids):
            pred_answer = self.id2label.get(predicted_id.item(), "[UNK]") # Failsafe
            self.predictions.append({
                "question_id": question_id.item(),
                "answer": pred_answer
            })
            
    def after_eval(self, return_preds: bool = True, **kwargs):
        """Finalizes the evaluation run and returns results."""
        # The metric_logger already has the global average score and loss
        print(f'* Score {self.metric_logger.score.global_avg:.3f} | Loss {self.metric_logger.loss.global_avg:.3f}')
        
        results = {
            "score": self.metric_logger.score.global_avg,
            "loss": self.metric_logger.loss.global_avg
        }
        if return_preds:
            results["prediction"] = self.predictions
        
        return results
    
@torch.no_grad()
def pali_evaluate(data_loader, model, tokenizer, device, handler, return_preds: bool = True, num_beams=1, batch_size=8):
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

        if isinstance(handler, PaLIClassificationHandler):
                handler.eval_batch(
                    model=model,
                    batch_size=batch_size,
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    qid=qid,
                )
        elif isinstance(handler, PaLIHandler):
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