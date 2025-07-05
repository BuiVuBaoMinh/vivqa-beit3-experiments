import math
import sys
import json
from typing import Iterable, Optional

import torch
import torch.nn as nn

from tqdm import tqdm


from timm.utils import ModelEma

import utils

class PaligemmaHandler(object):
    """
    Handles training and evaluation for a PaLI-based classification model.
    Uses Accuracy as the primary metric and avoids slow text generation during evaluation.
    """
    def __init__(self) -> None:
        super().__init__()
        self.predictions = []
        self.id2label = None # Will be populated with the mapping from the dataset

    def train_batch(
        self,
        model,
        input_ids,
        pixel_values,
        attention_mask,
        token_type_ids,
        paligemma_labels, # Pass to Paligemma
        labels, # Use for our classification
        qid=None
    ):
        """
        Performs a single training step.
        """

        if torch.any(labels < 0) or torch.any(labels >= len(model.label2answer)-1):
            print(f"Invalid or UNKNOWN labels detected WHILE TRAINING! Min: {labels.min().item()}, Max: {labels.max().item()}")
            print(f"Labels: {labels.cpu().tolist()}")
            print(f"Model labels: 0..{len(model.label2answer)}")
            raise ValueError("Label out of bounds for logits")
        
        try:
            # Get model output from a single forward pass
            outputs = model(
                input_ids = input_ids,
                pixel_values = pixel_values,
                attention_mask = attention_mask,
                token_type_ids = token_type_ids,
                paligemma_labels = paligemma_labels,
                labels=labels,
            )

            loss = outputs.loss

            # Check for NaNs or Infs in the loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(">>> Warning: NaN or Inf in loss!")
                raise ValueError("Loss is NaN or Inf")
            
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


    def eval_batch(
        self, 
        model, 
        batch_size, 
        pixel_values, 
        input_ids, 
        attention_mask, 
        token_type_ids,
        paligemma_labels, # Pass to Paligemma
        labels, 
        qid
    ):
        """
        Performs a single evaluation step for one batch.
        """
        
        if torch.any(labels < 0) or torch.any(labels > len(model.label2answer)-1):
            print(f"Invalid labels detected WHILE EVALUATING! Min: {labels.min().item()}, Max: {labels.max().item()}")
            print(f"Labels: {labels.cpu().tolist()}")
            print(f"Model labels: 0..{len(model.label2answer)}")
            raise ValueError("Label out of bounds for logits")

        outputs = model(
            input_ids = input_ids,
            pixel_values = pixel_values,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            paligemma_labels = paligemma_labels,
            labels=labels,
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
def paligemma_evaluate(data_loader, model, tokenizer, device, handler: PaligemmaHandler, return_preds: bool = True):
    metric_logger = utils.MetricLogger(delimiter="  ")

    # switch to evaluation mode
    model.eval()
    handler.before_eval(metric_logger=metric_logger, data_loader=data_loader)

    for step, batch in enumerate(tqdm(data_loader, desc=f"Test: ", leave=False)):
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        paligemma_labels = batch['paligemma_labels'].to(device)
        labels = batch["labels"].to(device)
        qid = batch["qid"]

        handler.eval_batch(
            model=model,
            batch_size=8,
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            paligemma_labels=paligemma_labels,
            labels=labels,
            qid=qid
        )

    return handler.after_eval(return_preds = return_preds)