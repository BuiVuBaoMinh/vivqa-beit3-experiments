import math
import torch
from tqdm import tqdm
import utils

class Blip2VQAHandler:
    """
    Handles training and evaluation logic for the Blip2VQAClassification model.
    """
    def __init__(self):
        self.predictions = []
        self.id2label = None
        self.metric_logger = None

    def train_batch(
        self, model, 
        pixel_values, 
        input_ids, attention_mask, 
        decoder_input_ids, decoder_attention_mask,
        labels, qid=None
    ):
        """Performs a single training step."""
        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids = decoder_input_ids,
            decoder_attention_mask = decoder_attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        if not torch.isfinite(loss):
            raise ValueError(f"Loss is {loss.item()}, stopping training")
        
        pred_ids = torch.argmax(outputs.logits, dim=-1)
        accuracy = (pred_ids == labels).float().mean() * 100.0
        return {"loss": loss, "score": accuracy}

    def before_eval(self, metric_logger, data_loader):
        """Prepares for an evaluation run."""
        self.predictions.clear()
        self.metric_logger = metric_logger
        self.id2label = data_loader.dataset.label_to_answer

    def eval_batch(
        self, model, 
        pixel_values, 
        input_ids, attention_mask, 
        decoder_input_ids, decoder_attention_mask,
        labels, qid
    ):
        """Performs a single evaluation step."""
        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )
        
        loss = outputs.loss
        batch_size = input_ids.shape[0]
        pred_ids = torch.argmax(outputs.logits, dim=-1)
        accuracy = (pred_ids == labels).float().mean() * 100.0

        self.metric_logger.meters['score'].update(accuracy.item(), n=batch_size)
        self.metric_logger.meters['loss'].update(loss.item(), n=batch_size)

        for question_id, predicted_id in zip(qid, pred_ids):
            pred_answer = self.id2label.get(predicted_id.item(), "[UNK]")
            self.predictions.append({"question_id": question_id, "answer": pred_answer})
            
    def after_eval(self, return_preds: bool = True):
        """Finalizes the evaluation run."""
        print(f'* Score {self.metric_logger.score.global_avg:.3f} | Loss {self.metric_logger.loss.global_avg:.3f}')
        
        results = {
            "score": self.metric_logger.score.global_avg,
            "loss": self.metric_logger.loss.global_avg
        }
        if return_preds:
            results["prediction"] = self.predictions
        
        return results

@torch.no_grad()
def blip2_evaluate(args, data_loader, model, device, handler: Blip2VQAHandler, return_preds: bool = True):
    metric_logger = utils.MetricLogger(delimiter="  ")
    model.eval()
    handler.before_eval(metric_logger=metric_logger, data_loader=data_loader)

    for batch in tqdm(data_loader, desc="Evaluating", leave=False):
        if batch is None: continue
        
        pixel_values = batch["pixel_values"].to(device, dtype=torch.bfloat16)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        decoder_input_ids = batch["decoder_input_ids"].to(device=device)
        decoder_attention_mask = batch["decoder_attention_mask"].to(device=device)
        labels = batch["labels"].to(device)
        qid = batch["qid"]

        handler.eval_batch(
            model=model,
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            qid=qid
        )

    return handler.after_eval(return_preds=return_preds)
