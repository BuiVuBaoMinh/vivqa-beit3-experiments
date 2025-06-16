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

    def train_batch(self, model, pixel_values, input_ids, attention_mask, labels):
        outputs = model(
            pixel_values = pixel_values,
            input_ids = input_ids,
            attention_mask = attention_mask,
            labels = labels
        )

        logits = outputs.logits
        loss = outputs.loss

        batch_size = input_ids.shape[0]
        scores = PaLIF1Score()(logits, labels) * 100.0

        return {
            # "loss": self.criterion(input=logits.float(), target=labels.float()) * labels.shape[1], 
            "loss": loss,
            "scores": scores, \
            "batch_size": batch_size,
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
        
def pali_train_one_epoch(
        model: torch.nn.Module, data_loader: Iterable, 
        optimizer: torch.optim.Optimizer, device: torch.device, 
        handler: PaLIHandler, epoch: int, start_steps: int, 
        lr_schedule_values: list, loss_scaler, max_norm: float = 0, 
        update_freq: int = 1, model_ema: Optional[ModelEma] = None, 
        log_writer: Optional[utils.TensorboardLogger] = None, 
        task = None, mixup_fn=None,
):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()


    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        global_step = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[global_step] * param_group["lr_scale"]
        # put input data into cuda
        for tensor_key in data.keys():
            data[tensor_key] = data[tensor_key].to(device, non_blocking=True)
            # print("input %s = %s" % (tensor_key, data[tensor_key]))
            if loss_scaler is None and tensor_key.startswith("image"):
                data[tensor_key] = data[tensor_key].half()

        if loss_scaler is None:
            results = handler.train_batch(model, **data)
        else:
            with torch.cuda.amp.autocast():
                results = handler.train_batch(model, **data)

        loss = results.pop("loss")
        loss_value = loss.item()

        scores = results.pop("scores")
        batch_size = results.pop("batch_size")

        metric_logger.meters['score'].update(scores.item(), n=batch_size)

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = utils.get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            kwargs = {
                "loss": loss_value, 
            }
            for key in results:
                kwargs[key] = results[key]
            log_writer.update(head="train", **kwargs)

            kwargs = {
                "loss_scale": loss_scale_value, 
                "lr": max_lr, 
                "min_lr": min_lr, 
                "weight_decay": weight_decay_value, 
                "grad_norm": grad_norm, 
            }
            log_writer.update(head="opt", **kwargs)
            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


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