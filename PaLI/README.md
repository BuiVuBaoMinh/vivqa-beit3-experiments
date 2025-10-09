# PaLI-Phobert normal train
```bash
torchrun --nproc-per-node=1 train_pali.py \
    --batch_size 8 \
    --epochs 100 \
    --layer_decay 1 \
    --lr 6e-5 \
    --min_lr 2e-5 \
    --update_freq 1 \
    --warmup_epochs 1 \
    --data_path /home/21khac.dd/bm/data/vivqa \
    --output_dir /home/21khac.dd/bm/pali-pb-4 \
    --num_workers=10 \
    --weight_decay 0.01 \
    --save_ckpt_freq 1 \
    --task_head_lr_weight 20 \
    --opt_betas 0.9 0.98 \
    --early_stopping "val_score" \
    --patience 5 \
    --phobert \
    --lr_sched_type cos \
    --resume "latest"
```

# PaLI-PhoBERT staged train
```bash
torchrun --nproc-per-node=1 pali_staged_training.py \
    --batch_size 8 \
    --eval_batch_size 8 \
    --epochs 70 \
    --layer_decay 1 \
    --update_freq 1 \
    --warmup_epochs 0 \
    --data_path /home/21khac.dd/bm/data/vivqa \
    --output_dir /home/21khac.dd/bm/pali-classification-3-gemini-answer2label \
    --num_workers=10 \
    --weight_decay 0.05 \
    --save_ckpt_freq 1 \
    --task_head_lr_weight 20 \
    --opt_betas 0.9 0.98 \
    --early_stopping "val_score" \
    --patience 6 \
    --lr_sched_type cos \
    --staged_training \
    --pali_class pali_classification \
    --answer2label "/home/21khac.dd/bm/data/vivqa/annotations/dicts/answer2label_en_gemini_translated.txt" \
    --device "cuda:6" \
    --phobert \
    --resume "latest" \
    --no_resume_optimizer \

```

PaLI-PhoBERT eval
```bash
torchrun --nproc-per-node=1 pali_staged_training.py \
    --batch_size 8 \
    --eval_batch_size 8 \
    --data_path /home/21khac.dd/bm/data/vivqa \
    --output_dir /home/21khac.dd/bm/pali-classification-3-gemini-answer2label \
    --resume "best" \
    --eval \
    --eval_num_beams 1 \
    --staged_training \
    --pali_class pali_classification \
    --answer2label "/home/21khac.dd/bm/data/vivqa/annotations/dicts/answer2label_en_gemini_translated.txt" \
    --device "cuda:5" \
    --phobert
```

```bash
torchrun --nproc-per-node=1 pali_staged_training.py \
    --batch_size 8 \
    --eval_batch_size 8 \
    --data_path /home/21khac.dd/bm/data/vivqa \
    --output_dir /home/21khac.dd/bm/pali-pb-4 \
    --resume "best" \
    --eval \
    --eval_num_beams 1 \
    --staged_training \
    --pali_class pali_generative \
    --device "cuda:5" \
    --phobert
```