```bash
torchrun --nproc-per-node=1 train_blip.py \
    --batch_size 8 \
    --eval_batch_size 8 \
    --epochs 50 \
    --layer_decay 0.99 \
    --update_freq 1 \
    --warmup_epochs 0 \
    --data_path /home/21khac.dd/bm/data/vivqa \
    --output_dir /home/21khac.dd/bm/blip-pb-5 \
    --num_workers=10 \
    --weight_decay 0.05 \
    --save_ckpt_freq 1 \
    --task_head_lr_weight 0 \
    --opt_betas 0.9 0.98 \
    --early_stopping "val_score" \
    --patience 6 \
    --lr_sched_type cos \
    --answer2label "/home/21khac.dd/bm/data/vivqa/annotations/dicts/answer2label_en_gemini_translated.txt" \
    --device "cuda:5" \
    --phobert \
    --freeze_word_embeddings

    --staged_training \
```

```bash
torchrun --nproc-per-node=1 train_blip.py \
    --batch_size 8 \
    --eval_batch_size 8 \
    --data_path /home/21khac.dd/bm/data/vivqa \
    --output_dir /home/21khac.dd/bm/blip-pb-5 \
    --staged_training \
    --answer2label "/home/21khac.dd/bm/data/vivqa/annotations/dicts/answer2label_en_gemini_translated.txt" \
    --device "cuda:5" \
    --eval \
    --resume "best" \
    --phobert
```