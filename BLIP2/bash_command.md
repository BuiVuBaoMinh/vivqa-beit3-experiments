```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun --nproc-per-node=1 train_blip2.py \
    --batch_size 1 \
    --eval_batch_size 1 \
    --min_lr 1e-6 \
    --epochs 50 \
    --layer_decay 0.99 \
    --update_freq 1 \
    --warmup_epochs 0 \
    --data_path /home/21khac.dd/bm/data/vivqa \
    --output_dir /home/21khac.dd/bm/blip2-pb-with-adapter-2-train-embed-tokens \
    --num_workers=4 \
    --weight_decay 0.05 \
    --save_ckpt_freq 1 \
    --task_head_lr_weight 0 \
    --opt_betas 0.9 0.98 \
    --early_stopping "val_score" \
    --patience 5 \
    --lr_sched_type cos \
    --answer2label "/home/21khac.dd/bm/data/vivqa/annotations/dicts/answer2label_en_gemini_translated.txt" \
    --device "cuda:0" \
    --phobert \
    --resume "latest" \
    --no_resume_optimizer \
    --freeze_embed_tokens


    --staged_training \
```

```bash
torchrun --nproc-per-node=1 train_blip2.py \
    --batch_size 1 \
    --eval_batch_size 1 \
    --data_path /home/21khac.dd/bm/data/vivqa \
    --output_dir /home/21khac.dd/bm/blip2-pb-with-adapter-2-train-embed-tokens \
    --answer2label "/home/21khac.dd/bm/data/vivqa/annotations/dicts/answer2label_en_gemini_translated.txt" \
    --device "cuda:0" \
    --eval \
    --resume "best" \
    --phobert
```