```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun --nproc-per-node=1 train_smolvlm.py \
    --batch_size 2 \
    --eval_batch_size 1 \
    --epochs 50 \
    --layer_decay 0.99 \
    --update_freq 1 \
    --warmup_epochs 0 \
    --data_path /home/21khac.dd/bm/data/vivqa \
    --output_dir /home/21khac.dd/bm/smolvlm-1 \
    --num_workers=10 \
    --weight_decay 0.05 \
    --save_ckpt_freq 1 \
    --task_head_lr_weight 0 \
    --opt_betas 0.9 0.98 \
    --early_stopping "val_score" \
    --patience 5 \
    --lr_sched_type cos \
    --answer2label "/home/21khac.dd/bm/data/vivqa/annotations/dicts/answer2label_en_gemini_translated.txt" \
    --device "cuda:5" \
    --resume "latest"
    
    --phobert \
    --freeze_embed_tokens

    --staged_training \
```

```bash
torchrun --nproc-per-node=1 train_paligemma.py \
    --batch_size 8 \
    --eval_batch_size 8 \
    --data_path /home/21khac.dd/bm/data/vivqa \
    --output_dir /home/21khac.dd/bm/paligemma-2\
    --staged_training \
    --answer2label "/home/21khac.dd/bm/data/vivqa/annotations/dicts/answer2label_en_gemini_translated.txt" \
    --device "cuda:5" \
    --eval \
    --resume "best" \
    --phobert
```