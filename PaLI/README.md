```bash
torchrun --nproc-per-node=1 train_pali.py \
    --batch_size 10 \
    --data_path /home/21khac.dd/bm/data/vivqa \
    --output_dir /home/21khac.dd/bm/output-dir \
    --log_dir /home/21khac.dd/bm/output-dir/output-log \
    --resume "best" \
    --eval \
    --eval_num_beams 5
```

# PaLI-PhoBERT train
```bash
torchrun --nproc-per-node=1 train_pali.py \
    --batch_size 8 \
    --epochs 100 \
    --layer_decay 1 \
    --lr 6e-5 \
    --min_lr 1e-5 \
    --update_freq 1 \
    --warmup_epochs 1 \
    --data_path /home/21khac.dd/bm/data/vivqa \
    --output_dir /home/21khac.dd/bm/pali-pb-3 \
    --num_workers=10 \
    --weight_decay 0.01 \
    --save_ckpt_freq 1 \
    --task_head_lr_weight 20 \
    --opt_betas 0.9 0.98 \
    --early_stopping "val_score" \
    --patience 50 \
    --phobert \
    --lr_sched_type cos \
    --resume "latest" 
    --no_resume_optimizer
```

PaLI-PhoBERT eval
```bash
torchrun --nproc-per-node=1 train_pali.py \
    --batch_size 8 \
    --data_path /home/21khac.dd/bm/data/vivqa \
    --output_dir /home/21khac.dd/bm/pali-pb-3 \
    --resume "best" \
    --eval \
    --eval_num_beams 1 \
    --phobert
```