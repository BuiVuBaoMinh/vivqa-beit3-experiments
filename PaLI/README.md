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