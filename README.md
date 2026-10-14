# Vietnamese Visual Question Answering with BEiT-3 and PhoBERT

This repository contains the thesis experiments on ViVQA, a Vietnamese visual question answering benchmark. It provides data preparation, fine-tuning, inference and evaluation scripts for BEiT-3 and for six other vision-language models used as comparisons.

All models treat ViVQA as closed-set answer classification: a head on top of the fused image-question representation predicts one answer from a fixed answer dictionary. Two settings are compared:

- **Translate-to-English:** questions and answers are machine-translated to English and the model's original tokenizer is used.
- **PhoBERT (`--phobert`):** the model's word embeddings are replaced with those of [PhoBERT](https://github.com/VinAIResearch/PhoBERT) (`vinai/phobert-base-v2`), so Vietnamese questions are fed in directly. When the hidden sizes differ, a linear or non-linear adapter projects the 768-d PhoBERT embeddings into the model's embedding space.

The BEiT-3 code builds on the public [BEiT-3 implementation](https://github.com/microsoft/unilm/tree/master/beit3) and extends it with a ViVQA dataset and task, PhoBERT embedding replacement, optional LoRA, and early stopping.

## Repository Contents

```
run_beit3_finetuning.py        BEiT-3 training and inference entry point
datasets.py                    Dataset classes and index builders (ViVQADataset, OpenViVQADataset, upstream tasks)
engine_for_finetuning.py       Task handlers, training loop and evaluation loop
modeling_finetune.py           BEiT-3 task models and registered model configs (incl. *_vivqa)
modeling_utils.py              BEiT-3 base/large configs and wrapper
optim_factory.py               Optimizer and layer-wise learning-rate decay
utils.py                       Distributed setup, checkpointing, schedulers, VQA score, early-stopping state
my_utils.py                    Checkpoint rotation, resume and prediction dumps used by the other models
glossary.py                    Answer normalization (normalize_word, segment_normalize)
randaug.py                     RandAugment image augmentation

process_vivqa_data.py          Convert the original ViVQA CSVs to JSON; machine translation vi -> en
segment_vivqa.py               VnCoreNLP word segmentation for PhoBERT inputs
translate_openvivqa.py         Machine translation of OpenViVQA annotations
check_generated_beit3_jsonls.py  Sanity check of generated BEiT-3 index files against the source JSON
exp1.py                        Scratch script for building the BEiT-3 ViVQA index files
cal_train_time.py              Sum per-epoch training time from a log.txt
test-eval.py                   BLEU / METEOR / exact-match scoring of a prediction file

BLIP/                          BLIP (Salesforce/blip-vqa-base)
BLIP2/                         BLIP-2 (Salesforce/blip2-flan-t5-xl-coco)
PaLI/                          PaLI-style model built from ViT-B/16 + mT5-base (generative and classification)
paligemma/                     PaliGemma 2 (google/paligemma2-3b-pt-224) with LoRA
SmolVLM/                       SmolVLM2 (HuggingFaceTB/SmolVLM2-2.2B-Instruct)
Chameleon/                     Chameleon (facebook/chameleon-7b), experimental and incomplete

vivqa_evaluation/
  openvivqa_eval_vivqa.py      Token-level precision / recall / F1 and exact match
  scikit_learn_eval_vivqa.py   Label-level accuracy and macro precision / recall / F1
  plot_training_val_max.py     Plot train/val accuracy and loss curves from log.txt
```

Each model directory follows the same layout: `train_<model>.py` (entry point), `<model>_dataset.py`, `<model>_engine_for_finetuning.py`, a model definition file, and a `bash_commands.md` with the commands used for the experiments.

Datasets, pretrained weights, checkpoints and training logs are not included in this repository.

## Environment

The experiments were run on single CUDA GPUs. A CUDA device is required, because BEiT-3 evaluation runs under CUDA autocast.

```bash
python -m venv .venv
source .venv/bin/activate

# Install a PyTorch build that matches the local CUDA driver first.
pip install torch torchvision

# BEiT-3 (pins follow upstream BEiT-3)
pip install timm==0.4.12 torchscale==0.2.0 torchmetrics==0.7.3 tensorboardX loralib \
            transformers sentencepiece opencv-python Pillow numpy tqdm

# BLIP-2, SmolVLM, PaliGemma
pip install bitsandbytes peft

# Data preparation and evaluation
pip install pandas deep_translator py_vncorenlp nltk scikit-learn matplotlib
```

There is no single pinned environment file for the whole repository. `PaLI/requirements.txt` lists the packages used for the PaLI experiments. SmolVLM2 and PaliGemma 2 require a recent `transformers` release. `py_vncorenlp` requires Java.

PaliGemma 2 and Chameleon are gated on Hugging Face. Accept their licenses and run `huggingface-cli login` before downloading them.

### Paths to edit before running

Several scripts contain absolute paths from the original training servers:

- `MY_CACHE_DIR` in `BLIP2/my_blip2.py`, `SmolVLM/mySmolVLM.py`, `paligemma/paligemma.py` and `Chameleon/myChameleon.py`. SmolVLM and PaliGemma load with `local_files_only=True`, so their weights must already be in this cache.
- The default `--data_path` of the non-BEiT-3 training scripts (`/root/projects/exp1/data/vivqa`). Pass `--data_path` explicitly.
- The VnCoreNLP directory in `segment_vivqa.py` and `vivqa_evaluation/openvivqa_eval_vivqa.py`, and the `beit3.spm` path in `vivqa_evaluation/openvivqa_eval_vivqa.py`.
- The input paths in `process_vivqa_data.py`, `PaLI/generate_pali_val_splits.py` and `cal_train_time.py`.

## Data Layout

Prepare data under `data/vivqa/`:

```
data/vivqa/
  answer2label.txt                     BEiT-3 answer dictionary
  vivqa.train.jsonl                    BEiT-3 index files (see "BEiT-3 index files")
  vivqa.trainable_val.jsonl
  vivqa.test.jsonl
  images/
    train/<image_id>.jpg
    test/<image_id>.jpg
  vqa/
    train_vi_full.json   test_vi.json  Vietnamese QA pairs
    train_en_full.json   test_en.json  English translation
    {train,val,test}_en.json           Translate-to-English runs
    {train,val,test}_vi.json           PaLI generative with --phobert
    {train,val,test}_vi_en_ans.json    Vietnamese question + English answer (classification with --phobert)
  annotations/dicts/
    answer2label.txt
    answer2label_en_gemini_translated.txt
    combined_answers.txt
```

QA files are JSON lists of `{"id": int, "image": int, "question": str, "answer": str}`, where `image` is the image file name without `.jpg`.

Answer dictionaries are JSONL files with one `{"answer": str, "label": int}` object per line:

- `answer2label.txt` holds the Vietnamese answers used by BEiT-3.
- `answer2label_en_gemini_translated.txt` holds the English answers used by BLIP, PaLI, PaliGemma and SmolVLM.
- `combined_answers.txt` is used by BLIP-2 and also needs a `vi_answer` field.

The non-BEiT-3 models normalize answers with `glossary.segment_normalize`, remove duplicates, re-index labels from 0, and add an `UNKNOWN` class.

### Preparation scripts

1. `process_vivqa_data.py` converts the original ViVQA `train.csv` / `test.csv` into JSON and machine-translates questions and answers from Vietnamese to English.
2. `segment_vivqa.py` word-segments questions and answers with VnCoreNLP, as PhoBERT expects segmented input.
3. `PaLI/generate_pali_val_splits.py` holds out 1,000 random training examples as `val_{en,vi}.json`.

The `{split}_vi_en_ans.json` files are not produced by a script in this repository.

### BEiT-3 index files

BEiT-3 reads pre-tokenized JSONL index files. Build them from the repository root:

```python
from transformers import AutoTokenizer, XLMRobertaTokenizer
from datasets import ViVQADataset  # the local datasets.py

ViVQADataset.make_dataset_index(
    data_path="data/vivqa",
    tokenizer_spm=XLMRobertaTokenizer("/path/to/beit3.spm"),
    tokenizer_phobert=AutoTokenizer.from_pretrained("vinai/phobert-base-v2"),
    annotation_data_path="data/vivqa/vqa",
    predefined_dict_path="data/vivqa/annotations/dicts/answer2label.txt",
)
```

This writes two sets of index files:

- `vivqa_vi.*.jsonl`: Vietnamese questions tokenized with PhoBERT.
- `vivqa_en.*.jsonl`: English questions tokenized with the BEiT-3 sentencepiece model.

The training script reads `vivqa.train.jsonl`, `vivqa.trainable_val.jsonl` and `vivqa.test.jsonl`. Copy or symlink the set that matches the run: `vivqa_vi.*` with `--phobert`, `vivqa_en.*` without it. Then place the answer dictionary at `data/vivqa/answer2label.txt`.

To check that a generated file matches its source annotations, set the paths at the top of `check_generated_beit3_jsonls.py` and run it.

## Training Examples

### BEiT-3

Download `beit3.spm` and a pretrained checkpoint (e.g. `beit3_base_patch16_224.pth`) from the [BEiT-3 repository](https://github.com/microsoft/unilm/tree/master/beit3).

The `--model` name is combined with `--task` to select the registered config: `beit3_base_patch16_480` + `vivqa` gives `beit3_base_patch16_480_vivqa`, which has 308 answer classes.

BEiT-3 with PhoBERT embeddings:

```bash
python run_beit3_finetuning.py \
  --model beit3_base_patch16_480 \
  --input_size 480 \
  --task vivqa \
  --phobert \
  --sentencepiece_model /path/to/beit3.spm \
  --finetune /path/to/beit3_base_patch16_224.pth \
  --data_path data/vivqa \
  --output_dir output/beit3-vivqa-phobert \
  --batch_size 16 \
  --lr 3e-5 \
  --layer_decay 1.0 \
  --task_head_lr_weight 20 \
  --weight_decay 0.01 \
  --opt_betas 0.9 0.98 \
  --epochs 10 \
  --warmup_epochs 1 \
  --drop_path 0.1 \
  --randaug \
  --save_ckpt_freq 5 \
  --early_stopping val_score \
  --patience 5 \
  --seed 42
```

The optimizer and schedule values above follow the upstream BEiT-3 VQAv2 recipe; adjust them to the run being reproduced.

- **Translate-to-English baseline:** drop `--phobert` and link the `vivqa_en.*` index files.
- **LoRA:** add `--enable_lora`. Optional flags: `--lora_r`, `--lora_alpha`, `--lora_dropout`, `--lora_target_modules`.
- **Multiple GPUs:** launch with `torchrun --nproc-per-node=<N> run_beit3_finetuning.py ...`.

**Outputs:**
- Checkpoints go to `<output_dir>/checkpoints/`, and the best one is `checkpoint-best.pth`.
- Per-epoch metrics are appended to `<output_dir>/log.txt`.
- Each time the validation score improves, test-set predictions are rewritten to `<output_dir>/submit_vivqa_test.json`.

Training resumes automatically from the latest checkpoint in `<output_dir>/checkpoints/`, and the early-stopping state is restored from `log.txt`.

### Other models

The other models are trained from their own directories. Each directory's `bash_commands.md` (or `README.md` for PaLI) has the full commands used. For example, BLIP with PhoBERT:

```bash
cd BLIP
torchrun --nproc-per-node=1 train_blip.py \
  --data_path ../data/vivqa \
  --output_dir ../output/blip-phobert \
  --answer2label ../data/vivqa/annotations/dicts/answer2label_en_gemini_translated.txt \
  --phobert \
  --batch_size 8 \
  --eval_batch_size 8 \
  --epochs 50 \
  --warmup_epochs 0 \
  --weight_decay 0.05 \
  --opt_betas 0.9 0.98 \
  --lr_sched_type cos \
  --save_ckpt_freq 1 \
  --early_stopping val_score \
  --patience 6 \
  --device cuda:0
```

| Directory | Entry point | Notes |
|---|---|---|
| `BLIP/` | `train_blip.py` | `--freeze_word_embeddings`; `--staged_training` (2 stages, with `--phobert`) |
| `BLIP2/` | `train_blip2.py` | PhoBERT embeddings projected 768 -> 2048 into Flan-T5-XL |
| `PaLI/` | `train_pali.py` (generative), `pali_staged_training.py` | `--pali_class {pali_generative,pali_classification}`; `--staged_training` (4 stages); `--eval_num_beams` |
| `paligemma/` | `train_paligemma.py` | LoRA (r=16, alpha=32) via `peft`; PhoBERT embeddings projected 768 -> 2304 |
| `SmolVLM/` | `train_smolvlm.py` | `--phobert_embedding_adapter {linear,non-linear}`; `--freeze_embed_tokens` |

Shared flags:

- `--resume latest|best` resumes from `<output_dir>/checkpoints/`.
- `--no_resume_optimizer` restores the weights only.
- `--early_stopping val_score|val_loss` and `--patience` control early stopping.
- `--min_lr`, `--warmup_epochs` and `--lr_sched_type cos|linear` control the schedule.

`PaLI/train_pali.py` uses `--lr`. The other scripts set a learning rate for each model component inside the training script. BLIP-2, SmolVLM and PaliGemma train in bf16 with 8-bit AdamW (`bitsandbytes`). For these large models, setting `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` reduces memory fragmentation.

## Inference

BEiT-3: add `--eval` and point `--resume` at a checkpoint. Predictions are written to `<output_dir>/submit_vivqa_test.json`.

```bash
python run_beit3_finetuning.py \
  --model beit3_base_patch16_480 --input_size 480 --task vivqa --phobert \
  --sentencepiece_model /path/to/beit3.spm \
  --data_path data/vivqa \
  --output_dir output/beit3-vivqa-phobert \
  --resume output/beit3-vivqa-phobert/checkpoints/checkpoint-best.pth \
  --eval
```

Other models: run the training script with `--eval --resume best`. Predictions are written to `<output_dir>/submit_vivqa_<model>_test.json`.

All prediction files are JSON lists of `{"question_id": int, "answer": str}`. Classification models trained with an English answer dictionary (e.g. `answer2label_en_gemini_translated.txt`) output English answers.

## Evaluation

Token-level precision, recall, F1 and exact match:

```bash
python vivqa_evaluation/openvivqa_eval_vivqa.py \
  --gt_file data/vivqa/vqa/test_vi.json \
  --res_file output/beit3-vivqa-phobert/submit_vivqa_test.json \
  --simple_tokenize
```

- `--simple_tokenize` lowercases and splits on whitespace.
- Without it, answers are tokenized with the BEiT-3 sentencepiece model.
- With `--phobert`, answers are tokenized with PhoBERT after VnCoreNLP segmentation.

Label-level accuracy and precision / recall / F1:

```bash
python vivqa_evaluation/scikit_learn_eval_vivqa.py \
  --gt_file data/vivqa/vqa/test_vi.json \
  --res_file output/beit3-vivqa-phobert/submit_vivqa_test.json \
  --answer2label data/vivqa/answer2label.txt \
  --average macro
```

Test questions whose ground-truth answer is not in the answer dictionary are skipped by this script.

Use `test_en.json` as the ground truth for predictions in English.

To plot training curves:

```bash
python vivqa_evaluation/plot_training_val_max.py --log_path output/beit3-vivqa-phobert/log.txt
```

## License and Attribution

The BEiT-3 training pipeline (`run_beit3_finetuning.py`, `datasets.py`, `engine_for_finetuning.py`, `modeling_finetune.py`, `modeling_utils.py`, `optim_factory.py`, `utils.py`, `randaug.py`, `glossary.py`) is derived from [microsoft/unilm](https://github.com/microsoft/unilm/tree/master/beit3), released under the MIT License. Copyright notices are kept in the file headers.

The ViVQA dataset is from Tran et al., "ViVQA: Vietnamese Visual Question Answering" (PACLIC 2021).

```bibtex
@inproceedings{beit3,
  title={Image as a Foreign Language: {BEiT} Pretraining for Vision and Vision-Language Tasks},
  author={Wang, Wenhui and Bao, Hangbo and Dong, Li and Bjorck, Johan and Peng, Zhiliang and Liu, Qiang and Aggarwal, Kriti and Mohammed, Owais Khan and Singhal, Saksham and Som, Subhojit and Wei, Furu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023}
}

@inproceedings{phobert,
  title={{PhoBERT}: Pre-trained language models for {V}ietnamese},
  author={Nguyen, Dat Quoc and Nguyen, Anh Tuan},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2020},
  pages={1037--1042},
  year={2020}
}
```
