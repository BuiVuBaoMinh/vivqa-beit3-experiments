import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer, RobertaModel
import py_vncorenlp
import os

BASE_MODEL_ID = "google/mt5-base"
PHOBERT_ID = "vinai/phobert-base"

# Path to the VnCoreNLP.jar file. This must be downloaded separately.
# See: https://github.com/vncorenlp/VnCoreNLP
VNCORENLP_PATH = "/home/21khac.dd/bm/VnCoreNLP"
# Ensure the path is correct and the JAR file exists.
if not os.path.exists(VNCORENLP_PATH):
    raise FileNotFoundError(
        f"VnCoreNLP JAR not found at {VNCORENLP_PATH}. "
        "Please download it and update the path."
    )

py_vncorenlp.download_model(save_dir=VNCORENLP_PATH)

# --- 1. Load Models and Tokenizers ---

# Load the base model to be modified (our "PaLI" stand-in)
print(f"Loading base model: {BASE_MODEL_ID}")
pali_model = T5ForConditionalGeneration.from_pretrained(BASE_MODEL_ID)

# Load the PhoBERT tokenizer
print(f"Loading PhoBERT tokenizer: {PHOBERT_ID}")
phobert_tokenizer = AutoTokenizer.from_pretrained(PHOBERT_ID)

# Load a standalone PhoBERT model to act as the source for embedding weights
print(f"Loading PhoBERT model for embedding weights: {PHOBERT_ID}")
phobert_source_model = RobertaModel.from_pretrained(PHOBERT_ID)

# --- 2. Initialize the Mandatory Word Segmenter ---
print("Initializing VnCoreNLP for word segmentation...")
# This RDRSegmenter is required for pre-processing text for PhoBERT
rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir='/home/21khac.dd/bm/VnCoreNLP')

print("\n--- Initial Setup Complete ---")
print(f"Original PaLI (mT5) vocab size: {pali_model.config.vocab_size}")
print(f"Target PhoBERT vocab size: {phobert_tokenizer.vocab_size}")
print(f"PaLI (mT5) embedding dim: {pali_model.config.d_model}")
print(f"PhoBERT embedding dim: {phobert_source_model.config.hidden_size}")

# Verify that dimensions match for our primary strategy
assert pali_model.config.d_model == phobert_source_model.config.hidden_size, \
    "Embedding dimensions do not match. Contingency strategy (projection layer) required."