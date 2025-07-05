from transformers import AutoTokenizer, BlipConfig, RobertaModel, BlipModel, BlipForQuestionAnswering
from myBLIP import BlipForVQAClassification
import torch

# --- CONFIGURATION ---
ANSWER_SET = ["yes", "no", "cat", "dog", "red", "blue", "green"] # Example
NUM_LABELS = len(ANSWER_SET)
PHOBERT_ID = "vinai/phobert-base-v2"
BLIP_ID = "Salesforce/blip-vqa-base"

# --- LOAD MODELS AND TOKENIZER ---
print("Loading PhoBERT and BLIP configurations...")
phobert_model = RobertaModel.from_pretrained(PHOBERT_ID)
phobert_tokenizer = AutoTokenizer.from_pretrained(PHOBERT_ID)
print(type(phobert_tokenizer))
blip_config = BlipConfig.from_pretrained(BLIP_ID)

# --- INSTANTIATE AND MANUALLY LOAD WEIGHTS (THE ROBUST WAY) ---

# 1. Instantiate our randomly initialized custom model
print(f"Instantiating custom BlipForVQAClassification model with {NUM_LABELS} labels...")
model = BlipForVQAClassification(config=blip_config, num_labels=NUM_LABELS)

# 2. Load the full pre-trained VQA model into a temporary variable
print("Loading full pre-trained BlipForQuestionAnswering model temporarily...")
temp_vqa_model = BlipForQuestionAnswering.from_pretrained(BLIP_ID)

# 3. Manually copy the weights from the temporary model to our custom model
print("Manually copying pre-trained weights...")
model.blip.vision_model.load_state_dict(temp_vqa_model.vision_model.state_dict())
# Note the source is temp_vqa_model.text_encoder and the destination is model.blip.text_model
model.blip.text_model.load_state_dict(temp_vqa_model.text_encoder.state_dict(), strict=False)
print("✅ Pre-trained weights for vision and text models loaded successfully.")

# Clean up the temporary model to save memory
del temp_vqa_model

# --- PERFORM WEIGHT TRANSPLANT ---
print("\nPerforming weight transplant for the text model...")

# 1. Resize the text model's embeddings
new_vocab_size = len(phobert_tokenizer)
model.blip.text_model.resize_token_embeddings(new_vocab_size)
print(f"Resized text model embeddings to: {new_vocab_size}")

# 2. Copy the weights from PhoBERT
with torch.no_grad():
    model.blip.text_model.embeddings.word_embeddings.weight.data = \
        phobert_model.embeddings.word_embeddings.weight.data.clone()

# --- VERIFICATION ---
assert torch.equal(model.blip.text_model.embeddings.word_embeddings.weight, phobert_model.embeddings.word_embeddings.weight)
print("✅ PhoBERT embedding transplant successful and verified!")

# Create Dataset
from blip_dataset import create_blip_dataset_by_split

dataset_train, data_loader_train, dataset_val, data_loader_val = create_pali_datasets(
    args,
    phobert_tokenizer=phobert_tokenizer
)