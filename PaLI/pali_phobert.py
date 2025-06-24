import torch
from transformers import AutoTokenizer, RobertaModel
import py_vncorenlp
import os

from pali import PaLI

BASE_MODEL_ID = "google/mt5-base"
PHOBERT_ID = "vinai/phobert-base-v2"

# Path to the VnCoreNLP.jar file. This must be downloaded separately.
# See: https://github.com/vncorenlp/VnCoreNLP
VNCORENLP_DIR = "/home/21khac.dd/bm/VnCoreNLP"
VNCORENLP_PATH = "/home/21khac.dd/bm/VnCoreNLP/VnCoreNLP-1.2.jar"

def create_pali_phobert_model(
        base_model_i: str = BASE_MODEL_ID,
        phobert_id: str = PHOBERT_ID,
        vncorenlp_dir: str = VNCORENLP_DIR,
        device = None
) -> tuple:
    """
    Performs the full model surgery to integrate PhoBERT components into a T5-based model.
    
    Returns:
        A tuple of (modified_model, phobert_tokenizer, rdrsegmenter).
    """
    # --- Load Components ---
    print(f"Loading base model: {base_model_i}")
    pali_model = PaLI(
        device=device
    ).to(device, non_blocking=True)

    print(f"Loading PhoBERT tokenizer: {phobert_id}")
    phobert_tokenizer = AutoTokenizer.from_pretrained(phobert_id)

    # Load a standalone PhoBERT model to act as the source for embedding weights
    print(f"Loading PhoBERT model for embedding weights: {phobert_id}")
    phobert_source_model = RobertaModel.from_pretrained(phobert_id).to(device, non_blocking=True)

    print("Initializing VnCoreNLP for word segmentation...")
    # This RDRSegmenter is required for pre-processing text for PhoBERT
    rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=vncorenlp_dir)

    print("\n--- Initial Setup Complete ---")
    print(f"Original PaLI (mT5) vocab size: {pali_model.mt5.config.vocab_size}")
    print(f"Target PhoBERT vocab size: {phobert_tokenizer.vocab_size}")
    print(f"PaLI (mT5) embedding dim: {pali_model.mt5.config.d_model}")
    print(f"PhoBERT embedding dim: {phobert_source_model.config.hidden_size}")
    
    # --- Verify Dimension Compatibility ---
    assert pali_model.mt5.config.d_model == phobert_source_model.config.hidden_size, \
    "Embedding dimensions do not match. Contingency strategy (projection layer) required."

    # --- Resize Embeddings ---
    # --- 3. Resize the Model's Token Embeddings ---
    new_vocab_size = len(phobert_tokenizer)
    print(f"\nResizing PaLI model's embeddings to new vocab size: {new_vocab_size}")

    # Get original embedding layers for comparison
    original_input_embeddings = pali_model.mt5.get_input_embeddings()
    original_output_embeddings = pali_model.mt5.get_output_embeddings()

    # This single call resizes both input and output embeddings if they are tied
    pali_model.mt5.resize_token_embeddings(new_vocab_size)

    print(f"Original input embedding shape: {original_input_embeddings.weight.shape}")
    print(f"New input embedding shape: {pali_model.mt5.get_input_embeddings()}")
    print(f"Original output embedding shape: {original_output_embeddings.weight.shape}")
    print(f"New output embedding shape: {pali_model.mt5.get_output_embeddings()}")

    # Verify the resize was successful
    assert pali_model.mt5.get_input_embeddings().weight.shape[0] == new_vocab_size ,"Input embedding mismatch"
    assert pali_model.mt5.get_output_embeddings().weight.shape[0] == new_vocab_size, "Output embedding mismatch"
    
    # --- 4. Transplant Embedding Weights ---
    print("\nTransplanting weights from PhoBERT to the resized PaLI model...")

    # Get the source and target embedding layers
    source_embeddings = phobert_source_model.get_input_embeddings()

    # Directly copy the pre-trained weights
    with torch.no_grad():
        pali_model.mt5.get_input_embeddings().weight.data = source_embeddings.weight.data.clone()

    # For an encoder-decoder model, we must also update the output LM head.
    # If the weights are tied, this is already done. If not, we must do it manually.
    # The `resize_token_embeddings` call preserves the tie.
    # We can check if they are the same object in memory.
    if pali_model.mt5.get_input_embeddings() is not pali_model.mt5.get_output_embeddings():
        print("Input and output embeddings are not tied. Transplanting output head weights separately.")
        with torch.no_grad():
            # In a real scenario, the source for output weights would need careful consideration.
            # For PhoBERT (encoder-only), there's no pre-trained decoder head.
            # A common practice is to initialize the output head with the input embedding weights.
            pali_model.mt5.get_output_embeddings().weight.data = source_embeddings.weight.data.clone()
    else:
        print("Input and output embeddings are tied. No separate transplantation needed for the output head.")

    print("Weight transplantation complete.")

    # --- Update Special Token IDs in Model Config ---
    print("\nUpdating model configuration with new special token IDs...")
    pali_model.mt5.config.bos_token_id = phobert_tokenizer.bos_token_id
    pali_model.mt5.config.eos_token_id = phobert_tokenizer.eos_token_id
    pali_model.mt5.config.pad_token_id = phobert_tokenizer.pad_token_id
    pali_model.mt5.config.decoder_start_token_id = phobert_tokenizer.bos_token_id # T5 uses this
    
    print("Model config updated.")
    
    return pali_model, phobert_tokenizer, rdrsegmenter