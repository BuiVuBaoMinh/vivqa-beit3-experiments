import json
import sys
import os

from typing import Optional

import torch
import torch.nn as nn
from transformers import (
    ChameleonConfig,
    ChameleonModel,
    ChameleonPreTrainedModel,
    ChameleonForConditionalGeneration,
    AutoProcessor,
    PhobertTokenizer,
    RobertaModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

CHAMELEON_MODEL_ID = "facebook/chameleon-7b"
PHOBERT_MODEL_ID = "vinai/phobert-base-v2"
MY_CACHE_DIR = "/home/21khac.dd/bm/my-cache-dir"
DEFAULT_ANSWER2LABEL = "/home/21khac.dd/bm/data/vivqa/annotations/dicts/answer2label_en_gemini_translated.txt"

class ChameleonForVQAClassification(ChameleonPreTrainedModel):
    """
    A custom Chameleon model for Visual Question Answering framed as a classification task.
    This model adds a classification head on top of the pretrained ChameleonModel.
    """
    def __init__(self, config: ChameleonConfig, num_labels: int, answer2label_path: str):
        super().__init__(config)

        self.answer2label, self.label2answer = self._load_answer_mappings(answer2label_path)
        self.num_labels = len(self.label2answer)
        print(f"ChameleonForVQAClassification: num_labels = {self.num_labels}")

        # Base Chameleon model
        self.model = ChameleonModel(config)
        
        # Classification head
        self.dropout = nn.Dropout(config.hidden_size * 0.1) # Use dropout value from config or a default
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Forward pass for VQA classification.
        
        Args:
            input_ids (torch.LongTensor): Tokenized input text.
            pixel_values (torch.FloatTensor): Processed image pixels.
            attention_mask (Optional[torch.Tensor]): Mask to avoid performing attention on padding token indices.
            labels (Optional[torch.LongTensor]): Ground truth labels for classification.
        
        Returns:
            SequenceClassifierOutput: An object containing the loss and logits.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 1. Pass inputs to the base Chameleon model
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            return_dict=return_dict,
        )

        # 2. Get the last hidden state from the language model's output.
        # Shape: (batch_size, sequence_length, hidden_size)
        hidden_states = outputs.last_hidden_state

        # 3. Use the hidden state of the LAST token as the pooled representation.
        # This token's state is conditioned on both the image and the full question.
        # Shape: (batch_size, hidden_size)
        pooled_output = hidden_states[:, -1, :]

        # 4. Apply dropout and the classification head
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def _load_answer_mappings(self, file_path):
        """Loads the answer to label mappings from the provided file."""
        answer_to_label = {}
        label_to_answer = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                answer = item['answer'] # Assuming segment_normalize is not needed here or done elsewhere
                label = item['label']
                answer_to_label[answer] = label
                label_to_answer[label] = answer
        
        # Add UNKNOWN label if not present
        if "UNKNOWN" not in answer_to_label:
            new_label = len(answer_to_label)
            answer_to_label["UNKNOWN"] = new_label
            label_to_answer[new_label] = "UNKNOWN"
            print("Appended a default UNKNOWN label.")

        print(f"ChameleonForVQAClassification: Loaded {len(answer_to_label)} answer-to-label mappings from {file_path}.")
        return answer_to_label, label_to_answer

    def get_num_layers(self):
        """Returns the number of layers in the vision and language models."""
        num_text_layers = self.model.language_model.config.num_hidden_layers
        num_vision_layers = self.model.vision_tower.config.num_hidden_layers
        print(f"Number of text layers: {num_text_layers}")
        print(f"Number of vision layers: {num_vision_layers}")
        return num_text_layers + num_vision_layers
    
    @torch.jit.ignore
    def no_weight_decay(self):
        """Specifies parameters that should not be subject to weight decay."""
        no_decay = {'bias', 'LayerNorm.weight'}
        return no_decay

def get_vivqa_chameleon(
    chameleon_model_id: str = CHAMELEON_MODEL_ID,
    chameleon_config: ChameleonConfig = None,
    num_labels: int = 336,
    answer2label_path: str = DEFAULT_ANSWER2LABEL,
    device: str = "cuda"
) -> ChameleonForVQAClassification:
    """
    Instantiates the custom ChameleonForVQAClassification model and loads pretrained weights.
    """
    if chameleon_config is None:
        chameleon_config = ChameleonConfig.from_pretrained(chameleon_model_id, cache_dir=MY_CACHE_DIR)

    chameleon_config.use_cache = False

    # 1. Instantiate our randomly initialized custom model
    print(f"Instantiating custom ChameleonForVQAClassification model with {num_labels} labels...")
    model = ChameleonForVQAClassification(config=chameleon_config, num_labels=num_labels, answer2label_path=answer2label_path)

    # 2. Load the full pre-trained generative model into a temporary variable
    print("Loading full pre-trained ChameleonForConditionalGeneration model temporarily...")
    temp_vqa_model = ChameleonForConditionalGeneration.from_pretrained(
        chameleon_model_id,
        cache_dir=MY_CACHE_DIR,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    ).to(device)

    # 3. Manually copy the weights from the temporary model to our custom model's base
    print("Manually copying pre-trained weights...")
    model.model.load_state_dict(temp_vqa_model.model.state_dict())

    print("✅ Pre-trained weights for vision and text models loaded successfully.")

    # Clean up the temporary model to save memory
    del temp_vqa_model
    torch.cuda.empty_cache()

    return model

def get_vivqa_chameleon_phobert(
    chameleon_model_id: str = CHAMELEON_MODEL_ID,
    phobert_id: str = PHOBERT_MODEL_ID,
    num_labels: int = 336,
    answer2label_path: str = DEFAULT_ANSWER2LABEL,
    device: str = "cuda"
):
    """
    Creates a Chameleon VQA model with its text embeddings replaced by PhoBERT's.
    """
    # --- LOAD MODELS AND PROCESSOR ---
    print("Loading PhoBERT model and tokenizer...")
    phobert_model = RobertaModel.from_pretrained(phobert_id, cache_dir=MY_CACHE_DIR)
    phobert_tokenizer = PhobertTokenizer.from_pretrained(phobert_id, cache_dir=MY_CACHE_DIR)

    print("Loading Chameleon processor to get special tokens...")
    temp_processor = AutoProcessor.from_pretrained(chameleon_model_id, cache_dir=MY_CACHE_DIR)
    
    # Add Chameleon's special tokens to PhoBERT's tokenizer
    special_tokens_to_add = [
        temp_processor.tokenizer.eos_token, # End of text
        "<image>", # Image placeholder
    ]
    phobert_tokenizer.add_special_tokens({'additional_special_tokens': list(set(special_tokens_to_add))})
    print(f"Added special tokens to PhobertTokenizer.")

    # --- GET BASE CHAMELEON MODEL ---
    # We first get a standard Chameleon VQA model
    chameleon_model = get_vivqa_chameleon(
        chameleon_model_id=chameleon_model_id,
        num_labels=num_labels,
        answer2label_path=answer2label_path,
        device=device
    )

    # --- PERFORM WEIGHT TRANSPLANT ---
    print("\nPerforming weight transplant for the Chameleon language model embeddings...")
    
    # 1. Resize the language model's token embeddings to match the new tokenizer size
    new_vocab_size = len(phobert_tokenizer)
    chameleon_model.model.language_model.resize_token_embeddings(new_vocab_size)
    print(f"Resized language model embeddings to: {chameleon_model.model.language_model.get_input_embeddings().weight.data.shape}")

    # 2. Copy the weights from PhoBERT's embeddings
    with torch.no_grad():
        source_embeddings = phobert_model.embeddings.word_embeddings
        target_embeddings = chameleon_model.model.language_model.embed_tokens

        # Copy weights for all tokens in PhoBERT's vocabulary
        num_tokens_to_copy = source_embeddings.weight.shape[0]
        target_embeddings.weight.data[:num_tokens_to_copy, :] = source_embeddings.weight.data.clone().to(target_embeddings.weight.dtype)
        print(f"Copied weights for {num_tokens_to_copy} tokens from PhoBERT into Chameleon.")

    # --- UPDATE MODEL CONFIGURATION ---
    print("\nUpdating model configuration with new special token IDs...")
    chameleon_model.config.text_config.bos_token_id = phobert_tokenizer.bos_token_id
    chameleon_model.config.text_config.eos_token_id = phobert_tokenizer.eos_token_id
    chameleon_model.config.text_config.pad_token_id = phobert_tokenizer.pad_token_id
    chameleon_model.config.pad_token_id = phobert_tokenizer.pad_token_id
    chameleon_model.model.language_model.config.pad_token_id = phobert_tokenizer.pad_token_id
    
    print("✅ PhoBERT embedding transplant successful and verified!")

    return chameleon_model
