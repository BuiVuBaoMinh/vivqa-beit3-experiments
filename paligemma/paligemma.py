import json
import sys
import os

import torch
import torch.nn as nn
from transformers import (
    PaliGemmaModel, PaliGemmaPreTrainedModel, PaliGemmaConfig, PaliGemmaForConditionalGeneration,
    Gemma2Model, SiglipVisionModel,
    PhobertTokenizer, RobertaModel
)
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.tokenization_utils_base import AddedToken
from transformers.models.paligemma.processing_paligemma import IMAGE_TOKEN, EXTRA_TOKENS

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from glossary import segment_normalize

PALIGEMMA_MODEL_ID = "google/paligemma2-3b-pt-224"
PHOBERT_MODEL_ID = "vinai/phobert-base-v2"
MY_CACHE_DIR = "/home/21khac.dd/bm/my-cache-dir"

class PaligemmaForVQAClassification(PaliGemmaPreTrainedModel):
    def __init__(self, config, num_labels, answer2label_path):
        super().__init__(config)

        self.answer2label, self.label2answer = self._load_answer_mappings(answer2label_path)
        self.num_labels = len(self.label2answer)
        print(f"PaligemmaForVQAClassification: num_labels = {self.num_labels}")
        
        # Use the base BLIP model which contains the vision and text encoders
        self.paligemma = PaliGemmaModel(config)
        
        # Add a dropout and a classification head
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(config.text_config.hidden_size, num_labels)

    def forward(
        self,
        input_ids,
        pixel_values,
        attention_mask=None,
        token_type_ids=None,
        paligemma_labels=None, # Pass to Paligemma
        labels=None, # Use for our classification
        return_dict=True,
    ):

        # 1. Pass all inputs to the base PaliGemmaModel.
        # It will handle the image feature extraction, projection, and fusion
        # before passing the combined sequence to the Gemma2 language model.
        outputs = self.paligemma(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids, # Important for masking during training
            labels=paligemma_labels, # Important for causal masking during training
            return_dict=True,
        )

        # 2. Get the last hidden state from the language model's output.
        hidden_states = outputs.last_hidden_state

        # 3. Use the hidden state of the LAST token as the pooled representation.
        # This token's state is conditioned on the entire image and question.
        # Shape: (batch_size, hidden_size)
        pooled_output = hidden_states[:, -1, :]

        # 4. Apply dropout and the classification head.
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            # Use standard CrossEntropyLoss for multi-class classification
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + (outputs.hidden_states, outputs.attentions)
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def _load_answer_mappings(self, file_path):
        """
        Loads the answer to label mappings from the provided JSONL file.
        Returns:
            - A dictionary mapping answer strings to integer labels.
            - A dictionary mapping integer labels to answer strings.
        """
        answer_to_label = {} # there are same answers with different labels in translated en!
        label_to_answer = {}
        seen_answers = set()
        unique_answers = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                answer = segment_normalize(item['answer'])
                label_to_answer[item['label']] = answer

                if answer in seen_answers:
                    continue  # Skip duplicates
                seen_answers.add(answer)
                unique_answers.append(answer)

                answer_to_label[answer] = item['label']

        # assert len(answer_to_label) == len(label_to_answer), \
        #     f"answer_to_label and label_to_answer dicts mismatch {len(answer_to_label)} vs {len(label_to_answer)}!"

        if len(answer_to_label) != len(label_to_answer):
            print(f"Warning: answer_to_label and label_to_answer dicts mismatch {len(answer_to_label)} vs {len(label_to_answer)}!")
            print("This could be translation or data source error.")
            print("Modifying label_to_answer to match answer_to_label..")
            label_to_answer = {}
            label_to_answer = {v: k for k, v in answer_to_label.items()}

        # Duplication removal could results in labels out of bound
        answer_to_label = {ans: idx for idx, ans in enumerate(unique_answers)}
        label_to_answer = {idx: ans for ans, idx in answer_to_label.items()}
        print(f"Reindexed {len(answer_to_label)} unique answers to labels 0 through {len(answer_to_label) - 1}.")

        answer_to_label["UNKNOWN"] = len(answer_to_label)
        label_to_answer[len(label_to_answer)] = "UNKNOWN"
        print("Append a default UNKNOWN label.")

        print(f"PaligemmaForVQAClassification: Loaded {len(answer_to_label)} answer-to-label mappings from {file_path}.")
        return answer_to_label, label_to_answer


    def get_num_layers(self):
        """
        Returns the total number of transformer layers in both the
        vision and language model backbones.
        """
        num_text_layers = len(self.paligemma.language_model.layers)
        # Vision tower layers are typically in an 'encoder.layer' attribute
        num_vision_layers = len(self.paligemma.vision_tower.vision_model.encoder.layers)
        print(type(self.paligemma.vision_tower))
        print(f"Num layers: {num_text_layers + num_vision_layers}")
        return num_text_layers + num_vision_layers
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'position_embedding', 'class_embedding', 'logit_scale'}

def get_vivqa_paligemma(
        paligemma_model_id = PALIGEMMA_MODEL_ID,
        paligemma_config: PaliGemmaConfig = None,
        num_labels = 336, # 335 labels plus an UNKNOWN
        answer2label_path = None,
        device = "cuda"
) -> PaligemmaForVQAClassification:
    
    if paligemma_config is None:
        paligemma_config = PaliGemmaConfig.from_pretrained(paligemma_model_id)

    # --- INSTANTIATE AND MANUALLY LOAD WEIGHTS ---

    # 1. Instantiate our randomly initialized custom model
    print(f"Instantiating custom PaligemmaForVQAClassification model with {num_labels} labels...")
    model = PaligemmaForVQAClassification(config=paligemma_config, num_labels=num_labels, answer2label_path=answer2label_path)

    # 2. Load the full pre-trained VQA model into a temporary variable
    print("Loading full pre-trained PaliGemmaModel model temporarily...")
    temp_vqa_model = PaliGemmaModel.from_pretrained(paligemma_model_id, cache_dir=MY_CACHE_DIR, local_files_only=True).to(device)

    # 3. Manually copy the weights from the temporary model to our custom model
    print("Manually copying pre-trained weights...")
    model.paligemma.vision_tower.load_state_dict(temp_vqa_model.vision_tower.state_dict())
    model.paligemma.language_model.load_state_dict(temp_vqa_model.language_model.state_dict(), strict=False)
    print("✅ Pre-trained weights for vision and text models loaded successfully.")

    print(f"paligemma.pad_token_id: {model.paligemma.pad_token_id}, paligemma.language_model.padding_idx: {model.paligemma.language_model.padding_idx}")

    # Clean up the temporary model to save memory
    del temp_vqa_model
    torch.cuda.empty_cache()

    return model

def get_vivqa_paligemma_phobert(
    paligemma_model_id = PALIGEMMA_MODEL_ID,
    phobert_id = PHOBERT_MODEL_ID,
    num_labels = 336, # 335 labels plus an UNKNOWN
    answer2label_path = None,
    device = "cuda"
):

    # --- LOAD MODELS AND TOKENIZER ---
    print("Loading PhoBERT and Paligemma configurations...")
    phobert_model = RobertaModel.from_pretrained(phobert_id)
    phobert_tokenizer = PhobertTokenizer.from_pretrained(phobert_id)

    if not hasattr(phobert_tokenizer, "image_token"):
        image_token = AddedToken(IMAGE_TOKEN, normalized=False, special=True)
        tokens_to_add = {"additional_special_tokens": [image_token]}
        phobert_tokenizer.add_special_tokens(tokens_to_add)
        image_token_id = phobert_tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        image_token = IMAGE_TOKEN

        phobert_tokenizer.add_tokens(EXTRA_TOKENS)
        phobert_tokenizer.add_bos_token = False
        phobert_tokenizer.add_eos_token = False

        print(f"Added '<image>' token to PhobertTokenizer with new ID: {image_token_id}")

    custom_paligemma_config = PaliGemmaConfig.from_pretrained(paligemma_model_id)
    custom_paligemma_config.pad_token_id = phobert_model.config.pad_token_id
    custom_paligemma_config.text_config.pad_token_id = phobert_model.config.pad_token_id
    custom_paligemma_config.image_token_id = image_token_id
    
    paligemma_model = get_vivqa_paligemma(
        paligemma_model_id=paligemma_model_id, 
        paligemma_config = custom_paligemma_config,
        num_labels=num_labels, 
        answer2label_path=answer2label_path, 
        device=device
    )
    
    # --- PERFORM WEIGHT TRANSPLANT ---
    print("\nPerforming weight transplant for the Paligemma language model...")

    # 1. Resize the text model's embeddings
    new_vocab_size = len(phobert_tokenizer)
    # paligemma_model.paligemma.language_model.resize_token_embeddings(new_vocab_size)
    # print(f"Resized text model embeddings to: {paligemma_model.paligemma.get_input_embeddings().weight.data.shape}")

    paligemma_model.paligemma.resize_token_embeddings(new_vocab_size)
    print(f"Resized text model embeddings to: {paligemma_model.paligemma.get_input_embeddings().weight.data.shape}")
    # paligemma_model.paligemma.config.vocab_size = new_vocab_size

    # 2. Copy the weights from PhoBERT
    # --- PERFORM PARTIAL WEIGHT TRANSPLANT ---
    print("\nPerforming partial weight transplant from PhoBERT to PaliGemma...")
    with torch.no_grad():
        source_embeddings = phobert_model.embeddings.word_embeddings
        target_embeddings = paligemma_model.paligemma.language_model.embed_tokens

        source_vocab_size, source_dim = source_embeddings.weight.shape
        target_vocab_size, target_dim = target_embeddings.weight.shape
        
        # Determine the number of tokens to copy (the original PhoBERT vocab size)
        num_tokens_to_copy = source_vocab_size
        
        # Copy the 768-dim PhoBERT weights into the first 768 dims of the 2304-dim PaliGemma weights
        paligemma_model.paligemma.language_model.embed_tokens.weight.data[0:num_tokens_to_copy, 0:source_dim] = \
            phobert_model.embeddings.word_embeddings.weight.data[0:num_tokens_to_copy, :].clone()
        print(f"Copied weights for {num_tokens_to_copy} tokens from PhoBERT (dim {source_dim}) into PaliGemma (dim {target_dim}).")

    # --- Update Special Token IDs in Model Config ---
    print("\nUpdating model configuration with new special token IDs...")
    paligemma_model.paligemma.language_model.config.bos_token_id = phobert_tokenizer.bos_token_id
    paligemma_model.paligemma.language_model.config.eos_token_id = phobert_tokenizer.eos_token_id
    paligemma_model.paligemma.language_model.config.pad_token_id = phobert_tokenizer.pad_token_id

    print(f"phobert_tokenizer.pad_token_id: {phobert_tokenizer.pad_token_id}, phobert's pad_token_id: {phobert_model.config.pad_token_id}")

    # --- VERIFICATION ---
    print("\nVerifying transplant...")
    with torch.no_grad():
        # Get the target slice from PaliGemma that was just modified
        target_slice = paligemma_model.paligemma.language_model.embed_tokens.weight[0:num_tokens_to_copy, 0:source_dim]
        
        # Get the source slice from the original PhoBERT model
        source_slice = phobert_model.embeddings.word_embeddings.weight[0:num_tokens_to_copy, :]

        # In the assertion, we cast the source_slice to the same dtype as the target_slice
        # This compares bfloat16 to bfloat16, which will pass.
        assert torch.equal(target_slice, source_slice.to(target_slice.dtype))

    assert paligemma_model.paligemma.pad_token_id == phobert_model.config.pad_token_id, \
        f"pad_token_id mismatch! paligemma: {paligemma_model.paligemma.pad_token_id}, phobert: {phobert_model.config.pad_token_id}"
    assert paligemma_model.paligemma.language_model.padding_idx == phobert_model.config.pad_token_id, \
        f"pad_token_id mismatch! paligemma: {paligemma_model.paligemma.language_model.padding_idx}, phobert: {phobert_model.config.pad_token_id}"
    print("✅ PhoBERT embedding transplant successful and verified!")

    return paligemma_model
