import json
import sys
import os

import torch
import torch.nn as nn
from transformers import (
    Idefics3Processor, Idefics3ForConditionalGeneration, Idefics3Model, Idefics3Config,
    LlamaModel,
    SmolVLMConfig, SmolVLMForConditionalGeneration, SmolVLMModel, SmolVLMPreTrainedModel,
    PhobertTokenizer, RobertaModel
)
from transformers.modeling_outputs import SequenceClassifierOutput

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from glossary import segment_normalize

SMOLVLM_500M_MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
PHOBERT_MODEL_ID = "vinai/phobert-base-v2"
MY_CACHE_DIR = "/home/21khac.dd/bm/my-cache-dir"
DEFAUT_ANSWER2LABEL = "/home/21khac.dd/bm/data/vivqa/annotations/dicts/answer2label_en_gemini_translated.txt"

class SmolVLMForVQAClassification(SmolVLMPreTrainedModel):
    def __init__(self, config, num_labels, answer2label_path):
        super().__init__(config)

        self.answer2label, self.label2answer = self._load_answer_mappings(answer2label_path)
        self.num_labels = len(self.label2answer)
        print(f"SmolVLMForVQAClassification: num_labels = {self.num_labels}")

        self.smolvlm = Idefics3Model(config)
        
        # Add a dropout and a classification head
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(config.text_config.hidden_size, num_labels)

    def forward(
        self,
        input_ids,
        pixel_values,
        attention_mask=None,
        pixel_attention_mask=None,
        labels=None, # Use for our classification
        return_dict=True,
    ):

        # 1. Pass all inputs to the base SmolVLM.
        outputs = self.smolvlm(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            pixel_attention_mask=pixel_attention_mask,
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

        print(f"SmolVLMForVQAClassification: Loaded {len(answer_to_label)} answer-to-label mappings from {file_path}.")
        return answer_to_label, label_to_answer


    def get_num_layers(self):
        """
        Returns the total number of transformer layers in both the
        vision and language model backbones.
        """
        # The language model in Idefics3 is a LlamaModel
        num_text_layers = len(self.smolvlm.text_model.layers)
        
        # The vision model in Idefics3 is a CLIP-style model
        num_vision_layers = len(self.smolvlm.vision_model.encoder.layers)
        
        print(f"Number of text layers: {num_text_layers}")
        print(f"Number of vision layers: {num_vision_layers}")
        
        return num_text_layers + num_vision_layers
    
    @torch.jit.ignore
    def no_weight_decay(self):
        """
        Specifies parameters that should not be subject to weight decay.
        This version also excludes all biases and normalization layers.
        """
        # Start with the predefined set
        no_decay = {'position_embedding', 'class_embedding', 'logit_scale'}

        # Add all bias and normalization parameters
        for name, param in self.named_parameters():
            if 'bias' in name or 'norm' in name.lower():
                no_decay.add(name)

        return no_decay

def get_vivqa_smolvlm(
        smolvlm_model_id = SMOLVLM_500M_MODEL_ID,
        smolvlm_config: Idefics3Config = None,
        num_labels = 336, # 335 labels plus an UNKNOWN
        answer2label_path = DEFAUT_ANSWER2LABEL,
        device = "cuda"
) -> SmolVLMForVQAClassification:
    
    if smolvlm_config is None:
        smolvlm_config = Idefics3Config.from_pretrained(smolvlm_model_id)

    # --- INSTANTIATE AND MANUALLY LOAD WEIGHTS ---

    # 1. Instantiate our randomly initialized custom model
    print(f"Instantiating custom SmolVLMForVQAClassification model with {num_labels} labels...")
    model = SmolVLMForVQAClassification(config=smolvlm_config, num_labels=num_labels, answer2label_path=answer2label_path)

    # 2. Load the full pre-trained VQA model into a temporary variable
    print("Loading full pre-trained Idefics3 model temporarily...")
    temp_vqa_model = Idefics3ForConditionalGeneration.from_pretrained(
        smolvlm_model_id, cache_dir=MY_CACHE_DIR, local_files_only=True
    ).to(device)

    # 3. Manually copy the weights from the temporary model to our custom model
    print("Manually copying pre-trained weights...")
    model.smolvlm.vision_model.load_state_dict(temp_vqa_model.model.vision_model.state_dict())
    model.smolvlm.connector.load_state_dict(temp_vqa_model.model.connector.state_dict())
    model.smolvlm.text_model.load_state_dict(temp_vqa_model.model.text_model.state_dict(), strict=False)

    print("✅ Pre-trained weights for vision and text models loaded successfully.")

    print(f"idefics3.padding_idx: {model.smolvlm.padding_idx}, idefics3.text_model.padding_idx: {model.smolvlm.text_model.padding_idx}")

    # Clean up the temporary model to save memory
    del temp_vqa_model
    torch.cuda.empty_cache()

    return model

def get_vivqa_smolvlm_phobert(
    smolvlm_model_id = SMOLVLM_500M_MODEL_ID,
    phobert_id = PHOBERT_MODEL_ID,
    num_labels = 336, # 335 labels plus an UNKNOWN
    answer2label_path = DEFAUT_ANSWER2LABEL,
    device = "cuda"
):

    # --- LOAD MODELS AND TOKENIZER ---
    print("Loading PhoBERT and Paligemma configurations...")
    phobert_model = RobertaModel.from_pretrained(phobert_id)
    phobert_tokenizer = PhobertTokenizer.from_pretrained(phobert_id)

    temp_processor = Idefics3Processor.from_pretrained(smolvlm_model_id, use_fast=True)
    tokens_to_add = {
        "additional_special_tokens": [
            temp_processor.fake_image_token,
            temp_processor.image_token,
            temp_processor.end_of_utterance_token,
        ]
    }

    phobert_tokenizer.add_special_tokens(tokens_to_add)
    image_token_id = phobert_tokenizer.convert_tokens_to_ids(temp_processor.image_token)
    fake_image_token_id = phobert_tokenizer.convert_tokens_to_ids(temp_processor.fake_image_token)
    end_of_utterance_token = phobert_tokenizer.convert_tokens_to_ids(temp_processor.end_of_utterance_token)
    print(f"Added {temp_processor.image_token} token to PhobertTokenizer with new ID: {image_token_id}")
    print(f"Added {temp_processor.fake_image_token} token to PhobertTokenizer with new ID: {fake_image_token_id}")
    print(f"Added {temp_processor.end_of_utterance_token} token to PhobertTokenizer with new ID: {end_of_utterance_token}")

    custom_smolvlm_config = Idefics3Config.from_pretrained(SMOLVLM_500M_MODEL_ID)
    custom_smolvlm_config.pad_token_id = phobert_model.config.pad_token_id
    custom_smolvlm_config.text_config.pad_token_id = phobert_model.config.pad_token_id
    custom_smolvlm_config.image_token_id = image_token_id
    
    smolvlm_model = get_vivqa_smolvlm(
        smolvlm_model_id=smolvlm_model_id, 
        smolvlm_config= custom_smolvlm_config,
        num_labels=num_labels, 
        answer2label_path=answer2label_path, 
        device=device
    )
    
    # --- PERFORM WEIGHT TRANSPLANT ---
    print("\nPerforming weight transplant for the SmolVLM text model...")

    # 1. Resize the text model's embeddings
    new_vocab_size = len(phobert_tokenizer)

    smolvlm_model.smolvlm.resize_token_embeddings(new_vocab_size)
    print(f"Resized text model embeddings to: {smolvlm_model.smolvlm.get_input_embeddings().weight.data.shape}")

    # 2. Copy the weights from PhoBERT
    # --- PERFORM PARTIAL WEIGHT TRANSPLANT ---
    print("\nPerforming partial weight transplant from PhoBERT to SmolVLM...")
    with torch.no_grad():
        source_embeddings = phobert_model.embeddings.word_embeddings
        target_embeddings = smolvlm_model.smolvlm.text_model.embed_tokens

        source_vocab_size, source_dim = source_embeddings.weight.shape
        target_vocab_size, target_dim = target_embeddings.weight.shape
        
        # Determine the number of tokens to copy (the original PhoBERT vocab size)
        num_tokens_to_copy = source_vocab_size
        
        # Copy the 768-dim PhoBERT weights into the first 768 dims of the 2304-dim PaliGemma weights
        smolvlm_model.smolvlm.text_model.embed_tokens.weight.data[0:num_tokens_to_copy, 0:source_dim] = \
            phobert_model.embeddings.word_embeddings.weight.data[0:num_tokens_to_copy, :].clone()
        print(f"Copied weights for {num_tokens_to_copy} tokens from PhoBERT (dim {source_dim}) into SmolVLM (dim {target_dim}).")

    # --- Update Special Token IDs in Model Config ---
    print("\nUpdating model configuration with new special token IDs...")
    smolvlm_model.smolvlm.text_model.config.bos_token_id = phobert_tokenizer.bos_token_id
    smolvlm_model.smolvlm.text_model.config.eos_token_id = phobert_tokenizer.eos_token_id
    smolvlm_model.smolvlm.text_model.config.pad_token_id = phobert_tokenizer.pad_token_id

    smolvlm_model.smolvlm.text_model.padding_idx = phobert_tokenizer.pad_token_id
    # TODO: delete this print line after confirmation
    print(f"smolvlm_model.smolvlm.text_model.padding_idx: {smolvlm_model.smolvlm.text_model.padding_idx}")
    print(f"smolvlm_model.smolvlm.text_model.config.pad_token_id: {smolvlm_model.smolvlm.text_model.config.pad_token_id}")

    print(f"phobert_tokenizer.pad_token_id: {phobert_tokenizer.pad_token_id}, phobert's pad_token_id: {phobert_model.config.pad_token_id}")

    # --- VERIFICATION ---
    print("\nVerifying transplant...")
    with torch.no_grad():
        # Get the target slice from SmolVLM that was just modified
        target_slice = smolvlm_model.smolvlm.text_model.embed_tokens.weight[0:num_tokens_to_copy, 0:source_dim]
        
        # Get the source slice from the original PhoBERT model
        source_slice = phobert_model.embeddings.word_embeddings.weight[0:num_tokens_to_copy, :]

        # In the assertion, we cast the source_slice to the same dtype as the target_slice
        # This compares bfloat16 to bfloat16, which will pass.
        assert torch.equal(target_slice, source_slice.to(target_slice.dtype))

    assert smolvlm_model.smolvlm.padding_idx == phobert_model.config.pad_token_id, \
        f"pad_token_id mismatch! smolvlm: {smolvlm_model.smolvlm.padding_idx}, phobert: {phobert_model.config.pad_token_id}"
    assert smolvlm_model.smolvlm.text_model.padding_idx == phobert_model.config.pad_token_id, \
        f"pad_token_id mismatch! smolvlm: {smolvlm_model.smolvlm.text_model.padding_idx}, phobert: {phobert_model.config.pad_token_id}"
    print("✅ PhoBERT embedding transplant successful and verified!")

    return smolvlm_model

