import json
import sys
import os
from copy import deepcopy

import torch
import torch.nn as nn
from transformers import (
    Idefics3Processor,
    VisualBertForQuestionAnswering, LxmertForQuestionAnswering,
    LlamaModel,
    SmolVLMConfig, SmolVLMForConditionalGeneration, SmolVLMModel, SmolVLMPreTrainedModel, SmolVLMProcessor, AutoProcessor,
    PhobertTokenizer, RobertaModel, RobertaConfig
)
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import Callable, List, Optional, Tuple, Union
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack
from transformers.models.smolvlm.modeling_smolvlm import SmolVLMBaseModelOutputWithPast, logger
from transformers.cache_utils import DynamicCache

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from glossary import segment_normalize

SMOLVLM_256M_MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
SMOLVLM_500M_MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
SMOLVLM_2B_MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
SMOLVLM_ID = SMOLVLM_2B_MODEL_ID # Use this
PHOBERT_MODEL_ID = "vinai/phobert-base-v2"
MY_CACHE_DIR = "/home/21khac.dd/bm/my-cache-dir"
DEFAUT_ANSWER2LABEL = "/home/21khac.dd/bm/data/vivqa/annotations/dicts/answer2label_en_gemini_translated.txt"

class SmolVLMForVQAClassification(SmolVLMPreTrainedModel):
    def __init__(self, config, answer2label_path):
        super().__init__(config)

        self.answer2label, self.label2answer = self._load_answer_mappings(answer2label_path)
        self.num_labels = len(self.label2answer)
        print(f"SmolVLMForVQAClassification: num_labels = {self.num_labels}")

        self.smolvlm = MySmolVLMModel(config)
        
        # Add a dropout and a classification head
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(config.text_config.hidden_size, self.num_labels)

        self.use_phobert_adapter = False

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
        if self.use_phobert_adapter:
            outputs = self.smolvlm(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                pixel_attention_mask=pixel_attention_mask,
                return_dict=True,
                use_phobert_adapter = self.use_phobert_adapter,
                phobert_embedding_adapter = self.phobert_embedding_adapter
            )
        else:
            outputs = self.smolvlm(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                pixel_attention_mask=pixel_attention_mask,
                return_dict=True,
                use_phobert_adapter = self.use_phobert_adapter,
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
        no_decay = {'position_embeddings', 'class_embedding', 'logit_scale'}

        # Add all bias and normalization parameters
        for name, param in self.named_parameters():
            if 'bias' in name or 'norm' in name.lower():
                no_decay.add(name)

        return no_decay

def get_vivqa_smolvlm(
        smolvlm_model_id = SMOLVLM_ID,
        smolvlm_config: SmolVLMConfig = None,
        answer2label_path = DEFAUT_ANSWER2LABEL,
        device = "cuda"
) -> SmolVLMForVQAClassification:
    
    if smolvlm_config is None:
        smolvlm_config = SmolVLMConfig.from_pretrained(smolvlm_model_id)

    # smolvlm_config._attn_implementation = "flash_attention_2"
    smolvlm_config._attn_implementation = "sdpa"


    # --- INSTANTIATE AND MANUALLY LOAD WEIGHTS ---

    # 1. Instantiate our randomly initialized custom model
    print(f"Instantiating custom SmolVLMForVQAClassification model...")
    model = SmolVLMForVQAClassification(config=smolvlm_config, answer2label_path=answer2label_path)

    # 2. Load the full pre-trained VQA model into a temporary variable
    print("Loading full pre-trained SmolVLMForVQAClassification model temporarily...")
    temp_vqa_model = SmolVLMForConditionalGeneration.from_pretrained(
        smolvlm_model_id, cache_dir=MY_CACHE_DIR, local_files_only=True, # TODO: set local_files_only to True if alreay cached
        torch_dtype=torch.bfloat16#, attn_implementation="flash_attention_2"
    ).to(device)

    # 3. Manually copy the weights from the temporary model to our custom model
    print("Manually copying pre-trained weights...")
    model.smolvlm.vision_model.load_state_dict(temp_vqa_model.model.vision_model.state_dict())
    model.smolvlm.connector.load_state_dict(temp_vqa_model.model.connector.state_dict())
    model.smolvlm.text_model.load_state_dict(temp_vqa_model.model.text_model.state_dict(), strict=False)

    print("✅ Pre-trained weights for vision and text models loaded successfully.")

    print(f"smolvlm.padding_idx: {model.smolvlm.padding_idx}, smolvlm.text_model.padding_idx: {model.smolvlm.text_model.padding_idx}")

    # Clean up the temporary model to save memory
    del temp_vqa_model
    torch.cuda.empty_cache()

    return model

def get_vivqa_smolvlm_phobert(
    smolvlm_model_id = SMOLVLM_ID,
    phobert_id = PHOBERT_MODEL_ID,
    answer2label_path = DEFAUT_ANSWER2LABEL,
    device = "cuda"
):

    # --- LOAD MODELS AND TOKENIZER ---
    print("Loading PhoBERT and SmolVLM configurations...")
    phobert_model = RobertaModel.from_pretrained(phobert_id)
    phobert_tokenizer = PhobertTokenizer.from_pretrained(phobert_id)

    temp_processor = SmolVLMProcessor.from_pretrained(smolvlm_model_id, use_fast=True)
    # temp_processor = AutoProcessor.from_pretrained(smolvlm_model_id, use_fast=True)
    tokens_to_add = {
        "additional_special_tokens": [
            temp_processor.fake_image_token,
            temp_processor.image_token,
            temp_processor.end_of_utterance_token,
            temp_processor.global_image_token
        ]
    }

    phobert_tokenizer.add_special_tokens(tokens_to_add)

    print("In get_vivqa_smolvlm_phobert():")
    image_token_id = phobert_tokenizer.convert_tokens_to_ids(temp_processor.image_token)
    fake_image_token_id = phobert_tokenizer.convert_tokens_to_ids(temp_processor.fake_image_token)
    end_of_utterance_token_id = phobert_tokenizer.convert_tokens_to_ids(temp_processor.end_of_utterance_token)

    print(f"Added {temp_processor.image_token} token to PhobertTokenizer with new ID: {image_token_id}")
    print(f"Added {temp_processor.fake_image_token} token to PhobertTokenizer with new ID: {fake_image_token_id}")
    print(f"Added {temp_processor.end_of_utterance_token} token to PhobertTokenizer with new ID: {end_of_utterance_token_id}")
    if isinstance(temp_processor, SmolVLMProcessor):
        global_image_token_id = phobert_tokenizer.convert_tokens_to_ids(temp_processor.global_image_token)
        print(f"Added {temp_processor.global_image_token} token to PhobertTokenizer with new ID: {global_image_token_id}")

    custom_smolvlm_config = SmolVLMConfig.from_pretrained(smolvlm_model_id)
    custom_smolvlm_config.pad_token_id = phobert_model.config.pad_token_id
    custom_smolvlm_config.text_config.pad_token_id = phobert_model.config.pad_token_id
    custom_smolvlm_config.image_token_id = image_token_id
    
    smolvlm_model = get_vivqa_smolvlm(
        smolvlm_model_id=smolvlm_model_id, 
        smolvlm_config= custom_smolvlm_config,
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
        
        smolvlm_model.smolvlm.text_model.embed_tokens.weight.data[0:num_tokens_to_copy, 0:source_dim] = \
            phobert_model.embeddings.word_embeddings.weight.data[0:num_tokens_to_copy, :].clone()
        print(f"Copied weights for {num_tokens_to_copy} tokens from PhoBERT (dim {source_dim}) into SmolVLM (dim {target_dim}).")

    # --- Update Special Token IDs in Model Config ---
    print("\nUpdating model configuration with new special token IDs...")
    smolvlm_model.smolvlm.text_model.config.bos_token_id = phobert_tokenizer.bos_token_id
    smolvlm_model.smolvlm.text_model.config.eos_token_id = phobert_tokenizer.eos_token_id
    smolvlm_model.smolvlm.text_model.config.pad_token_id = phobert_tokenizer.pad_token_id

    smolvlm_model.smolvlm.text_model.padding_idx = phobert_tokenizer.pad_token_id
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

def get_vivqa_smolvlm_phobert_with_adapter(
    args,
    smolvlm_model_id = SMOLVLM_ID,
    phobert_id = PHOBERT_MODEL_ID,
    answer2label_path = DEFAUT_ANSWER2LABEL,
    device = "cuda"
):
    ### 1. LOAD BASE SMOLVLM TO GET TARGET MAX LENGTH ###
    print("Loading base SmolVLM model to determine target configuration...")
    smolvlm_model = get_vivqa_smolvlm(
        smolvlm_model_id=smolvlm_model_id,
        answer2label_path=answer2label_path,
        device=device
    )

    # --- LOAD MODELS AND TOKENIZER ---
    print("Loading PhoBERT and SmolVLM configurations...")
    phobert_model = RobertaModel.from_pretrained(phobert_id)
    phobert_tokenizer = PhobertTokenizer.from_pretrained(phobert_id)

    temp_processor = SmolVLMProcessor.from_pretrained(smolvlm_model_id, use_fast=True)
    # temp_processor = AutoProcessor.from_pretrained(smolvlm_model_id, use_fast=True)
    tokens_to_add = {
        "additional_special_tokens": [
            temp_processor.fake_image_token,
            temp_processor.image_token,
            temp_processor.end_of_utterance_token,
            temp_processor.global_image_token
        ]
    }

    phobert_tokenizer.add_special_tokens(tokens_to_add)

    print("In get_vivqa_smolvlm_phobert():")
    image_token_id = phobert_tokenizer.convert_tokens_to_ids(temp_processor.image_token)
    fake_image_token_id = phobert_tokenizer.convert_tokens_to_ids(temp_processor.fake_image_token)
    end_of_utterance_token_id = phobert_tokenizer.convert_tokens_to_ids(temp_processor.end_of_utterance_token)

    print(f"Added {temp_processor.image_token} token to PhobertTokenizer with new ID: {image_token_id}")
    print(f"Added {temp_processor.fake_image_token} token to PhobertTokenizer with new ID: {fake_image_token_id}")
    print(f"Added {temp_processor.end_of_utterance_token} token to PhobertTokenizer with new ID: {end_of_utterance_token_id}")
    if isinstance(temp_processor, SmolVLMProcessor):
        global_image_token_id = phobert_tokenizer.convert_tokens_to_ids(temp_processor.global_image_token)
        print(f"Added {temp_processor.global_image_token} token to PhobertTokenizer with new ID: {global_image_token_id}")

    custom_smolvlm_config = SmolVLMConfig.from_pretrained(smolvlm_model_id)
    custom_smolvlm_config.pad_token_id = phobert_model.config.pad_token_id
    custom_smolvlm_config.text_config.pad_token_id = phobert_model.config.pad_token_id
    custom_smolvlm_config.image_token_id = image_token_id

    # =================== START: Old Logic ===================

    print("\nReplacing SmolVLM's LM embeddings with PhoBERT's and enabling adapter...")

    # 1. Get the original PhoBERT embedding layer and its dimensions
    source_embeddings = phobert_model.embeddings.word_embeddings
    source_vocab_size, source_dim = source_embeddings.weight.shape # source_dim is 768

    # 2. Create a NEW embedding layer with the correct final size and PhoBERT's dimension
    # The new layer will accommodate the special tokens and have the desired 768 dimension.
    new_vocab_size = len(phobert_tokenizer)
    new_embedding_layer = torch.nn.Embedding(new_vocab_size, source_dim, padding_idx=phobert_tokenizer.pad_token_id)

    # 3. Manually copy the weights from the original PhoBERT layer into the new one
    with torch.no_grad():
        new_embedding_layer.weight.data[:source_vocab_size, :] = source_embeddings.weight.data.clone()
        print(f"Created new embedding layer of shape {new_embedding_layer.weight.shape}.")
        print(f"Copied weights for {source_vocab_size} tokens from PhoBERT.")

    # 4. Set the model's input embeddings to be this new, correctly-sized layer
    # smolvlm_model.smolvlm.text_model.set_input_embeddings(new_embedding_layer)
    smolvlm_model.smolvlm.text_model.set_input_embeddings(deepcopy(new_embedding_layer))

    # 5. Add the adapter to project from PhoBERT's dim (768) to SmolVLM's dim (e.g., 2048)
    if args.phobert_embedding_adapter == 'linear':
        smolvlm_model.phobert_embedding_adapter = torch.nn.Linear(
            source_dim, 
            smolvlm_model.config.text_config.hidden_size
        )
    else:
        smolvlm_model.phobert_embedding_adapter = NonLinearAdapter(
            source_dim,
            smolvlm_model.config.text_config.hidden_size
        )
    smolvlm_model.use_phobert_adapter = True
    print("✅ PhoBERT embedding transplant with adapter is ready!")

    # =================== END: OLD LOGIC ===================

    # Update the model's config to reflect the new vocab size
    smolvlm_model.smolvlm.text_model.config.vocab_size = new_vocab_size
    smolvlm_model.config.text_config.vocab_size = new_vocab_size
    smolvlm_model.config.vocab_size = new_vocab_size

    print("✅ PhoBERT embedding transplant with adapter is ready!")

    # --- Update Special Token IDs in Model Config ---
    print("\nUpdating model configuration with new special token IDs...")
    smolvlm_model.smolvlm.text_model.config.bos_token_id = phobert_tokenizer.bos_token_id
    smolvlm_model.smolvlm.text_model.config.eos_token_id = phobert_tokenizer.eos_token_id
    smolvlm_model.smolvlm.text_model.config.pad_token_id = phobert_tokenizer.pad_token_id
    smolvlm_model.smolvlm.text_model.padding_idx = phobert_tokenizer.pad_token_id

    smolvlm_model.smolvlm.config.bos_token_id = phobert_tokenizer.bos_token_id
    smolvlm_model.smolvlm.config.eos_token_id = phobert_tokenizer.eos_token_id
    smolvlm_model.smolvlm.config.pad_token_id = phobert_tokenizer.pad_token_id

    smolvlm_model.smolvlm.padding_idx = phobert_tokenizer.pad_token_id

    print(f"smolvlm_model.smolvlm.text_model.padding_idx: {smolvlm_model.smolvlm.text_model.padding_idx}")
    print(f"smolvlm_model.smolvlm.text_model.config.pad_token_id: {smolvlm_model.smolvlm.text_model.config.pad_token_id}")

    print(f"phobert_tokenizer.pad_token_id: {phobert_tokenizer.pad_token_id}, phobert's pad_token_id: {phobert_model.config.pad_token_id}")

    assert smolvlm_model.smolvlm.padding_idx == phobert_model.config.pad_token_id, \
        f"pad_token_id mismatch! smolvlm: {smolvlm_model.smolvlm.padding_idx}, phobert: {phobert_model.config.pad_token_id}"
    assert smolvlm_model.smolvlm.text_model.padding_idx == phobert_model.config.pad_token_id, \
        f"pad_token_id mismatch! smolvlm: {smolvlm_model.smolvlm.text_model.padding_idx}, phobert: {phobert_model.config.pad_token_id}"
    print("✅ PhoBERT embedding transplant successful and verified!")

    # Clean up the standalone PhoBERT model to save memory
    del phobert_model
    del phobert_tokenizer
    torch.cuda.empty_cache()

    return smolvlm_model


class MySmolVLMModel(SmolVLMModel):
    def __init__(self, config: SmolVLMConfig):
        super().__init__(config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.BoolTensor] = None,
        image_hidden_states: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_phobert_adapter = False,
        phobert_embedding_adapter = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, SmolVLMBaseModelOutputWithPast]:
        r"""
        pixel_attention_mask (`torch.Tensor` of shape `(batch_size, image_size, image_size)`, *optional*):
            Mask to avoid performing attention on padding pixel indices.
        image_hidden_states (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The hidden states of the image encoder after modality projection.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.training and self.text_model.gradient_checkpointing and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

        # retrieve input_ids and inputs_embeds
        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_seen_tokens = 0
        if use_cache:
            if past_key_values is None:
                past_key_values = DynamicCache()
            past_seen_tokens = past_key_values.get_seq_length()

        if inputs_embeds is not None and input_ids is None and past_seen_tokens == 0:
            raise ValueError("When first calling the model, if input_embeds are passed, input_ids should not be None.")

        if inputs_embeds is None:
            inputs_embeds = self.text_model.get_input_embeddings()(input_ids).to(input_ids.device)

        # --- MODIFIED: Use the passed-in adapter ---
        if use_phobert_adapter:
            if phobert_embedding_adapter is None:
                raise ValueError("`phobert_embedding_adapter` must be provided when `use_phobert_adapter` is True.")
            # Project from PhoBERT's dimension (768) to the main model's dimension
            inputs_embeds = phobert_embedding_adapter(inputs_embeds)

        # START VISUAL INPUTS INTEGRATION
        if pixel_values is not None and image_hidden_states is not None:
            raise ValueError("You cannot specify both pixel_values and image_hidden_states at the same time")
        elif pixel_values is not None:
            image_hidden_states = self.get_image_features(pixel_values, pixel_attention_mask)
        elif image_hidden_states is not None:
            image_hidden_states = image_hidden_states.to(dtype=self.dtype, device=input_ids.device)

        if inputs_embeds is not None and image_hidden_states is not None:
            # When we generate, we don't want to replace the potential image_token_id that we generated by images
            # that simply don't exist
            inputs_embeds = self.inputs_merger(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_hidden_states=image_hidden_states,
            )

        outputs = self.text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        return SmolVLMBaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_hidden_states,
        )

class NonLinearAdapter(nn.Module):
    """A simple non-linear adapter using an expansion-compression MLP."""
    def __init__(self, input_dim, output_dim, expansion_factor=2):
        super().__init__()
        # Calculate an intermediate dimension
        hidden_dim = input_dim * expansion_factor
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.adapter(x)