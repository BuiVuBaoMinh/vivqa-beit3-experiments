import json
import sys
import os

import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    PaliGemmaModel, PaliGemmaPreTrainedModel, PaliGemmaConfig, PaliGemmaForConditionalGeneration,
    Gemma2Model, SiglipVisionModel,
    PhobertTokenizer, RobertaModel
)
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.tokenization_utils_base import AddedToken
from transformers.models.paligemma.processing_paligemma import IMAGE_TOKEN, EXTRA_TOKENS

from transformers.utils import is_torchdynamo_compiling
from transformers.models.paligemma.modeling_paligemma import PaligemmaModelOutputWithPast
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack
from typing import Any, Callable, Optional, Tuple, Union, List

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
        
        # Use the base PaliGemma model which contains the vision and text encoders
        self.paligemma = MyPaliGemma(config)
        
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(config.text_config.hidden_size, num_labels)

        self.use_phobert_adapter = False
        self.phobert_embedding_adapter = None

    def forward(
        self,
        input_ids,
        pixel_values,
        attention_mask=None,
        token_type_ids=None,
        paligemma_labels=None, # Pass to Paligemma
        labels=None, # Use for our classification
        inputs_embeds=None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
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
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            use_phobert_adapter=self.use_phobert_adapter,
            phobert_embedding_adapter=self.phobert_embedding_adapter,
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

        # if not return_dict:
        #     output = (logits,) + (outputs.hidden_states, outputs.attentions)
        #     return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def get_input_embeddings(self):
        """
        Delegates the call to the underlying language model to get the input embeddings.
        This is required by the PEFT library for gradient checkpointing.
        """
        return self.paligemma.language_model.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        """
        Delegates the call to the underlying language model to set new input embeddings.
        """
        self.paligemma.language_model.set_input_embeddings(new_embeddings)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        """
        Delegates the call to the underlying PaliGemma model. This is required
        by PEFT for generation tasks.
        """
        return self.paligemma.prepare_inputs_for_generation(*args, **kwargs)
    
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
        num_vision_layers = len(self.paligemma.vision_tower.vision_model.encoder.layers)
        print(type(self.paligemma.vision_tower))
        print(f"Num layers: {num_text_layers + num_vision_layers}")
        return num_text_layers + num_vision_layers
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'position_embedding', 'class_embedding', 'logit_scale'}

def get_vivqa_paligemma(
        paligemma_model_id=PALIGEMMA_MODEL_ID,
        paligemma_config: PaliGemmaConfig = None,
        num_labels=336,
        answer2label_path=None,
        device="cuda",
        **kwargs # This will receive the quantization_config
) -> PaligemmaForVQAClassification:
    
    if paligemma_config is None:
        paligemma_config = PaliGemmaConfig.from_pretrained(paligemma_model_id)

    # 1. Instantiate your custom model wrapper. 
    # At this point, its `self.paligemma` attribute is the default, un-trained version.
    print(f"Instantiating custom PaligemmaForVQAClassification shell with {num_labels} labels...")
    model = PaligemmaForVQAClassification(config=paligemma_config, num_labels=num_labels, answer2label_path=answer2label_path)

    # 2. Load the full pre-trained PaliGemmaModel directly from Hugging Face with quantization.
    # This is the powerful base model we want to use.
    print("Loading full pre-trained and quantized PaliGemmaModel...")
    base_paligemma_model = PaliGemmaModel.from_pretrained(
        paligemma_model_id,
        cache_dir=MY_CACHE_DIR,
        local_files_only=True,
        **kwargs  # The quantization_config gets passed here
    )

    # 3. Manually copy the weights from the temporary model to our custom model
    print("Manually copying pre-trained weights...")
    model.paligemma.vision_tower.load_state_dict(base_paligemma_model.vision_tower.state_dict(), strict=False)
    model.paligemma.language_model.load_state_dict(base_paligemma_model.language_model.state_dict(), strict=False)
    print("✅ Pre-trained weights for vision and text models loaded successfully.")

    # Clean up the temporary model to save memory
    del base_paligemma_model
    torch.cuda.empty_cache()

    return model

def get_vivqa_paligemma_phobert_with_adapter(
    paligemma_model_id=PALIGEMMA_MODEL_ID,
    phobert_id=PHOBERT_MODEL_ID,
    num_labels=336,
    answer2label_path=None,
    device="cuda",
    **kwargs
):
    print("Loading PhoBERT model and tokenizer...")
    phobert_model = RobertaModel.from_pretrained(phobert_id, cache_dir=MY_CACHE_DIR)
    phobert_tokenizer = PhobertTokenizer.from_pretrained(phobert_id, cache_dir=MY_CACHE_DIR)

    custom_paligemma_config = PaliGemmaConfig.from_pretrained(paligemma_model_id, cache_dir=MY_CACHE_DIR)

    # Add special tokens to the PhoBERT tokenizer.
    if not hasattr(phobert_tokenizer, "image_token"):
        image_token = AddedToken(IMAGE_TOKEN, normalized=False, special=True)
        tokens_to_add = {"additional_special_tokens": [image_token]}
        phobert_tokenizer.add_special_tokens(tokens_to_add)
        # NOTE: We SUSPECT adding <loc****> and <seg***> tokens (EXTRA_TOKENS) to the PhoBERT tokenizer here.
                # These tokens are used in some vision-language models (like PaLI-Gemma) for fine-grained grounding:
                # - <loc****>: Refers to specific image regions or detected object locations.
                # - <seg***> : Refers to text segments, often used in layout-aware inputs (e.g., document understanding).
        # phobert_tokenizer.add_tokens(EXTRA_TOKENS)
        phobert_tokenizer.add_bos_token = False
        phobert_tokenizer.add_eos_token = False
    image_token_id = phobert_tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    print(f"Added '<image>' token to PhoBERT tokenizer with id: {image_token_id}. New vocab size: {len(phobert_tokenizer)}")

    # Get a standard Paligemma VQA model
    paligemma_model = get_vivqa_paligemma(
        paligemma_model_id=paligemma_model_id,
        num_labels=num_labels,
        answer2label_path=answer2label_path,
        device=device,
        **kwargs
    )

    # Initialize the adapter
    phobert_hidden_size = phobert_model.config.hidden_size # 768
    paligemma_hidden_size = paligemma_model.config.text_config.hidden_size # e.g., 2304
    # paligemma_model.phobert_embedding_adapter = nn.Linear(phobert_hidden_size, paligemma_hidden_size)
    adapter_hidden_dim = (phobert_hidden_size + paligemma_hidden_size) // 2
    paligemma_model.phobert_embedding_adapter = nn.Sequential(
        nn.Linear(phobert_hidden_size, adapter_hidden_dim),
        nn.GELU(),
        nn.Linear(adapter_hidden_dim, paligemma_hidden_size)
    )

    print("\nReplacing Paligemma's LM embeddings with PhoBERT's and enabling adapter...")
    
    # Get PhoBERT's word embedding layer
    source_embeddings = phobert_model.embeddings.word_embeddings
    
    # Set the entire language model's input embedding layer to be PhoBERT's
    paligemma_model.paligemma.language_model.set_input_embeddings(source_embeddings)
    print(f"Swapped embedding layer. New embedding dimension: {paligemma_model.paligemma.get_input_embeddings().weight.shape[1]}")

    new_vocab_size = len(phobert_tokenizer)
    paligemma_model.paligemma.resize_token_embeddings(new_vocab_size)
    print(f"Resized swapped embeddings to accommodate new tokens. Final embedding shape: {paligemma_model.paligemma.get_input_embeddings().weight.data.shape}")
    
    # Update the model's config to reflect the new vocab size from PhoBERT
    paligemma_model.paligemma.language_model.config.vocab_size = len(phobert_tokenizer)
    paligemma_model.config.text_config.vocab_size = len(phobert_tokenizer)
    
    # Activate the adapter in the forward pass
    paligemma_model.use_phobert_adapter = True

    print(f"Replaced LM embeddings. New embedding shape: {paligemma_model.paligemma.language_model.get_input_embeddings().weight.shape}")
    print("✅ PhoBERT embedding transplant with adapter is ready!")

    print("\nUpdating model configuration with new special token IDs...")
    # Update all relevant token IDs
    paligemma_model.paligemma.language_model.config.bos_token_id = phobert_tokenizer.bos_token_id
    paligemma_model.paligemma.language_model.config.eos_token_id = phobert_tokenizer.eos_token_id
    paligemma_model.paligemma.language_model.config.pad_token_id = phobert_tokenizer.pad_token_id
    
    # Also update the top-level config for consistency
    paligemma_model.config.pad_token_id = phobert_tokenizer.pad_token_id
    paligemma_model.config.text_config.pad_token_id = phobert_tokenizer.pad_token_id
    paligemma_model.config.image_token_index = image_token_id # Note: Paligemma uses image_token_index
    paligemma_model.config.image_token_id = image_token_id
    paligemma_model.config._vocab_size = new_vocab_size

    print("✅ PhoBERT embedding transplant with adapter successful!")
    
    del phobert_model
    del phobert_tokenizer
    
    return paligemma_model


def get_vivqa_paligemma_phobert_with_adapter_dev(
    paligemma_model_id=PALIGEMMA_MODEL_ID,
    phobert_id=PHOBERT_MODEL_ID,
    num_labels=336,
    answer2label_path=None,
    device="cuda"
):
    print("Loading PhoBERT model and tokenizer...")
    phobert_model = AutoModel.from_pretrained(phobert_id, cache_dir=MY_CACHE_DIR)
    phobert_tokenizer = PhobertTokenizer.from_pretrained(phobert_id, cache_dir=MY_CACHE_DIR)

    custom_paligemma_config = PaliGemmaConfig.from_pretrained(paligemma_model_id, cache_dir=MY_CACHE_DIR)

    # Add special tokens to the PhoBERT tokenizer.
    if not hasattr(phobert_tokenizer, "image_token"):
        image_token = AddedToken(IMAGE_TOKEN, normalized=False, special=True)
        tokens_to_add = {"additional_special_tokens": [image_token]}
        phobert_tokenizer.add_special_tokens(tokens_to_add)
        # NOTE: We do NOT add <loc****> and <seg***> tokens (EXTRA_TOKENS) to the PhoBERT tokenizer here.
                # These tokens are used in some vision-language models (like PaLI-Gemma) for fine-grained grounding:
                # - <loc****>: Refers to specific image regions or detected object locations.
                # - <seg***> : Refers to text segments, often used in layout-aware inputs (e.g., document understanding).
                #
                # However, our current dataset does NOT contain any input text that references these tokens.
                # Including 1,154 unused special tokens would unnecessarily increase the PhoBERT vocabulary size
                # (from 64,000 to 65,154), leading to extra randomly initialized embeddings that:
                #   - Increase memory usage
                #   - Add trainable parameters without utility
                #   - May destabilize fine-tuning
                #
                # Therefore, we only add the <image> token, which is required for vision-language fusion,
                # and we skip adding EXTRA_TOKENS to keep the model compact and focused.

                # phobert_tokenizer.add_tokens(EXTRA_TOKENS)
        phobert_tokenizer.add_bos_token = False
        phobert_tokenizer.add_eos_token = False
    image_token_id = phobert_tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    print(f"Added '<image>' token to PhoBERT tokenizer with id: {image_token_id}. New vocab size: {len(phobert_tokenizer)}")

    # Get a standard Paligemma VQA model
    paligemma_model = get_vivqa_paligemma(
        paligemma_model_id=paligemma_model_id,
        num_labels=num_labels,
        answer2label_path=answer2label_path,
        device=device
    )

    # Initialize the adapter
    phobert_hidden_size = phobert_model.config.hidden_size # 768
    paligemma_hidden_size = paligemma_model.config.text_config.hidden_size # e.g., 2304
    paligemma_model.phobert_embedding_adapter = nn.Linear(phobert_hidden_size, paligemma_hidden_size)

    print("\nReplacing Paligemma's LM embeddings with PhoBERT's and enabling adapter...")
    
    # Get PhoBERT's word embedding layer
    # source_embeddings = phobert_model.embeddings.word_embeddings
    source_embeddings = phobert_model.embeddings
    
    # Set the entire language model's input embedding layer to be PhoBERT's
    paligemma_model.paligemma.language_model.set_input_embeddings(source_embeddings)
    print(f"Swapped embedding layer. New embedding dimension: {paligemma_model.paligemma.get_input_embeddings()}")

    new_vocab_size = len(phobert_tokenizer)
    # Target the parent module that holds the word_embeddings layer
    embedding_module = paligemma_model.paligemma.get_input_embeddings()

    # Get the original word embedding layer
    old_word_embeddings = embedding_module.word_embeddings
    old_num_tokens, embedding_dim = old_word_embeddings.weight.shape

    # Only resize if the new size is different
    if old_num_tokens != new_vocab_size:
        print(f"Manually resizing word embeddings from {old_num_tokens} to {new_vocab_size}...")
        
        # 1. Create a new embedding layer with the correct new size
        new_word_embeddings = torch.nn.Embedding(
            num_embeddings=new_vocab_size,
            embedding_dim=embedding_dim,
            # Ensure the new layer is on the same device and has the same dtype
            device=old_word_embeddings.weight.device,
            dtype=old_word_embeddings.weight.dtype,
            padding_idx=old_word_embeddings.padding_idx
        )

    # 2. Copy the old weights into the new layer's weight matrix
    # The new tokens will have randomly initialized weights
    new_word_embeddings.weight.data[:old_num_tokens, :] = old_word_embeddings.weight.data

    # 3. Replace the old embedding layer with our new, resized layer
    embedding_module.word_embeddings = new_word_embeddings
    # Update the model's config to reflect the new vocab size from PhoBERT
    paligemma_model.paligemma.language_model.config.vocab_size = len(phobert_tokenizer)
    paligemma_model.config.text_config.vocab_size = len(phobert_tokenizer)
    
    # Activate the adapter in the forward pass
    paligemma_model.use_phobert_adapter = True

    # print(f"Replaced LM embeddings. New embedding shape: {paligemma_model.paligemma.language_model.get_input_embeddings().weight.shape}")
    print("✅ PhoBERT embedding transplant with adapter is ready!")

    print("\nUpdating model configuration with new special token IDs...")
    # Update all relevant token IDs
    paligemma_model.paligemma.language_model.config.bos_token_id = phobert_tokenizer.bos_token_id
    paligemma_model.paligemma.language_model.config.eos_token_id = phobert_tokenizer.eos_token_id
    paligemma_model.paligemma.language_model.config.pad_token_id = phobert_tokenizer.pad_token_id
    
    # Also update the top-level config for consistency
    paligemma_model.config.pad_token_id = phobert_tokenizer.pad_token_id
    paligemma_model.config.text_config.pad_token_id = phobert_tokenizer.pad_token_id
    paligemma_model.config.image_token_index = image_token_id # Note: Paligemma uses image_token_index
    paligemma_model.config.image_token_id = image_token_id

    print("✅ PhoBERT embedding transplant with adapter successful!")
    
    del phobert_model
    del phobert_tokenizer
    
    return paligemma_model

class MyPaliGemma(PaliGemmaModel):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__(config)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_phobert_adapter: bool = False,
        phobert_embedding_adapter: Optional[Callable] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, PaligemmaModelOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.text_config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.text_config.vocab_size]`.

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

        >>> model = PaliGemmaForConditionalGeneration.from_pretrained("google/paligemma2-3b-mix-224")
        >>> processor = AutoProcessor.from_pretrained("google/paligemma2-3b-mix-224")

        >>> prompt = "Where is the cat standing?"
        >>> url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, text=prompt,  return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs,)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Where is the cat standing?\nsnow"
        ```"""

        # if (input_ids is None) ^ (inputs_embeds is not None):
        #     raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        is_training = token_type_ids is not None and labels is not None

        # Replace image id woth PAD if the image token if OOV, to avoid index-errors
        if input_ids is not None and self.config.image_token_id >= self.vocab_size:
            special_image_mask = input_ids == self.config.image_token_id
            llm_input_ids = input_ids.clone()
            llm_input_ids[special_image_mask] = 0
        else:
            llm_input_ids = input_ids

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(llm_input_ids)

        if use_phobert_adapter:
            if phobert_embedding_adapter is None:
                raise ValueError("`phobert_embedding_adapter` must be provided when `use_phobert_adapter` is True.")
            # Project from PhoBERT's dimension (e.g., 768) to Paligemma's dimension (e.g., 2304)
            inputs_embeds = phobert_embedding_adapter(inputs_embeds)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0) + 1  # Paligemma positions are 1-indexed

        # Merge text and images
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values)

            # if input_ids is None:
            #     special_image_mask = inputs_embeds == self.get_input_embeddings()(
            #         torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            #     )
            # else:
            #     special_image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1)
            #     special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)

            special_image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1)
            special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)

            if not is_torchdynamo_compiling() and inputs_embeds[special_image_mask].numel() != image_features.numel():
                image_tokens_in_text = (special_image_mask).sum(dim=1).sum(dim=0)[0]
                raise ValueError(
                    f"Number of images does not match number of special image tokens in the input text. "
                    f"Got {image_tokens_in_text} image tokens in the text but {image_features.shape[0] * image_features.shape[1]} "
                    "tokens from image embeddings."
                )
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        causal_mask = self._update_causal_mask(
            attention_mask, token_type_ids, past_key_values, cache_position, inputs_embeds, is_training
        )
        outputs = self.language_model(
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        return PaligemmaModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )
