import json
import sys
import os

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import (
    Blip2Config,
    Blip2Model,
    Blip2PreTrainedModel,
    Blip2ForConditionalGeneration, T5ForConditionalGeneration,
    AutoProcessor,
    PhobertTokenizer,
    RobertaModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput, Seq2SeqLMOutput
from transformers.processing_utils import Unpack
from transformers.models.blip_2.processing_blip_2 import AddedToken
from transformers.models.blip_2.modeling_blip_2 import Blip2ForConditionalGenerationModelOutput, KwargsForCausalLM
from typing import Any, Callable, Optional, Tuple, Union

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Define constants for model IDs and paths
BLIP2_DECODER_BASED_ID = "Salesforce/blip2-opt-2.7b"
BLIP2_ENCODER_DECODER_ID = "Salesforce/blip2-flan-t5-xl-coco"
BLIP2_MODEL_ID = BLIP2_ENCODER_DECODER_ID
PHOBERT_MODEL_ID = "vinai/phobert-base-v2"
MY_CACHE_DIR = "/home/21khac.dd/bm/my-cache-dir"
DEFAULT_ANSWER2LABEL = "/home/21khac.dd/bm/data/vivqa/annotations/dicts/answer2label_en_gemini_translated.txt"

class Blip2ForVQAClassification(Blip2PreTrainedModel):
    """
    A custom BLIP-2 model for Visual Question Answering framed as a classification task.
    This model adds a classification head on top of the pretrained Blip2Model.
    """
    def __init__(self, config: Blip2Config, num_labels: int, answer2label_path: str):
        super().__init__(config)

        self.answer2label, self.label2answer = self._load_answer_mappings(answer2label_path)
        self.num_labels = len(self.label2answer)
        print(f"Blip2ForVQAClassification: num_labels = {self.num_labels}")

        # Base BLIP-2 model
        # self.blip2 = Blip2Model(config)
        self.blip2 = MyBlip2Model(config)
        
        # Classification head
        # The hidden size is from the language model part of BLIP-2
        self.classifier = nn.Linear(config.text_config.hidden_size, self.num_labels)
        self.dropout = nn.Dropout(0.1)

        # The adapter projects from PhoBERT's dimension (768) to the language model's dimension
        self.use_phobert_adapter = False
        # self.phobert_embedding_adapter = nn.Linear(768, config.text_config.hidden_size)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor,
        decoder_input_ids: torch.LongTensor = None,
        decoder_attention_mask: torch.LongTensor = None,
        attention_mask=None,
        labels=None,
        return_dict=True,
    ):

        # Get the batch size and device from the input_ids tensor.
        # batch_size = input_ids.shape[0]
        # device = input_ids.device

        # Manually create the decoder_input_ids for the T5 model.
        # We'll start the decoder with the pad token. This is a common
        # technique for getting a single pooled output from a decoder
        # for a classification task.
        # decoder_input_ids = torch.full(
        #     (batch_size, 1),
        #     fill_value=self.config.text_config.pad_token_id,
        #     dtype=torch.long,
        #     device=device
        # )

        # Pass inputs to the base BLIP-2 model.
        if self.use_phobert_adapter:
            outputs = self.blip2(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                # Pass the adapter and the flag to activate it
                use_phobert_adapter=self.use_phobert_adapter,
                phobert_embedding_adapter=self.phobert_embedding_adapter,
            )
        else:
            outputs = self.blip2(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
            )

        # 2. Get the last hidden state from the language model's output.
        # Shape: (batch_size, sequence_length, hidden_size)
        last_hidden_state = outputs.language_model_outputs.decoder_hidden_states[-1]

        # 3. Use the hidden state of the LAST token as the pooled representation.
        # This token's state is conditioned on both the image (via Q-Former) and the full question.
        # Shape: (batch_size, hidden_size)
        pooled_output = last_hidden_state[:, -1, :]

        # 4. Apply dropout and the classification head
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            # Reconstruct tuple output
            output_tuple = (logits,) + (outputs.language_model_outputs.hidden_states, outputs.language_model_outputs.attentions)
            return ((loss,) + output_tuple) if loss is not None else output_tuple

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.language_model_outputs.decoder_hidden_states,
            attentions=outputs.language_model_outputs.decoder_attentions,
        )
    
    def _load_answer_mappings(self, file_path):
        """Loads the answer to label mappings from the provided file."""
        answer_to_label = {}
        label_to_answer = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                answer = item['answer']
                label = item['label']
                answer_to_label[answer] = label
                label_to_answer[label] = answer
        
        if "UNKNOWN" not in answer_to_label:
            new_label = len(answer_to_label)
            answer_to_label["UNKNOWN"] = new_label
            label_to_answer[new_label] = "UNKNOWN"
            print("Appended a default UNKNOWN label.")

        print(f"Blip2ForVQAClassification: Loaded {len(answer_to_label)} mappings from {file_path}.")
        return answer_to_label, label_to_answer

    def get_num_layers(self):
        """
        Returns the total number of transformer layers from the vision model,
        Q-Former, and language model.
        """
        num_vision_layers = self.blip2.vision_model.config.num_hidden_layers
        num_qformer_layers = self.blip2.qformer.config.num_hidden_layers
        num_text_layers = self.blip2.language_model.config.num_hidden_layers
        
        total_layers = num_vision_layers + num_qformer_layers + num_text_layers
        
        print(f"Number of vision layers: {num_vision_layers}")
        print(f"Number of Q-Former layers: {num_qformer_layers}")
        print(f"Number of language layers: {num_text_layers}")
        print(f"Total layers for decay: {total_layers}")
        
        return total_layers
    
    @torch.jit.ignore
    def no_weight_decay(self):
        """Specifies parameters that should not be subject to weight decay."""
        return {'bias', 'LayerNorm.weight', 'position_embedding', 'class_embedding', 'logit_scale'}

def get_vivqa_blip2(
    blip2_model_id: str = BLIP2_MODEL_ID,
    config: Blip2Config = None,
    num_labels: int = 336,
    answer2label_path: str = DEFAULT_ANSWER2LABEL,
    device: str = "cuda"
) -> Blip2ForVQAClassification:
    """
    Instantiates the custom Blip2ForVQAClassification model and loads pretrained weights.
    """
    # If no custom config is provided, load the default one.
    if config is None:
        config = Blip2Config.from_pretrained(blip2_model_id, cache_dir=MY_CACHE_DIR)
    
    print(f"Instantiating custom Blip2ForVQAClassification model with {num_labels} labels...")
    model = Blip2ForVQAClassification(config=config, num_labels=num_labels, answer2label_path=answer2label_path)

    print("Loading full pre-trained Blip2ForConditionalGeneration model temporarily...")
    temp_generative_model = Blip2ForConditionalGeneration.from_pretrained(
        blip2_model_id,
        cache_dir=MY_CACHE_DIR,
        torch_dtype=torch.bfloat16,
    ).to(device)

    print("Manually copying pre-trained weights...")
    # Copy weights from the vision model, qformer, and language model
    model.blip2.vision_model.load_state_dict(temp_generative_model.vision_model.state_dict())
    model.blip2.qformer.load_state_dict(temp_generative_model.qformer.state_dict())
    model.blip2.language_model.load_state_dict(temp_generative_model.language_model.state_dict())

    print("✅ Pre-trained weights for all components loaded successfully.")

    del temp_generative_model
    torch.cuda.empty_cache()

    return model

def get_vivqa_blip2_phobert(
    blip2_model_id: str = BLIP2_MODEL_ID,
    phobert_id: str = PHOBERT_MODEL_ID,
    num_labels: int = 336,
    answer2label_path: str = DEFAULT_ANSWER2LABEL,
    device: str = "cuda"
):
    """
    Creates a BLIP-2 VQA model with its language model embeddings replaced by PhoBERT's.
    """
    print("Loading PhoBERT tokenizer and base BLIP-2 config...")
    phobert_model = RobertaModel.from_pretrained(phobert_id, cache_dir=MY_CACHE_DIR)
    phobert_tokenizer = PhobertTokenizer.from_pretrained(phobert_id, cache_dir=MY_CACHE_DIR)

    custom_blip2_config = Blip2Config.from_pretrained(blip2_model_id, cache_dir=MY_CACHE_DIR)

    # 1. Add special tokens to the PhoBERT tokenizer.
    # The Blip2Processor would add "<image>", so we replicate that behavior here.
    image_token = AddedToken("<image>", normalized=False, special=True)
    phobert_tokenizer.add_tokens([image_token], special_tokens=True)
    image_token_id = phobert_tokenizer.convert_tokens_to_ids("<image>")

    print(f"Added '<image>' token to PhoBERT tokenizer with id: {image_token_id}. New vocab size: {len(phobert_tokenizer)}")

    # Get a standard BLIP-2 VQA model
    blip2_model = get_vivqa_blip2(
        blip2_model_id=blip2_model_id,
        num_labels=num_labels,
        answer2label_path=answer2label_path,
        device=device
    )

    print("\nPerforming weight transplant for the BLIP-2 language model embeddings...")
    
    # 1. Resize the language model's token embeddings
    new_vocab_size = len(phobert_tokenizer)
    blip2_model.blip2.language_model.resize_token_embeddings(new_vocab_size)
    print(f"Resized language model embeddings to: {blip2_model.blip2.language_model.get_input_embeddings().weight.data.shape}")

    # 2. Copy the weights from PhoBERT's embeddings
    with torch.no_grad():
        source_embeddings = phobert_model.embeddings.word_embeddings
        target_embeddings = blip2_model.blip2.language_model.get_input_embeddings()

        # Get the dimensions
        source_dim = source_embeddings.weight.shape[1] # This will be 768
        num_tokens_to_copy = source_embeddings.weight.shape[0]

        # Only copy into the first `source_dim` (768) columns of the target tensor
        target_embeddings.weight.data[:num_tokens_to_copy, :source_dim] = \
            source_embeddings.weight.data.clone().to(target_embeddings.weight.dtype)
        
        print(f"Copied weights for {num_tokens_to_copy} tokens from PhoBERT (dim {source_dim}) into BLIP-2's language model (dim {target_embeddings.weight.shape[1]}).")

    # 3. Update model configuration with new tokenizer settings
    print("\nUpdating model configuration with new special token IDs...")

    # Update all relevant token IDs
    blip2_model.blip2.language_model.config.bos_token_id = phobert_tokenizer.bos_token_id
    blip2_model.blip2.language_model.config.eos_token_id = phobert_tokenizer.eos_token_id
    blip2_model.blip2.language_model.config.pad_token_id = phobert_tokenizer.pad_token_id

    # Also update the top-level config for consistency
    blip2_model.config.text_config.pad_token_id = phobert_tokenizer.pad_token_id
    blip2_model.config.image_token_index = image_token_id

    print("✅ PhoBERT embedding transplant successful!")

    return blip2_model

def get_vivqa_blip2_phobert_with_adapter(
    blip2_model_id: str = BLIP2_MODEL_ID,
    phobert_id: str = PHOBERT_MODEL_ID,
    num_labels: int = 336,
    answer2label_path: str = DEFAULT_ANSWER2LABEL,
    device: str = "cuda"
):
    print("Loading PhoBERT model and tokenizer...")
    phobert_model = RobertaModel.from_pretrained(phobert_id, cache_dir=MY_CACHE_DIR)
    phobert_tokenizer = PhobertTokenizer.from_pretrained(phobert_id, cache_dir=MY_CACHE_DIR)
    
    custom_blip2_config = Blip2Config.from_pretrained(blip2_model_id, cache_dir=MY_CACHE_DIR)

    # Add special tokens to the PhoBERT tokenizer.
    # The Blip2Processor would add "<image>", so we replicate that behavior here.
    image_token = AddedToken("<image>", normalized=False, special=True)
    phobert_tokenizer.add_tokens([image_token], special_tokens=True)
    image_token_id = phobert_tokenizer.convert_tokens_to_ids("<image>")

    print(f"Added '<image>' token to PhoBERT tokenizer with id: {image_token_id}. New vocab size: {len(phobert_tokenizer)}")

    # Get a standard BLIP-2 VQA model
    blip2_model = get_vivqa_blip2(
        blip2_model_id=blip2_model_id,
        num_labels=num_labels,
        answer2label_path=answer2label_path,
        device=device
    )

    blip2_model.phobert_embedding_adapter = nn.Linear(768, blip2_model.config.text_config.hidden_size)

    print("\nReplacing BLIP-2's LM embeddings with PhoBERT's and enabling adapter...")
    
    # Get PhoBERT's word embedding layer
    source_embeddings = phobert_model.embeddings.word_embeddings
    
    # Set the entire language model's input embedding layer to be PhoBERT's
    blip2_model.blip2.language_model.set_input_embeddings(source_embeddings)
    
    # Update the model's config to reflect the new vocab size
    blip2_model.blip2.language_model.config.vocab_size = len(phobert_tokenizer)
    blip2_model.config.text_config.vocab_size = len(phobert_tokenizer)
    
    # 4. Activate the adapter in the forward pass
    blip2_model.use_phobert_adapter = True

    print(f"Replaced LM embeddings. New embedding shape: {blip2_model.blip2.language_model.get_input_embeddings().weight.shape}")
    print("✅ PhoBERT embedding transplant with adapter is ready!")

    # Optional: Freeze the new embedding layer and the main LM
    # for param in blip2_model.blip2.language_model.get_input_embeddings().parameters():
    #     param.requires_grad = False
    
    # Might also freeze the entire language model except for the adapter
    # for param in blip2_model.blip2.language_model.parameters():
    #     param.requires_grad = False
    # for param in blip2_model.phobert_embedding_adapter.parameters():
    #     param.requires_grad = True

    # Update model configuration with new tokenizer settings
    print("\nUpdating model configuration with new special token IDs...")

    # Update all relevant token IDs
    blip2_model.blip2.language_model.config.bos_token_id = phobert_tokenizer.bos_token_id
    blip2_model.blip2.language_model.config.eos_token_id = phobert_tokenizer.eos_token_id
    blip2_model.blip2.language_model.config.pad_token_id = phobert_tokenizer.pad_token_id

    # Also update the top-level config for consistency
    blip2_model.config.text_config.pad_token_id = phobert_tokenizer.pad_token_id
    blip2_model.config.text_config.bos_token_id = phobert_tokenizer.bos_token_id
    blip2_model.config.text_config.eos_token_id = phobert_tokenizer.eos_token_id

    blip2_model.config.image_token_index = image_token_id

    print("✅ PhoBERT embedding transplant with adapter successful!")

    del phobert_tokenizer
    del phobert_model

    return blip2_model

class MyBlip2Model(Blip2Model):
    def __init__(self, config: Blip2Config):
        super().__init__(config)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
        # --- MODIFIED: Accept the adapter and a flag from the parent ---
        use_phobert_adapter: bool = False,
        phobert_embedding_adapter: Optional[Callable] = None,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, Blip2ForConditionalGenerationModelOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # step 1: forward the images through the vision encoder,
        # to get image embeddings of shape (batch_size, seq_len, hidden_size)
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )
        image_embeds = vision_outputs[0]

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        query_output = query_outputs[0]

        # Qformer is kept in fp32, we downcast the output back if needed
        if query_output.dtype != image_embeds.dtype:
            query_output = query_output.to(image_embeds.dtype)

        # step 3: use the language model, conditioned on the query outputs and the prompt
        language_model_inputs = self.language_projection(query_output)
        language_model_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # --- MODIFIED: Use the passed-in adapter ---
        if use_phobert_adapter:
            if phobert_embedding_adapter is None:
                raise ValueError("`phobert_embedding_adapter` must be provided when `use_phobert_adapter` is True.")
            # Project from PhoBERT's dimension (768) to the main model's dimension
            inputs_embeds = phobert_embedding_adapter(inputs_embeds)

        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        expected_device = language_model_attention_mask.device
        attention_mask = torch.cat([language_model_attention_mask, attention_mask.to(expected_device)], dim=1)

        if self.config.use_decoder_only_language_model:
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )
            logits = outputs.logits if return_dict else outputs[0]
            loss = None
            # we compute the loss here since we need to take into account the sequence length of the query embeds
            if labels is not None:
                labels = labels.to(logits.device)
                logits = logits[:, -labels.size(1) :, :]
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous().to(logits.device)

                # Flatten the tokens
                loss_fct = CrossEntropyLoss(reduction="mean")

                loss = loss_fct(shift_logits.view(-1, self.config.text_config.vocab_size), shift_labels.view(-1))
        else:
            decoder_inputs_embeds = None
            if use_phobert_adapter:
                if phobert_embedding_adapter is None:
                    raise ValueError("`phobert_embedding_adapter` must be provided when `use_phobert_adapter` is True.")
                
                if decoder_input_ids is not None:
                    # 1. Get the raw 768-dim embeddings from the swapped PhoBERT layer.
                    raw_decoder_embeds = self.language_model.get_input_embeddings()(decoder_input_ids)
                    # 2. Apply the adapter to project them to the T5's 2048 dimension.
                    decoder_inputs_embeds = phobert_embedding_adapter(raw_decoder_embeds)

            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids if decoder_inputs_embeds is None else None,
                decoder_inputs_embeds=decoder_inputs_embeds,
                decoder_attention_mask=decoder_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,  # toggle for easier access to loss/logits below
                labels=labels,
                **kwargs,
            )

            loss = outputs.loss
            logits = outputs.logits
            outputs = outputs.to_tuple() if not return_dict else outputs

        if not return_dict:
            output = (logits, vision_outputs, query_outputs, outputs)
            return ((loss,) + output) if loss is not None else output

        return Blip2ForConditionalGenerationModelOutput(
            loss=loss,
            logits=logits,
            vision_outputs=vision_outputs,
            qformer_outputs=query_outputs,
            language_model_outputs=outputs,
        )