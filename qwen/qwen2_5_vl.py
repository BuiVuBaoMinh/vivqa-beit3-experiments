import json
import sys
import os

import torch
import torch.nn as nn
from transformers import (
    Qwen2_5_VLConfig,
    AutoModel, AutoProcessor,
    Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLModel, Qwen2_5_VLPreTrainedModel,
    PhobertTokenizer, RobertaModel
)
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModelOutputWithPast

from typing import Optional, Tuple, Union, List, Callable

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from glossary import segment_normalize

QWEN_2_5_VL_3B_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
PHOBERT_MODEL_ID = "vinai/phobert-base-v2"
MY_CACHE_DIR = "/home/21khac.dd/bm/my-cache-dir"

class Qwen_2_5_VL_ForVQAClassification(Qwen2_5_VLPreTrainedModel):
    def __init__(self, qwen_base_model, num_labels, answer2label_path):
        super().__init__(qwen_base_model.config)

        self.answer2label, self.label2answer = self._load_answer_mappings(answer2label_path)
        self.num_labels = len(self.label2answer)
        print(f"Qwen_2_5_VL_ForVQAClassification: num_labels = {self.num_labels}")
        
        self.qwen_2_5_vl = qwen_base_model
        
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(qwen_base_model.config.text_config.hidden_size, num_labels)

        self.use_phobert_adapter = False
        self.phobert_embedding_adapter = None

    def forward(
        self,
        input_ids,
        pixel_values,
        attention_mask=None,
        labels=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        return_dict=True,
    ):

        outputs = self.qwen_2_5_vl(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            image_grid_thw = image_grid_thw,
            return_dict=True,
            use_phobert_adapter=self.use_phobert_adapter,
            phobert_embedding_adapter=self.phobert_embedding_adapter,
        )

        hidden_states = outputs.last_hidden_state

        pooled_output = hidden_states[:, -1, :]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
            attentions=outputs.attentions,
        )

    def get_input_embeddings(self):
        return self.qwen_2_5_vl.language_model.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.qwen_2_5_vl.language_model.set_input_embeddings(new_embeddings)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        """
        Delegates the call to the underlying Qwen2_5_VL model. This is required
        by PEFT for generation tasks.
        """
        return self.qwen_2_5_vl.prepare_inputs_for_generation(*args, **kwargs)
    
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

        print(f"Qwen_2_5_VL_ForVQAClassification: Loaded {len(answer_to_label)} answer-to-label mappings from {file_path}.")
        return answer_to_label, label_to_answer


    def get_num_layers(self):
        """
        Returns the total number of transformer layers in both the
        vision and language model backbones.
        """
        num_text_layers = len(self.qwen_2_5_vl.language_model.layers)
        num_vision_layers = len(self.qwen_2_5_vl.visual.blocks)
        print(f"Num layers: {num_text_layers + num_vision_layers}")
        return num_text_layers + num_vision_layers
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'position_embedding', 'class_embedding', 'logit_scale'}

def get_vivqa_qwen_2_5_vl(
        qwen_2_5_vl_model_id = QWEN_2_5_VL_3B_ID,
        qwen_2_5_vl_config: Qwen2_5_VLConfig = None,
        num_labels=336,
        answer2label_path=None,
        device="cuda",
        **kwargs
) -> Qwen_2_5_VL_ForVQAClassification:
    
    print("Loading full pre-trained and quantized Qwen2_5_VLModel...")
    # Step 1: Load the base model from Hugging Face with quantization enabled.
    # The object created is of the class `Qwen2_5_VLModel`.
    base_qwen_2_5_vl_model = Qwen2_5_VLModel.from_pretrained(
        qwen_2_5_vl_model_id,
        cache_dir=MY_CACHE_DIR,
        local_files_only=False,
        **kwargs
    )

    print("Dynamically changing the class of the loaded model to MyQwen_2_5_VL...")
    # Step 2: Change the class of the instance at runtime.
    # The `base_qwen_2_5_vl_model` object is now an instance of your custom class.
    # It inherits the custom `forward` method while retaining its quantized weights and layers.
    base_qwen_2_5_vl_model.__class__ = MyQwen_2_5_VL

    print(f"Instantiating custom Qwen_2_5_VL_ForVQAClassification wrapper...")
    # Step 3: Pass this modified, quantized object to your top-level wrapper.
    model = Qwen_2_5_VL_ForVQAClassification(
        qwen_base_model=base_qwen_2_5_vl_model, 
        num_labels=num_labels, 
        answer2label_path=answer2label_path
    )

    print("✅ Custom model with quantized base and custom forward method is ready.")
    return model

def get_vivqa_qwen_2_5_vl_phobert_with_adapter(
    qwen_2_5_vl_model_id=QWEN_2_5_VL_3B_ID,
    phobert_id=PHOBERT_MODEL_ID,
    num_labels=336,
    answer2label_path=None,
    device="cuda",
    **kwargs
):
    print("Loading PhoBERT model and tokenizer...")
    phobert_model = RobertaModel.from_pretrained(phobert_id, cache_dir=MY_CACHE_DIR)
    phobert_tokenizer = PhobertTokenizer.from_pretrained(phobert_id, cache_dir=MY_CACHE_DIR)

    custom_qwen_2_5_vl_config = Qwen2_5_VLConfig.from_pretrained(qwen_2_5_vl_model_id, cache_dir=MY_CACHE_DIR)

    # Get a standard Qwen2.5-VL VQA model
    qwen_2_5_vl_model = get_vivqa_qwen_2_5_vl(
        qwen_2_5_vl_model_id=qwen_2_5_vl_model_id,
        num_labels=num_labels,
        answer2label_path=answer2label_path,
        device=device,
        **kwargs
    )

    # Initialize the adapter
    phobert_hidden_size = phobert_model.config.hidden_size # 768
    qwen_2_5_vl_hidden_size = qwen_2_5_vl_model.config.text_config.hidden_size # e.g., 2304
    # paligemma_model.phobert_embedding_adapter = nn.Linear(phobert_hidden_size, paligemma_hidden_size)
    adapter_hidden_dim = (phobert_hidden_size + qwen_2_5_vl_hidden_size) // 2
    qwen_2_5_vl_model.phobert_embedding_adapter = nn.Sequential(
        nn.Linear(phobert_hidden_size, adapter_hidden_dim),
        nn.GELU(),
        nn.Linear(adapter_hidden_dim, qwen_2_5_vl_hidden_size)
    )

    print("\nReplacing Paligemma's LM embeddings with PhoBERT's and enabling adapter...")
    
    # Get PhoBERT's word embedding layer
    source_embeddings = phobert_model.embeddings.word_embeddings
    
    # Set the entire language model's input embedding layer to be PhoBERT's
    qwen_2_5_vl_model.qwen_2_5_vl.language_model.set_input_embeddings(source_embeddings)
    print(f"Swapped embedding layer. New embedding dimension: {qwen_2_5_vl_model.qwen_2_5_vl.get_input_embeddings().weight.shape[1]}")

    new_vocab_size = len(phobert_tokenizer)
    qwen_2_5_vl_model.qwen_2_5_vl.resize_token_embeddings(new_vocab_size)
    print(f"Resized swapped embeddings to accommodate new tokens. Final embedding shape: {qwen_2_5_vl_model.qwen_2_5_vl.get_input_embeddings().weight.data.shape}")
    
    # Update the model's config to reflect the new vocab size from PhoBERT
    qwen_2_5_vl_model.qwen_2_5_vl.language_model.config.vocab_size = len(phobert_tokenizer)
    qwen_2_5_vl_model.config.text_config.vocab_size = len(phobert_tokenizer)
    
    # Activate the adapter in the forward pass
    qwen_2_5_vl_model.use_phobert_adapter = True

    print(f"Replaced LM embeddings. New embedding shape: {qwen_2_5_vl_model.qwen_2_5_vl.language_model.get_input_embeddings().weight.shape}")
    print("✅ PhoBERT embedding transplant with adapter is ready!")

    print("\nUpdating model configuration with new special token IDs...")
    # Update all relevant token IDs
    qwen_2_5_vl_model.qwen_2_5_vl.language_model.config.bos_token_id = phobert_tokenizer.bos_token_id
    qwen_2_5_vl_model.qwen_2_5_vl.language_model.config.eos_token_id = phobert_tokenizer.eos_token_id
    qwen_2_5_vl_model.qwen_2_5_vl.language_model.config.pad_token_id = phobert_tokenizer.pad_token_id
    
    # Also update the top-level config for consistency
    qwen_2_5_vl_model.config.pad_token_id = phobert_tokenizer.pad_token_id
    qwen_2_5_vl_model.config.text_config.pad_token_id = phobert_tokenizer.pad_token_id
    qwen_2_5_vl_model.config._vocab_size = new_vocab_size

    print("✅ PhoBERT embedding transplant with adapter successful!")
    
    del phobert_model
    del phobert_tokenizer
    
    return qwen_2_5_vl_model


class MyQwen_2_5_VL(Qwen2_5_VLModel):
    def __init__(self, config: Qwen2_5_VLConfig):
        super().__init__(config)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        use_phobert_adapter: bool = False,
        phobert_embedding_adapter: Optional[Callable] = None,
    ) -> Union[Tuple, Qwen2_5_VLModelOutputWithPast]:
        r"""
        pixel_values_videos (`torch.FloatTensor` of shape `(seq_length, num_channels * temporal_size * image_size * image_size)):
            The tensors corresponding to the input videos. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`Qwen2_5_VLImageProcessor.__call__`] for details. [`Qwen2_5_VLProcessor`] uses
            [`Qwen2_5_VLImageProcessor`] for processing videos.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
        second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
            The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

            if use_phobert_adapter:
                if phobert_embedding_adapter is None:
                    raise ValueError("`phobert_embedding_adapter` must be provided when `use_phobert_adapter` is True.")
                # Project from PhoBERT's dimension (e.g., 768) to Paligemma's dimension (e.g., 2304)
                inputs_embeds = phobert_embedding_adapter(inputs_embeds)

            if pixel_values is not None:
                image_embeds = self.get_image_features(pixel_values, image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
        )

        output = Qwen2_5_VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )
        return output if return_dict else output.to_tuple()