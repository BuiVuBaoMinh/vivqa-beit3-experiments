import torch
import torch.nn as nn
import timm

from transformers import MT5ForConditionalGeneration, ViTModel, MT5Config, RobertaModel, AutoTokenizer, MT5ForQuestionAnswering

import py_vncorenlp

class PaLI(nn.Module):
    def __init__(self, vit_model_name="google/vit-base-patch16-224", text_component_model_name="google/mt5-base", device=None):
        super().__init__()
        self.vit = ViTModel.from_pretrained(pretrained_model_name_or_path=vit_model_name)
        self.mt5_config = MT5Config.from_pretrained(text_component_model_name)
        self.mt5 = MT5ForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path = text_component_model_name,
            config = self.mt5_config
        )
        self.vision_proj = nn.Linear(self.vit.config.hidden_size, self.mt5.config.d_model) # Project to match mT5, to train fusion
        self.vision_dropout = nn.Dropout(p=0.1) # add regularizing effect to vision_proj
        self.vision_layernorm = nn.LayerNorm(self.mt5.config.d_model) # may stablilize fusion?
        self.device = device

    def forward(self, pixel_values, input_ids, attention_mask, labels=None):

        vision_embeds = self.vit(pixel_values).last_hidden_state[:, 1:, :] # get unpooled output from ViT, according to paper.
        vision_embeds = self.vision_proj(vision_embeds)
        vision_embeds = self.vision_dropout(vision_embeds)
        vision_embeds = self.vision_layernorm(vision_embeds)
        image_attention_masks = torch.ones(vision_embeds.shape[:2], dtype=torch.long, device=self.device)

        text_embeds = self.mt5.shared(input_ids)
        text_attention_masks = attention_mask

        encoder_input_embeds = torch.cat([vision_embeds, text_embeds], dim=1)
        encoder_attention_masks = torch.cat([image_attention_masks, text_attention_masks], dim=1)

        encoder_outputs = self.mt5.encoder(
            inputs_embeds=encoder_input_embeds,
            attention_mask=encoder_attention_masks
        )

        output = self.mt5(
            attention_mask=encoder_attention_masks,
            encoder_outputs=encoder_outputs,
            labels=labels
        )
        return output

    def generate(self, pixel_values, input_ids, attention_mask, labels=None, **generate_kwargs):
        vision_embeds = self.vit(pixel_values).last_hidden_state[:, 1:, :] # get unpooled output from ViT, according to paper.
        vision_embeds = self.vision_proj(vision_embeds)
        image_attention_masks = torch.ones(vision_embeds.shape[:2], dtype=torch.long, device=self.device)

        text_embeds = self.mt5.shared(input_ids)
        text_attention_masks = attention_mask

        encoder_input_embeds = torch.cat([vision_embeds, text_embeds], dim=1)
        encoder_attention_masks = torch.cat([image_attention_masks, text_attention_masks], dim=1)

        encoder_outputs = self.mt5.encoder(
            inputs_embeds=encoder_input_embeds,
            attention_mask=encoder_attention_masks
        )

        generated_ids = self.mt5.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attention_masks,
            **generate_kwargs
        )

        return generated_ids

    
    def get_num_layers(self):
        # return {
        # 'vit': len(self.vit.encoder.layer),
        # 'mt5': self.mt5.encoder.config.num_layers
        # }
        return len(self.vit.encoder.layer) + self.mt5.encoder.config.num_layers

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'logit_scale'}


class PaLI_PhoBERT(nn.Module):
    def __init__(
        self,
        vit_model_name="google/vit-base-patch16-224",
        text_component_model_name="google/mt5-base",
        phobert_model_name="vinai/phobert-base-v2",
        device=None
    ):
        super().__init__()
        self.vit = ViTModel.from_pretrained(pretrained_model_name_or_path=vit_model_name)
        self.mt5_config = MT5Config.from_pretrained(text_component_model_name)
        self.mt5 = MT5ForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path = text_component_model_name,
            config = self.mt5_config
        )

        phobert_tokenizer = AutoTokenizer.from_pretrained(phobert_model_name)
        phobert = RobertaModel.from_pretrained(phobert_model_name)

        assert self.mt5.config.d_model == phobert.config.hidden_size, \
            f"Embeddings dim mismatch. mt5:{self.mt5.config.d_model}, phobert: {phobert.config.hidden_size}"

        original_input_embeddings = self.mt5.get_input_embeddings()
        original_output_embeddings = self.mt5.get_output_embeddings()
        
        new_vocab_size = len(phobert_tokenizer)
        self.mt5.resize_token_embeddings(new_vocab_size)

        print(f"Original input embedding shape: {original_input_embeddings.weight.shape}")
        print(f"New input embedding shape: {self.mt5.get_input_embeddings().weight.shape}")
        print(f"Original output embedding shape: {original_output_embeddings.weight.shape}")
        print(f"New output embedding shape: {self.mt5.get_output_embeddings().weight.shape}")

        assert self.mt5.get_input_embeddings().weight.shape[0] == new_vocab_size ,"Input embedding mismatch"
        assert self.mt5.get_output_embeddings().weight.shape[0] == new_vocab_size, "Output embedding mismatch"

        # --- Transplant Embedding Weights ---
        phobert_word_embeddings = phobert.embeddings.word_embeddings.weight.data
        with torch.no_grad():
            self.mt5.get_input_embeddings().weight.data[:new_vocab_size, :] = phobert_word_embeddings.clone()
            # For mT5, input and output embeddings are tied by default,
            # so resizing and modifying the input embeddings should also affect the output embeddings (lm_head).
            # We can verify this.
            if self.mt5.get_input_embeddings() is self.mt5.get_output_embeddings():
                 print("Input and output embeddings are tied. No separate transplantation needed for the output head.")
            else:
                print("Input and output embeddings are not tied. Transplanting output head weights separately.")
                self.mt5.get_output_embeddings().weight.data[:new_vocab_size, :] = phobert_word_embeddings.clone()

        print("Weight transplantation of word embeddings complete.")

        # --- Update Special Token IDs in Model Config ---
        print("\nUpdating model configuration with new special token IDs...")
        self.mt5.config.bos_token_id = phobert_tokenizer.bos_token_id
        self.mt5.config.eos_token_id = phobert_tokenizer.eos_token_id
        self.mt5.config.pad_token_id = phobert_tokenizer.pad_token_id
        self.mt5.config.decoder_start_token_id = phobert_tokenizer.bos_token_id # T5 uses this
        
        print("Model config updated.")
        self.vision_proj = nn.Linear(self.vit.config.hidden_size, self.mt5.config.d_model) # Project to match mT5, to train fusion
        self.vision_dropout = nn.Dropout(p=0.1) # add regularizing effect to vision_proj
        self.vision_layernorm = nn.LayerNorm(self.mt5.config.d_model) # may stablilize fusion?
        self.device = device

    def forward(self, pixel_values, input_ids, attention_mask, labels=None):

        vision_embeds = self.vit(pixel_values).last_hidden_state[:, 1:, :] # get unpooled output from ViT, according to paper.
        vision_embeds = self.vision_proj(vision_embeds)
        vision_embeds = self.vision_dropout(vision_embeds)
        vision_embeds = self.vision_layernorm(vision_embeds)
        image_attention_masks = torch.ones(vision_embeds.shape[:2], dtype=torch.long, device=self.device)

        text_embeds = self.mt5.shared(input_ids)
        text_attention_masks = attention_mask

        encoder_input_embeds = torch.cat([vision_embeds, text_embeds], dim=1)
        encoder_attention_masks = torch.cat([image_attention_masks, text_attention_masks], dim=1)

        encoder_outputs = self.mt5.encoder(
            inputs_embeds=encoder_input_embeds,
            attention_mask=encoder_attention_masks
        )

        output = self.mt5(
            attention_mask=encoder_attention_masks,
            encoder_outputs=encoder_outputs,
            labels=labels
        )
        return output

    def generate(self, pixel_values, input_ids, attention_mask, labels=None, **generate_kwargs):
        vision_embeds = self.vit(pixel_values).last_hidden_state[:, 1:, :] # get unpooled output from ViT, according to paper.
        vision_embeds = self.vision_proj(vision_embeds)
        image_attention_masks = torch.ones(vision_embeds.shape[:2], dtype=torch.long, device=self.device)

        text_embeds = self.mt5.shared(input_ids)
        text_attention_masks = attention_mask

        encoder_input_embeds = torch.cat([vision_embeds, text_embeds], dim=1)
        encoder_attention_masks = torch.cat([image_attention_masks, text_attention_masks], dim=1)

        encoder_outputs = self.mt5.encoder(
            inputs_embeds=encoder_input_embeds,
            attention_mask=encoder_attention_masks
        )

        generated_ids = self.mt5.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attention_masks,
            **generate_kwargs
        )

        return generated_ids

    
    def get_num_layers(self):
        # return {
        # 'vit': len(self.vit.encoder.layer),
        # 'mt5': self.mt5.encoder.config.num_layers
        # }
        return len(self.vit.encoder.layer) + self.mt5.encoder.config.num_layers

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'logit_scale'}

class PaLI_QA(nn.Module):
    def __init__(
        self, 
        vit_model_name="google/vit-base-patch16-224", 
        text_component_model_name="google/mt5-base", 
        num_labels=335,
        device=None
    ):
        super().__init__()
        self.vit = ViTModel.from_pretrained(pretrained_model_name_or_path=vit_model_name)

        self.mt5_config = MT5Config.from_pretrained(text_component_model_name)
        self.mt5_config.num_labels = num_labels
        self.mt5 = MT5ForQuestionAnswering.from_pretrained(
            pretrained_model_name_or_path = text_component_model_name,
            config = self.mt5_config
        )
        
        self.vision_proj = nn.Linear(self.vit.config.hidden_size, self.mt5.config.d_model) # Project to match mT5, to train fusion
        self.vision_dropout = nn.Dropout(p=0.1) # add regularizing effect to vision_proj
        self.vision_layernorm = nn.LayerNorm(self.mt5.config.d_model) # may stablilize fusion?
        self.device = device

    def forward(self, pixel_values, input_ids, attention_mask, labels=None):

        # The code below are copied from PaLI class, make necessary modifications
        vision_embeds = self.vit(pixel_values).last_hidden_state[:, 1:, :] # get unpooled output from ViT, according to paper.
        vision_embeds = self.vision_proj(vision_embeds)
        vision_embeds = self.vision_dropout(vision_embeds)
        vision_embeds = self.vision_layernorm(vision_embeds)
        image_attention_masks = torch.ones(vision_embeds.shape[:2], dtype=torch.long, device=self.device)

        text_embeds = self.mt5.shared(input_ids)
        text_attention_masks = attention_mask

        encoder_input_embeds = torch.cat([vision_embeds, text_embeds], dim=1)
        encoder_attention_masks = torch.cat([image_attention_masks, text_attention_masks], dim=1)

        encoder_outputs = self.mt5.encoder(
            inputs_embeds=encoder_input_embeds,
            attention_mask=encoder_attention_masks
        )

        output = self.mt5(
            attention_mask=encoder_attention_masks,
            encoder_outputs=encoder_outputs,
            labels=labels
        )
        return output

    # The code below are copied from PaLI class, make necessary modifications
    def generate(self, pixel_values, input_ids, attention_mask, labels=None, **generate_kwargs):
        vision_embeds = self.vit(pixel_values).last_hidden_state[:, 1:, :] # get unpooled output from ViT, according to paper.
        vision_embeds = self.vision_proj(vision_embeds)
        image_attention_masks = torch.ones(vision_embeds.shape[:2], dtype=torch.long, device=self.device)

        text_embeds = self.mt5.shared(input_ids)
        text_attention_masks = attention_mask

        encoder_input_embeds = torch.cat([vision_embeds, text_embeds], dim=1)
        encoder_attention_masks = torch.cat([image_attention_masks, text_attention_masks], dim=1)

        encoder_outputs = self.mt5.encoder(
            inputs_embeds=encoder_input_embeds,
            attention_mask=encoder_attention_masks
        )

        generated_ids = self.mt5.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attention_masks,
            **generate_kwargs
        )

        return generated_ids

    
    def get_num_layers(self):
        # return {
        # 'vit': len(self.vit.encoder.layer),
        # 'mt5': self.mt5.encoder.config.num_layers
        # }
        return len(self.vit.encoder.layer) + self.mt5.encoder.config.num_layers

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'logit_scale'}