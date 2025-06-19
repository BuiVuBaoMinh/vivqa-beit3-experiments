import torch
import torch.nn as nn
import timm

from transformers import MT5ForConditionalGeneration, AutoTokenizer

import torch
import torch.nn as nn
import timm
from transformers import MT5ForConditionalGeneration, ViTModel, MT5Config

class PaLI(nn.Module):
    def __init__(self, vit_model_name="google/vit-base-patch16-224", text_component_model_name="google/mt5-base", device=None):
        super().__init__()
        self.vit = ViTModel.from_pretrained(pretrained_model_name_or_path=vit_model_name)
        self.mt5_config = MT5Config.from_pretrained(text_component_model_name) # TODO: vocab_size fits PhoBERT if using PhoBERT
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

        # TODO: Check the output of this function
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