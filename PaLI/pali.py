import torch
import torch.nn as nn
import timm

from transformers import MT5ForConditionalGeneration, AutoTokenizer

import torch
import torch.nn as nn
import timm
from transformers import MT5ForConditionalGeneration, AutoTokenizer, MT5Model, ViTConfig, ViTModel, AutoImageProcessor

class PaLI(nn.Module):
    def __init__(self, vit_model_name="google/vit-base-patch16-224", text_component_model_name="google/mt5-base", device=None):
        super().__init__()
        self.vit = ViTModel.from_pretrained(pretrained_model_name_or_path=vit_model_name)
        self.mt5 = MT5ForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=text_component_model_name)
        self.vision_proj = nn.Linear(self.vit.config.hidden_size, self.mt5.config.d_model) # Project to match mT5, to train fusion
        

        self.device = device

    def forward(self, pixel_values, input_ids, attention_mask, labels=None):

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

        output = self.mt5(
            attention_mask=encoder_attention_masks,
            encoder_outputs=encoder_outputs,
            labels=labels
        )
        return output