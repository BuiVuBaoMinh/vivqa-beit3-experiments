import json

import torch
import torch.nn as nn
import timm

from transformers import MT5ForConditionalGeneration, ViTModel, MT5Config, RobertaModel, AutoTokenizer, MT5ForSequenceClassification, ViTConfig

import py_vncorenlp

from glossary import segment_normalize

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

        self.vit_config = ViTConfig.from_pretrained(vit_model_name)
        self.vit_config.hidden_dropout_prob = 0.2
        self.vit_config.attention_probs_dropout_prob = 0.1

        self.vit = ViTModel.from_pretrained(
            pretrained_model_name_or_path=vit_model_name,
            config = self.vit_config
        )
        
        self.mt5_config = MT5Config.from_pretrained(text_component_model_name)
        self.mt5_config.dropout_rate = 0.3
        self.mt5_config.classifier_dropout = 0.1

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
        
        # This approach only copies the word_embeddings weight
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
        # self.vision_dropout = nn.Dropout(p=0.1) # add regularizing effect to vision_proj
        self.vision_dropout = nn.Dropout(p=0.2) # add regularizing effect to vision_proj
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

class PaLI_Classification(nn.Module):
    def __init__(
        self, 
        vit_model_name="google/vit-base-patch16-224", 
        text_component_model_name="google/mt5-base", 
        answer2label_path: str = None,
        device=None
    ):
        super().__init__()

        # 1. Load the answer mappings
        self.answer2label, self.label2answer = self._load_answer_mappings(answer2label_path)
        num_labels = len(self.label2answer)
        print(f"PaLI_Classification: num_labels = {num_labels}")

        self.vit_config = ViTConfig.from_pretrained(vit_model_name)
        # self.vit_config.hidden_dropout_prob = 0.2
        # self.vit_config.attention_probs_dropout_prob = 0.1
        self.vit = ViTModel.from_pretrained(
            pretrained_model_name_or_path=vit_model_name,
            config = self.vit_config
        )

        self.mt5_config = MT5Config.from_pretrained(
            text_component_model_name,
            num_labels = num_labels,
            id2label = self.label2answer,
            label2id = self.answer2label,
        )
        # self.mt5_config.dropout_rate = 0.3
        # self.mt5_config.classifier_dropout = 0.1
        self.mt5 = MT5ForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path = text_component_model_name,
            config = self.mt5_config
        )

        self.vision_proj = nn.Linear(self.vit.config.hidden_size, self.mt5.config.d_model) # Project to match mT5, to train fusion
        # self.vision_dropout = nn.Dropout(p=0.2) # add regularizing effect to vision_proj, 0.1 is initial value
        self.vision_dropout = nn.Dropout(p=0.5)
        self.vision_layernorm = nn.LayerNorm(self.mt5.config.d_model) # may stablilize fusion?
        self.device = device

    def forward(self, batch_size, pixel_values, input_ids, attention_mask, labels=None):

        vision_embeds = self.vit(pixel_values).last_hidden_state[:, 1:, :] # get unpooled output from ViT, according to paper.
        vision_embeds = self.vision_proj(vision_embeds)
        vision_embeds = self.vision_dropout(vision_embeds)
        vision_embeds = self.vision_layernorm(vision_embeds)
        image_attention_masks = torch.ones(vision_embeds.shape[:2], dtype=torch.long, device=self.device)

        text_embeds = self.mt5.transformer.shared(input_ids)
        text_attention_masks = attention_mask

        encoder_input_embeds = torch.cat([vision_embeds, text_embeds], dim=1)
        encoder_attention_masks = torch.cat([image_attention_masks, text_attention_masks], dim=1)

        # Replicate MT5Model's forward (MT5ForSequenceClassification.tranformers) to get decoder_output
        encoder_outputs = self.mt5.transformer.encoder(
            inputs_embeds=encoder_input_embeds,
            attention_mask=encoder_attention_masks
        )

        dummy_input_ids = torch.full((batch_size, 2), fill_value=self.mt5.config.pad_token_id, device=self.device)
        dummy_input_ids[:, -1] = self.mt5.config.eos_token_id  # Ensure one <eos> per sample

        output = self.mt5(
            input_ids = dummy_input_ids,
            attention_mask=encoder_attention_masks,
            encoder_outputs=encoder_outputs,
            labels=labels
        )
        return output

    def infer(self, batch_size, image, input_ids, attention_mask):
        """
        Inference function to get a predicted answer.
        This is equivalent to the `.generate()` method of generative models.

        Args:
            image: The input image tensor.
            input_ids (torch.LongTensor): Token IDs for the question.
            attention_mask (torch.LongTensor): Attention mask for the question.

        Returns:
            str: The predicted answer string.
        """
        # Run the forward pass without labels
        outputs = self.forward(
            batch_size=batch_size,
            image=image, 
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=None
        )
        
        # Get the logits from the output
        # Logits will have shape (batch_size, num_labels), e.g., (1, 335)
        logits = outputs.logits
        
        # Find the index of the highest logit
        predicted_class_id = torch.argmax(logits, dim=-1).item()
        
        # Convert the predicted ID back to a string answer
        return self.mt5.config.id2label[predicted_class_id]

    def _load_answer_mappings(self, file_path):
        """
        Loads the answer to label mappings from the provided JSONL file.
        """
        answer2label = {} # there are same answers with different labels in translated en!
        label2answer = {}
        seen_answers = set()
        unique_answers = []
        with open(file_path, 'r') as f:
            for line in f:
                item = json.loads(line.strip())

                label = item['label']
                answer = item['answer']

                label2answer[label] = answer

                if item['answer'] in seen_answers:
                    continue  # Skip duplicates
                seen_answers.add(answer)
                unique_answers.append(answer)

                answer2label[answer] = label

        # assert len(answer2label) == len(label2answer), \
        if len(answer2label) != len(label2answer):
            print(f"Warning: answer2label and label2answer dicts mismatch {len(answer2label)} vs {len(label2answer)}!")
            print("This could be translation or data source error.")
            print("Modifying label2answer to match answer2label..")
            label2answer = {}
            label2answer = {v: k for k, v in answer2label.items()}

        # Duplication removal could results in labels out of bound
        answer2label = {ans: idx for idx, ans in enumerate(unique_answers)}
        label2answer = {idx: ans for ans, idx in answer2label.items()}
        print(f"Reindexed {len(answer2label)} unique answers to labels 0 through {len(answer2label) - 1}.")

        answer2label["UNKNOWN"] = len(answer2label)
        label2answer[len(label2answer)] = "UNKNOWN"
        print("Append a default UNKNOWN label.")

        print(f"Loaded {len(answer2label)-1} classes from {file_path}.")
        return answer2label, label2answer
    
    def get_num_layers(self):
        # return {
        # 'vit': len(self.vit.encoder.layer),
        # 'mt5': self.mt5.encoder.config.num_layers
        # }
        return len(self.vit.encoder.layer) + self.mt5.encoder.transformer.config.num_layers

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'logit_scale'}


class PaLI_Classification_PhoBERT(nn.Module):
    def __init__(
        self, 
        vit_model_name="google/vit-base-patch16-224", 
        text_component_model_name="google/mt5-base",
        phobert_model_name="vinai/phobert-base-v2",
        answer2label_path: str = None,
        device=None
    ):
        super().__init__()

        # 1. Load the answer mappings
        self.answer2label, self.label2answer = self._load_answer_mappings(answer2label_path)
        num_labels = len(self.label2answer)
        print(f"PaLI_Classification: num_labels = {num_labels}")

        self.vit_config = ViTConfig.from_pretrained(vit_model_name)
        self.vit_config.hidden_dropout_prob = 0.2
        self.vit_config.attention_probs_dropout_prob = 0.1

        self.vit = ViTModel.from_pretrained(
            pretrained_model_name_or_path=vit_model_name,
            config = self.vit_config
        )

        self.mt5_config = MT5Config.from_pretrained(
            text_component_model_name,
            num_labels = num_labels,
            id2label = self.label2answer,
            label2id = self.answer2label,
        )
        self.mt5_config.dropout_rate = 0.3
        self.mt5_config.classifier_dropout = 0.1

        self.mt5 = MT5ForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path = text_component_model_name,
            config = self.mt5_config
        )

        phobert_tokenizer = AutoTokenizer.from_pretrained(phobert_model_name)
        phobert = RobertaModel.from_pretrained(phobert_model_name)

        assert self.mt5.config.d_model == phobert.config.hidden_size, \
            f"Embeddings dim mismatch. mt5:{self.mt5.config.d_model}, phobert: {phobert.config.hidden_size}"
        
        original_input_embeddings = self.mt5.transformer.get_input_embeddings()
        
        new_vocab_size = len(phobert_tokenizer)

        self.mt5.transformer.resize_token_embeddings(new_vocab_size)

        print(f"Original input embedding shape: {original_input_embeddings.weight.shape}")
        print(f"New input embedding shape: {self.mt5.transformer.get_input_embeddings().weight.shape}")

        phobert_word_embeddings = phobert.embeddings.word_embeddings
        with torch.no_grad():
            self.mt5.transformer.get_input_embeddings().weight.data[:new_vocab_size, :] = \
                phobert_word_embeddings.weight.data.clone()
            
        print("Weight transplantation of word embeddings complete.")

        # --- Update Special Token IDs in Model Config ---
        print("\nUpdating model configuration with new special token IDs...")
        self.mt5.config.bos_token_id = phobert_tokenizer.bos_token_id
        self.mt5.config.eos_token_id = phobert_tokenizer.eos_token_id
        self.mt5.config.pad_token_id = phobert_tokenizer.pad_token_id
        self.mt5.config.decoder_start_token_id = phobert_tokenizer.bos_token_id # T5 uses this
        
        self.vision_proj = nn.Linear(self.vit.config.hidden_size, self.mt5.config.d_model) # Project to match mT5, to train fusion
        self.vision_dropout = nn.Dropout(p=0.2) # add regularizing effect to vision_proj
        self.vision_layernorm = nn.LayerNorm(self.mt5.config.d_model) # may stablilize fusion?
        self.device = device

    def forward(self, batch_size, pixel_values, input_ids, attention_mask, labels=None):

        vision_embeds = self.vit(pixel_values).last_hidden_state[:, 1:, :] # get unpooled output from ViT, according to paper.
        vision_embeds = self.vision_dropout(vision_embeds)
        vision_embeds = self.vision_layernorm(vision_embeds)
        image_attention_masks = torch.ones(vision_embeds.shape[:2], dtype=torch.long, device=self.device)

        text_embeds = self.mt5.transformer.shared(input_ids)
        text_attention_masks = attention_mask

        encoder_input_embeds = torch.cat([vision_embeds, text_embeds], dim=1)
        encoder_attention_masks = torch.cat([image_attention_masks, text_attention_masks], dim=1)

        # Replicate MT5Model's forward (MT5ForSequenceClassification.tranformers) to get decoder_output
        encoder_outputs = self.mt5.transformer.encoder(
            inputs_embeds=encoder_input_embeds,
            attention_mask=encoder_attention_masks
        )

        dummy_input_ids = torch.full((batch_size, 2), fill_value=self.mt5.config.pad_token_id, device=self.device)
        dummy_input_ids[:, -1] = self.mt5.config.eos_token_id  # Ensure one <eos> per sample

        output = self.mt5(
            input_ids = dummy_input_ids,
            attention_mask=encoder_attention_masks,
            encoder_outputs=encoder_outputs,
            labels=labels
        )
        return output

    def infer(self, batch_size, image, input_ids, attention_mask):
        """
        Inference function to get a predicted answer.
        This is equivalent to the `.generate()` method of generative models.

        Args:
            image: The input image tensor.
            input_ids (torch.LongTensor): Token IDs for the question.
            attention_mask (torch.LongTensor): Attention mask for the question.

        Returns:
            str: The predicted answer string.
        """
        # Run the forward pass without labels
        outputs = self.forward(
            batch_size=batch_size,
            image=image, 
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=None
        )
        
        # Get the logits from the output
        # Logits will have shape (batch_size, num_labels), e.g., (1, 335)
        logits = outputs.logits
        
        # Find the index of the highest logit
        predicted_class_id = torch.argmax(logits, dim=-1).item()
        
        # Convert the predicted ID back to a string answer
        return self.mt5.config.id2label[predicted_class_id]

    def _load_answer_mappings(self, file_path):
        """
        Loads the answer to label mappings from the provided JSONL file.
        """
        answer2label = {} # there are same answers with different labels in translated en!
        label2answer = {}
        seen_answers = set()
        unique_answers = []
        with open(file_path, 'r') as f:
            for line in f:
                item = json.loads(line.strip())

                label = item['label']
                answer = segment_normalize(item['answer'])

                label2answer[label] = answer

                if answer in seen_answers:
                    continue  # Skip duplicates
                seen_answers.add(answer)
                unique_answers.append(answer)

                answer2label[answer] = label

        # assert len(answer2label) == len(label2answer), \
        if len(answer2label) != len(label2answer):
            print(f"Warning: answer2label and label2answer dicts mismatch {len(answer2label)} vs {len(label2answer)}!")
            print("This could be translation or data source error.")
            print("Modifying label2answer to match answer2label..")
            label2answer = {}
            label2answer = {v: k for k, v in answer2label.items()}

        # Duplication removal could results in labels out of bound
        answer2label = {ans: idx for idx, ans in enumerate(unique_answers)}
        label2answer = {idx: ans for ans, idx in answer2label.items()}
        print(f"Reindexed {len(answer2label)} unique answers to labels 0 through {len(answer2label) - 1}.")

        answer2label["UNKNOWN"] = len(answer2label)
        label2answer[len(label2answer)] = "UNKNOWN"
        print("Append a default UNKNOWN label.")

        print(f"Loaded {len(answer2label)-1} classes from {file_path}.")
        return answer2label, label2answer
    
    def get_num_layers(self):
        # return {
        # 'vit': len(self.vit.encoder.layer),
        # 'mt5': self.mt5.encoder.config.num_layers
        # }
        return len(self.vit.encoder.layer) + self.mt5.transformer.encoder.config.num_layers

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'logit_scale'}