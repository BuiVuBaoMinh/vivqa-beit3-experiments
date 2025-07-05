import json

import torch
import torch.nn as nn
from transformers import BlipPreTrainedModel, BlipModel, BlipForQuestionAnswering, BlipConfig, PhobertTokenizer, RobertaModel, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput

from glossary import segment_normalize

class BlipForVQAClassification(BlipPreTrainedModel):
    def __init__(self, config, num_labels, answer2label_path):
        super().__init__(config)

        self.answer2label, self.label2answer = self._load_answer_mappings(answer2label_path)
        self.num_labels = len(self.label2answer)
        print(f"BlipForVQAClassification: num_labels = {self.num_labels}")
        
        # Use the base BLIP model which contains the vision and text encoders
        self.blip = BlipModel(config)
        
        # Add a dropout and a classification head
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(config.text_config.hidden_size, num_labels)

    def forward(
        self,
        input_ids,
        pixel_values,
        attention_mask=None,
        labels=None,
        return_dict=True,
    ):

        # 1. Get image embeddings from the vision model
        vision_outputs = self.blip.vision_model(
            pixel_values=pixel_values,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=return_dict
        )
        image_embeds = vision_outputs[0] 
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
        
        # 2. Pass image embeddings and text to the text model to perform fusion
        text_outputs = self.blip.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds, # Cross-attention
            encoder_attention_mask=image_atts,
            return_dict=return_dict,
        )

        # 3. Get the pooled representation of the FUSED output
        # The text_outputs[1] is the pooler_output from the text model
        pooled_output = text_outputs[1]

        pooled_output = self.dropout(pooled_output)
        
        # Pass the correctly fused and pooled output through our classification head
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            # Use standard CrossEntropyLoss for multi-class classification
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
        if not return_dict:
            output = (logits, vision_outputs, text_outputs)
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=text_outputs.hidden_states,
            attentions=text_outputs.attentions,
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

        print(f"ViVQAPaLIClassificationDataset: Loaded {len(answer_to_label)} answer-to-label mappings from {file_path}.")
        return answer_to_label, label_to_answer


    def get_num_layers(self):
        """
        Returns the number of transformer layers in the text model backbone.
        This is used by the optimizer factory to set up layer-wise learning rate decay,
        which will be applied to both the text and vision backbones.
        """
        return len(self.blip.text_model.encoder.layer)
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'position_embedding', 'class_embedding', 'logit_scale'}

def get_vivqa_blip(
        blip_id = "Salesforce/blip-vqa-base",
        num_labels = 336, # 335 labels plus an UNKNOWN
        answer2label_path = None,
        device = "cuda"
) -> BlipForVQAClassification:
    
    blip_config = BlipConfig.from_pretrained(blip_id)

    # # === Set Dropout ===
    # # Set dropout for the text model
    # blip_config.text_config.hidden_dropout_prob = 0.1
    # blip_config.text_config.attention_probs_dropout_prob = 0.1
    
    # # Set dropout for the vision model
    # blip_config.vision_config.dropout = 0.1
    # blip_config.vision_config.attention_dropout = 0.1
    
    # print("Updated model config with dropout.")
    # === End Set Dropout ===

    # --- INSTANTIATE AND MANUALLY LOAD WEIGHTS ---

    # 1. Instantiate our randomly initialized custom model
    print(f"Instantiating custom BlipForVQAClassification model with {num_labels} labels...")
    model = BlipForVQAClassification(config=blip_config, num_labels=num_labels, answer2label_path=answer2label_path)

    # 2. Load the full pre-trained VQA model into a temporary variable
    print("Loading full pre-trained BlipForQuestionAnswering model temporarily...")
    temp_vqa_model = BlipForQuestionAnswering.from_pretrained(blip_id).to(device)

    # 3. Manually copy the weights from the temporary model to our custom model
    print("Manually copying pre-trained weights...")
    model.blip.vision_model.load_state_dict(temp_vqa_model.vision_model.state_dict())
    # Note the source is temp_vqa_model.text_encoder and the destination is model.blip.text_model
    model.blip.text_model.load_state_dict(temp_vqa_model.text_encoder.state_dict(), strict=False)
    print("✅ Pre-trained weights for vision and text models loaded successfully.")

    # Clean up the temporary model to save memory
    del temp_vqa_model
    torch.cuda.empty_cache()

    return model

def get_vivqa_blip_phobert(
    blip_id = "Salesforce/blip-vqa-base",
    phobert_id = "vinai/phobert-base-v2",
    num_labels = 336, # 335 labels plus an UNKNOWN
    answer2label_path = None,
    device = "cuda"
):
    blip_model = get_vivqa_blip(
        blip_id=blip_id, num_labels=num_labels, device=device, answer2label_path=answer2label_path
    )

    # --- LOAD MODELS AND TOKENIZER ---
    print("Loading PhoBERT and BLIP configurations...")
    phobert_model = RobertaModel.from_pretrained(phobert_id)
    phobert_tokenizer = PhobertTokenizer.from_pretrained(phobert_id)
    
    # --- PERFORM WEIGHT TRANSPLANT ---
    print("\nPerforming weight transplant for the BLIP text model...")

    # 1. Resize the text model's embeddings
    new_vocab_size = len(phobert_tokenizer)
    blip_model.blip.text_model.resize_token_embeddings(new_vocab_size)
    print(f"Resized text model embeddings to: {new_vocab_size}")

    # 2. Copy the weights from PhoBERT
    with torch.no_grad():
        blip_model.blip.text_model.embeddings.word_embeddings.weight.data = \
            phobert_model.embeddings.word_embeddings.weight.data.clone()

    # --- Update Special Token IDs in Model Config ---
    print("\nUpdating model configuration with new special token IDs...")
    blip_model.blip.text_model.config.bos_token_id = phobert_tokenizer.bos_token_id
    blip_model.blip.text_model.config.eos_token_id = phobert_tokenizer.eos_token_id
    blip_model.blip.text_model.config.pad_token_id = phobert_tokenizer.pad_token_id

    # --- VERIFICATION ---
    assert torch.equal(blip_model.blip.text_model.embeddings.word_embeddings.weight, phobert_model.embeddings.word_embeddings.weight)
    print("✅ PhoBERT embedding transplant successful and verified!")

    return blip_model
