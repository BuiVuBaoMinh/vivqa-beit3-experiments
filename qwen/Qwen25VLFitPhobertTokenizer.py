from transformers import PhobertTokenizer
from typing import List, Optional

import numpy as np
from transformers import PaliGemmaProcessor
from transformers.image_processing_utils import BatchFeature
from transformers.models.paligemma.processing_paligemma import IMAGE_TOKEN

from transformers.models.paligemma.processing_paligemma import (
    _validate_images_text_input_order,
    PaliGemmaProcessorKwargs,
    logger,
    _is_str_or_image,
    is_valid_image,
    build_string_from_input,
    make_flat_list_of_images
    )

class Qwen2_5_VLFitPhobertTokenizer(PhobertTokenizer):
    def __init__(
        self,
        vocab_file,
        merges_file,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        **kwargs,
    ):
        super().__init__(
            vocab_file,
            merges_file,
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            **kwargs,
        )
        self.add_bos_token = False
        self.add_eos_token = False

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:

        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = bos_token_id + token_ids_0 + eos_token_id

        if token_ids_1 is not None:
            output = output + bos_token_id + token_ids_1 + eos_token_id

        return output

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
    
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        bos_token_id = [1] if self.add_bos_token else []
        eos_token_id = [1] if self.add_eos_token else []

        if token_ids_1 is None:
            return bos_token_id + ([0] * len(token_ids_0)) + eos_token_id
        return (
            bos_token_id
            + ([0] * len(token_ids_0))
            + eos_token_id
            + bos_token_id
            + ([0] * len(token_ids_1))
            + eos_token_id
        )

class CustomPaliGemmaProcessor(PaliGemmaProcessor):
    def __call__(self, text, images, suffix=None, **kwargs):
        # 1. Manually construct the full prompt string.
        # This ensures the format is always correct: <image>\n{question}\n{answer}</s>
        # The single EOS token at the very end is expected by the model for generation.
        # check if images and text inputs are reversed for BC
        images, text = _validate_images_text_input_order(images, text)

        output_kwargs = self._merge_kwargs(
            PaliGemmaProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        suffix = output_kwargs["text_kwargs"].pop("suffix", None)

        return_token_type_ids = True if suffix is not None else False

        if images is None:
            raise ValueError("`images` are expected as arguments to a `PaliGemmaProcessor` instance.")
        if text is None:
            logger.warning_once(
                "You are using PaliGemma without a text prefix. It will perform as a picture-captioning model."
            )
            text = ""

        if _is_str_or_image(text):
            text = [text]
        elif isinstance(text, list) and _is_str_or_image(text[0]):
            pass

        if text is not None and images is not None:
            if not any(IMAGE_TOKEN in sample for sample in text):
                logger.warning(
                    "You are passing both `text` and `images` to `PaliGemmaProcessor`. The processor expects special "
                    "image tokens in the text, as many tokens as there are images per each text. It is recommended to "
                    "add `<image>` tokens in the very beginning of your text. For this call, we will infer how many images "
                    "each text has and add special tokens."
                )

                if isinstance(text, List) and isinstance(images, List):
                    if len(images) != len(text):
                        raise ValueError(
                            f"Received {len(images)} images for {len(text)} prompts. Each prompt should be associated with an image or list of images."
                        )

                # make a nested list of lists to be able to iterate over the images and text below
                if is_valid_image(images):
                    images = [[images]]
                elif isinstance(images, (list, tuple)) and is_valid_image(images[0]):
                    images = [[image] for image in images]
                elif not (
                    isinstance(images, (list, tuple))
                    and isinstance(images[0], (list, tuple))
                    and is_valid_image(images[0][0])
                ):
                    raise ValueError("images must be an image, list of images or list of list of images")

                input_strings = [
                    build_string_from_input(
                        prompt=prompt,
                        bos_token=self.tokenizer.bos_token,
                        image_seq_len=self.image_seq_length,
                        image_token=IMAGE_TOKEN,
                        num_images=len(image_list) if isinstance(image_list, list) else 1,
                    )
                    for prompt, image_list in zip(text, images)
                ]
                images = make_flat_list_of_images(images)
            else:
                expanded_samples = []
                for sample in text:
                    expanded_sample = sample.replace(IMAGE_TOKEN, IMAGE_TOKEN * self.image_seq_length)
                    bos_rfind_index = expanded_sample.rfind(IMAGE_TOKEN)
                    bos_index = bos_rfind_index + len(IMAGE_TOKEN) if bos_rfind_index != -1 else 0
                    expanded_sample = (
                        expanded_sample[:bos_index] + self.tokenizer.bos_token + expanded_sample[bos_index:]
                    )
                    expanded_samples.append(expanded_sample)
                input_strings = [f"{sample}\n" for sample in expanded_samples]

        if suffix is not None and _is_str_or_image(suffix):
            print(suffix)
            suffix = [suffix]
        if suffix is not None:
            suffix = [sfx + self.tokenizer.eos_token for sfx in suffix]
        pixel_values = self.image_processor(images, **output_kwargs["images_kwargs"])["pixel_values"]

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        
        inputs = self.tokenizer(
            input_strings,
            text_pair=suffix,
            return_token_type_ids=return_token_type_ids,
            **output_kwargs["text_kwargs"],
        )
        self._check_special_mm_tokens(input_strings, inputs, modalities=["image"])

        return_data = {**inputs, "pixel_values": pixel_values}

        if return_token_type_ids:
            labels = np.array(inputs["input_ids"])
            labels[np.array(inputs["token_type_ids"]) == 0] = -100
            return_data.update({"labels": labels})
        return BatchFeature(data=return_data, tensor_type=return_tensors)
