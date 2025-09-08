from PIL import Image
import requests
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

print(processor)