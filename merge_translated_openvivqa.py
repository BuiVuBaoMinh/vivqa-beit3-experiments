import json
import os
import sys

json_folder_path = "./data/openvivqa/"
json_path = "data/openvivqa/original/openvivqa_test_v2.json"

output_translated_jsonl = os.path.join(json_folder_path, "translated/test/openvivqa_test_translated_lines.jsonl")

failed_translation_path = os.path.join(json_folder_path, "translated/test/failed_test_translations.txt")

output_translated_full_json = os.path.join(json_folder_path, "translated/test/openvivqa_test_translated_full.json")

# Load original images
if os.path.exists(json_path):
    with open(json_path, 'r', encoding='utf-8') as origin, \
        open(output_translated_jsonl, "r", encoding='utf-8') as out_jsonl:

        try:
            original_images = json.load(origin)["images"]
        except Exception as e:
            print(f"Error reading original images: {e}")
            sys.exit(1)

        annotations = []

        try:
            annotations = [json.loads(line) for line in out_jsonl]
        except json.JSONDecodeError:   
            print("Error decoding JSON from existing translated data.")
            sys.exit(1)

        with open(output_translated_full_json, "w", encoding="utf-8") as out_json_full:
            json.dump({"images": original_images, "annotations": annotations}, out_json_full, ensure_ascii=False, indent=2)