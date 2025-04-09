from deep_translator import GoogleTranslator
import json
import os
import sys

json_folder_path = "./data/openvivqa/"
json_path = "data/openvivqa/original/openvivqa_test_v2.json"

output_translated_path = os.path.join(json_folder_path, "translated/test/openvivqa_test_translated_lines.jsonl")

failed_translation_path = os.path.join(json_folder_path, "translated/test/failed_test_translations.txt")

ggl_translator = GoogleTranslator(source='vi', target='en')

translated_ids = set()
failed_translations_ids = set()

# Load existing translated data
if os.path.exists(output_translated_path):
    with open(output_translated_path, 'r', encoding='utf-8') as outfile:
        try:
            for line in outfile:
                existing_data = json.loads(line)
                translated_ids.add(existing_data['id'])
        except json.JSONDecodeError:
            print("Error decoding JSON from existing translated data.")
            sys.exit(1)

# Load failed translations
if os.path.exists(failed_translation_path):
    with open(failed_translation_path, 'r', encoding='utf-8') as f:
        try:
            for line in f:
                failed_translations_ids.add(line.strip())
        except Exception as e:
            print(f"Error reading failed translations: {e}")
            sys.exit(1)


# Load original data
with open(json_path, "r", encoding='utf-8') as in_file, \
    open(output_translated_path, "a", encoding='utf-8') as out_file, \
    open(failed_translation_path, "w", encoding='utf-8') as failed_file:
    
    original_data_annotations = json.load(in_file)["annotations"]

    for item in original_data_annotations:

        if item["id"] in translated_ids:
            continue  # Skip already translated items

        try:
            item["question"] = ggl_translator.translate(item["question"])
        except Exception as e:
            print(f"Error translating question for ID {item['id']}: {e}")
            failed_translations_ids.add(item["id"])
            continue
        # Translate answers
        for j in range(len(item["answers"])):
            try:
                item["answers"][j] = ggl_translator.translate(item["answers"][j])
            except Exception as e:
                print(f"Error translating answer for ID {item['id']}: {e}")
                failed_translations_ids.add(item["id"])
                continue

        # Save the translated item
        out_file.write(json.dumps(item, ensure_ascii=False) + "\n")
        out_file.flush()
        translated_ids.add(item["id"])

