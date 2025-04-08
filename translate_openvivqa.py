from deep_translator import GoogleTranslator
import json
import os
import threading

json_paths = [
    "data/openvivqa/openvivqa_test_v2.json",
    "data/openvivqa/openvivqa_train_v2.json",
    "data/openvivqa/openvivqa_dev_v2.json",
    "data/openvivqa/openvivqa_dev_v2_copy.json",
]

ggl_translator = GoogleTranslator(source='vi', target='en')

write_locks = threading.Lock()

translated_ids = set()
failed_translations_ids = set()

if 


for json_path in json_paths:
    # Output path
    out_path = json_path.replace(".json", "_translated.json")

    # Load original data
    with open(json_path, "r", encoding='utf-8') as f:
        original_data_annotations = json.load(f)["annotations"]
        original_data_questions = json.load(f)["questions"]

    # Load already translated data if exists
    if os.path.exists(out_path):
        with open(out_path, "r", encoding='utf-8') as f:
            translated_data = json.load(f)["annotations"]
            translated_ids = {item["question_id"] for item in translated_data}
    else:
        translated_data = []
        translated_ids = set()

    for item in original_data_annotations:
        if item["question_id"] in translated_ids:
            continue  # Skip already translated items

        if item["question"]:
            item["question"] = ggl_translator.translate(item["question"])

        for j in range(len(item["answers"])):
            if item["answers"][j]:
                item["answers"][j] = ggl_translator.translate(item["answers"][j])

        translated_data.append(item)

        # Save to file after each item
        with open(out_path, "w", encoding='utf-8') as f:
            json.dump(translated_data, f, indent=4, ensure_ascii=False)
