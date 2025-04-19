from deep_translator import GoogleTranslator
import json
import os
from deep_translator.exceptions import (
    TranslationNotFound
)
import sys

json_paths = [
    # "data/openvivqa/original/openvivqa_test_v2.json",
    # "data/openvivqa/original/openvivqa_train_v2.json",
    "data/openvivqa/original/openvivqa_dev_v2.json",
]

ggl_translator = GoogleTranslator(source='vi', target='en')

max_retries = 2


for json_path in json_paths:
    # Output path
    out_path = json_path.replace(".json", "_translated_full.json")
    out_lines_path = json_path.replace(".json", "_translated_lines.jsonl")
    # Failed path
    failed_path = json_path.replace(".json", "_failed.json")

    output_dir = "data/openvivqa/translated/dummy_split/"
    # Extract split name from json paths, exclude .json, e.g. "test", "train", "dev"
    split_name = os.path.basename(json_path).split("_")[1].replace(".json", "")
    output_dir = output_dir.replace("dummy_split", split_name)
    print(f"Output directory: {output_dir}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    out_path = os.path.join(output_dir, os.path.basename(out_path))
    failed_path = os.path.join(output_dir, os.path.basename(failed_path))
    out_lines_path = os.path.join(output_dir, os.path.basename(out_lines_path))

    print(f"Output path: {out_path}")
    print(f"Failed path: {failed_path}")
    print(f"Output lines path: {out_lines_path}")

    with open(json_path, "r", encoding='utf-8') as origin_file, \
        open(out_lines_path, "a", encoding='utf-8') as out_lines_file, \
        open(failed_path, "w", encoding='utf-8') as failed_file:

        translated_ids = set()

        # Load original data
        original_data = json.load(origin_file)
        original_data_annotations = original_data["annotations"]
        original_data_images = original_data["images"]

        # Load already translated data if exists
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            with open(out_path, "r", encoding='utf-8') as of:
                translated_data = json.load(of)["annotations"]
                translated_ids = {item["id"] for item in translated_data}
                print(f"Loaded {len(translated_data)} already translated items.")
        else:
            translated_data = []
            translated_ids = set()

        failed_translations = []

        for item in original_data_annotations:
            print(f"Translating item with id {item['id']}")
            if item["id"] in translated_ids:
                print(f"Skipping already translated item with id {item['id']}")
                continue # Skip already translated items
            try:
                if item["question"]:
                    item["question"] = ggl_translator.translate(item["question"])

                for j in range(len(item["answers"])):
                    if item["answers"][j]:
                        item["answers"][j] = ggl_translator.translate(item["answers"][j])

                # translated_data.append(item)

                # Save to file after each item
                json.dump(item, out_lines_file, ensure_ascii=False)
                out_lines_file.write("\n")
                translated_ids.add(item["id"])
                translated_data.append(item)
    
            except Exception as e:
                print(f"Error for id {item['id']}: {e}")
                failed_translations.append(item)

    # Save failed translations to file
    if failed_translations:
        with open(failed_path, "w", encoding='utf-8') as failed_file:
            json.dump(failed_translations, failed_file, indent=4, ensure_ascii=False)
            print(f"Saved failed translations to {failed_path}")

    num_org_annots = len(original_data_annotations)
    num_translated_annots = len(translated_ids)

    if num_org_annots == num_translated_annots:
        print(f"All {num_org_annots} annotations translated successfully.")
        # Merge original images with translated annotations
        merged_data = {
            "images": original_data_images,
            "annotations": translated_data
        }
        # Save merged data to output file
        with open(out_path, "w", encoding='utf-8') as out_file:
            json.dump(merged_data, out_file, indent=4, ensure_ascii=False)
            print(f"Saved translated data to {out_path}")
    else:
        print(f"Translation incomplete: {num_translated_annots}/{num_org_annots} annotations translated.")
        # Save partially translated data to output file
        with open(out_path, "w", encoding='utf-8') as out_file:
            json.dump({"annotations": translated_data}, out_file, indent=4, ensure_ascii=False)
            print(f"Saved partially translated data to {out_path}")

