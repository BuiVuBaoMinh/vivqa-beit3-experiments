import json
import os
from deep_translator import GoogleTranslator, MyMemoryTranslator
import pandas as pd
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


def process_vivqa_data(input_dir):
    # Merge train and test splits
    merged_df = pd.concat(
        [
            pd.read_csv(os.path.join(input_dir, "train.csv")), 
            pd.read_csv(os.path.join(input_dir, "test.csv"))
        ], ignore_index=True)

    # print(f"Total samples: {len(merged_df)}") # 15k samples

    merged_df = merged_df.drop(columns=["Unnamed: 0", "type"])

    # Print first 10 samples
    # print(merged_df.head(10))

    data = []
    for idx, row in merged_df.iterrows():
        data.append({
            "question": row["question"],
            "answer": row["answer"],
            "image": row["img_id"],
            "id": idx
        })

    # Split last 3000 samples as test set
    test_data = data[-3001:]
    train_data = data[:-3001]

    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")

    # Save train and test data to json files
    with open(os.path.join(input_dir, "train.json"), "w", encoding='utf-8') as f:
        json.dump(train_data, f, indent=4, ensure_ascii=False)
    with open(os.path.join(input_dir, "test.json"), "w", encoding='utf-8') as f:
        json.dump(test_data, f, indent=4, ensure_ascii=False)

def translate_vivqa_data(input_json, translator_class: str = "GoogleTranslator", max_workers=4):
    write_lock = threading.Lock()

    # Load train and test data
    with open(input_json, "r", encoding='utf-8') as f:
        data = json.load(f)
    
    # Initialize Google Translator
    if translator_class == "GoogleTranslator":
        translator = GoogleTranslator(source='vi', target='en')
    elif translator_class == "MyMemoryTranslator":
        translator = MyMemoryTranslator(source='vi-VN', target='en-US')

    output_json = input_json.replace(".json", "_translated.json")
    failed_json = input_json.replace(".json", "_failed.json")
    output_lines_json = input_json.replace(".json", "_translated_lines.jsonl")
        
    translated_data = []
    translated_ids = set()
    if os.path.exists(output_lines_json):
        with open(output_lines_json, "r", encoding='utf-8') as ojsonl:
            for line in ojsonl:
                item = json.loads(line)
                translated_data.append(item)
                translated_ids.add(item["id"])

    failed_translations = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(translate_one_vivqa_item, item, \
                                   output_lines_json, translated_ids, translated_data, failed_translations, write_lock, translator) \
                                    for item in data]
        
        for future in as_completed(futures):
            future.result()

    # Save failed translations to file
    if failed_translations:
        with open(failed_json, "w", encoding='utf-8') as failed_file:
            json.dump(failed_translations, failed_file, indent=4, ensure_ascii=False)

    num_org_annots = len(data)
    num_translated_annots = len(translated_ids)

    if num_org_annots == num_translated_annots:
        print(f"All {num_org_annots} annotations translated successfully.")
        # Merge original images with translated annotations
        saved_data = {
            "annotations": translated_data
        }
        # Save merged data to output file
        with open(output_json, "w", encoding='utf-8') as out_file:
            json.dump(saved_data, out_file, indent=1, ensure_ascii=False)
            print(f"Saved translated data to {output_json}")
    else:
        print(f"Translation incomplete: {num_translated_annots}/{num_org_annots} annotations translated.")
        # Save partially translated data to output file
        with open(output_json, "w", encoding='utf-8') as out_file:
            json.dump({"annotations": translated_data}, out_file, indent=4, ensure_ascii=False)
            print(f"Saved partially translated data to {output_json}")

def translate_one_vivqa_item(org_item, output_lines_json, translated_ids, translated_data, failed_translations, \
                            write_lock, translator):
    if org_item["id"] in translated_ids:
        print(f"Skipping already translated item with id {org_item['id']}")
        return
    try:
        if org_item["question"]:
            org_item["question"] = translator.translate(org_item["question"])

        for j in range(len(org_item["answer"])):
            org_item["answer"] = translator.translate(org_item["answer"])

        # Save to file after each item
        with write_lock:
            with open(output_lines_json, "a", encoding='utf-8') as output_jsonl:
                json.dump(org_item, output_jsonl, ensure_ascii=False)
                output_jsonl.write("\n")
                translated_ids.add(org_item["id"])
                translated_data.append(org_item)
                print(f"Translated item with id {org_item['id']}")

    except Exception as e:
        print(f"Error for id {org_item['id']}: {e}")
        failed_translations.append(org_item)

def translate_vivqa_submit_test(input_json, max_workers=4):
    write_lock = threading.Lock()

    # Load train and test data
    with open(input_json, "r", encoding='utf-8') as f:
        data = json.load(f)

    output_json = input_json.replace(".json", "_translated.json")
    failed_json = input_json.replace(".json", "_failed.json")
    output_lines_json = input_json.replace(".json", "_translated_lines.jsonl")
        
    translated_data = []
    translated_ids = set()
    if os.path.exists(output_lines_json):
        with open(output_lines_json, "r", encoding='utf-8') as ojsonl:
            for line in ojsonl:
                item = json.loads(line)
                translated_data.append(item)
                translated_ids.add(item["question_id"])

    failed_translations = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(translate_one_vivqa_test_item, item, \
                                   output_lines_json, translated_ids, translated_data, failed_translations, write_lock) \
                                    for item in data]
        
        for future in as_completed(futures):
            future.result()

    # Save failed translations to file
    if failed_translations:
        with open(failed_json, "w", encoding='utf-8') as failed_file:
            json.dump(failed_translations, failed_file, indent=4, ensure_ascii=False)

    num_org_annots = len(data)
    num_translated_annots = len(translated_ids)

    if num_org_annots == num_translated_annots:
        print(f"All {num_org_annots} test items translated successfully.")

        # Save merged data to output file
        with open(output_json, "w", encoding='utf-8') as out_file:
            json.dump(translated_data, out_file, indent=1, ensure_ascii=False)
            print(f"Saved translated test predictions to {output_json}")
    else:
        print(f"Translation incomplete: {num_translated_annots}/{num_org_annots} predictions translated.")
        # Save partially translated data to output file
        with open(output_json, "w", encoding='utf-8') as out_file:
            json.dump(translated_data, out_file, indent=4, ensure_ascii=False)
            print(f"Saved partially translated predictions to {output_json}")

def translate_one_vivqa_test_item(org_item, output_lines_json, translated_ids, translated_data, failed_translations, \
                            write_lock):
    if org_item["question_id"] in translated_ids:
        print(f"Skipping already translated item with id {org_item['question_id']}")
        return
    ggl_translator = GoogleTranslator(source='en', target='vi')
    # mm_translator = MyMemoryTranslator(source='en-US', target='vi-VN')
    try:
        for j in range(len(org_item["answer"])):
            # org_item["answer"] = ggl_translator.translate(org_item["answer"])
            org_item["answer"] = ggl_translator.translate(org_item["answer"])

        # Save to file after each item
        with write_lock:
            with open(output_lines_json, "a", encoding='utf-8') as output_jsonl:
                json.dump(org_item, output_jsonl, ensure_ascii=False)
                output_jsonl.write("\n")
                translated_ids.add(org_item["question_id"])
                translated_data.append(org_item)
                print(f"Translated item with id {org_item['question_id']}")

    except Exception as e:
        print(f"Error for id {org_item['question_id']}: {e}")
        failed_translations.append(org_item)

if __name__ == "__main__":
    # process_vivqa_data("data/vivqa/original")
    vivqa_split_files = [
        "/root/projects/exp1/data/vivqa/annotations/train.json",
        # "/root/projects/exp1/data/vivqa/annotations/test.json"
    ]

    viqua_test_preds = [
        # "/root/projects/exp1/submit_vivqa_test.json",
        "/root/projects/exp1/submit_vivqa_test copy.json"
    ]

    for json_path in vivqa_split_files:
        translate_vivqa_data(json_path, "MyMemoryTranslator", max_workers=4)