import json
from pathlib import Path
from transformers import AutoTokenizer

# --- Prerequisites ---
# Ensure you have a 'glossary.py' file with the 'segment_normalize' function
# in the same directory as this script.
try:
    from glossary import segment_normalize
except ImportError:
    print("ERROR: Could not find 'glossary.py'.")
    print("Please ensure the file containing the 'segment_normalize' function is in the same directory.")
    # Provide a simple placeholder if the file is missing, for demonstration
    def segment_normalize(text):
        return text.lower().strip()

# --- 1. CONFIGURE YOUR FILE PATHS HERE ---
# Adjust these paths to match the files you want to check.

# Base directory for the ViVQA dataset
BASE_DIR = Path("./data/vivqa")

# The original, full JSON annotation file (e.g., for the train split)
ORIGINAL_JSON_PATH = BASE_DIR / "vqa" / "train_vi_full.json"
# ORIGINAL_JSON_PATH = BASE_DIR / "vqa" / "test_vi.json"

# The .jsonl file that your script generated
GENERATED_JSONL_PATH = BASE_DIR / "vivqa_vi.train.jsonl"
# GENERATED_JSONL_PATH = BASE_DIR / "vivqa_vi.val.jsonl"
# GENERATED_JSONL_PATH = BASE_DIR / "vivqa_vi.trainable_val.jsonl"
# GENERATED_JSONL_PATH = BASE_DIR / "vivqa_vi.test.jsonl"

# The answer-to-label mapping file
ANS2LABEL_PATH = BASE_DIR / "answer2label.txt"

# The Hugging Face name for the tokenizer used to create the text_segment
TOKENIZER_NAME = "vinai/phobert-base-v2"


def load_original_data_map(path: Path) -> dict:
    """Loads the original JSON file and maps it by question ID for fast lookups."""
    print(f"Loading original annotations from: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert the list of questions into a dictionary keyed by 'id' (qid)
    return {item['id']: item for item in data}

def load_label_to_answer_map(path: Path) -> dict:
    """Loads the answer file and creates a label -> answer mapping."""
    print(f"Loading answer map from: {path}")
    label_map = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            label_map[item['label']] = item['answer']
    return label_map

def verify_files():
    """Main function to run the verification process."""
    # --- 2. LOAD ALL DATA ---
    try:
        original_data = load_original_data_map(ORIGINAL_JSON_PATH)
        label2ans = load_label_to_answer_map(ANS2LABEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    except FileNotFoundError as e:
        print(f"\nERROR: Could not find a required file. Please check your paths.")
        print(e)
        return
        
    print("\n--- STARTING VERIFICATION ---")
    print(f"Checking '{GENERATED_JSONL_PATH.name}' against '{ORIGINAL_JSON_PATH.name}'...")

    matches = 0
    mismatches = 0
    items_checked = 0

    # --- 3. ITERATE AND VERIFY ---
    with open(GENERATED_JSONL_PATH, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            generated_item = json.loads(line)
            qid = generated_item['qid']
            items_checked += 1

            # Find the corresponding original item
            original_item = original_data.get(qid)
            if not original_item:
                print(f"  [Warning] QID {qid} from line {line_num} not found in the original JSON file. Skipping.")
                continue

            # --- Verification for items with labels (e.g., train/val splits) ---
            if generated_item['labels']:
                # --- Verification for TRAIN/VAL items (check answer and question) ---
                generated_label = generated_item['labels'][0]
                generated_answer = label2ans.get(generated_label, "---LABEL NOT FOUND IN MAP---")
                original_answer_normalized = segment_normalize(original_item['answer'])
                
                decoded_question = tokenizer.decode(generated_item['text_segment'], skip_special_tokens=True)
                original_question_simple = original_item['question'].replace(" ", "")
                decoded_question_simple = decoded_question.replace(" ", "")

                if generated_answer == original_answer_normalized and original_question_simple == decoded_question_simple:
                    matches += 1
                else:
                    mismatches += 1
                    print(f"\n❌ MISMATCH FOUND on QID: {qid} (line {line_num})")
                    if generated_answer != original_answer_normalized:
                        print(f"  - Answer Mismatch:")
                        print(f"    Original (normalized): '{original_answer_normalized}'")
                        print(f"    Generated (from label {generated_label}): '{generated_answer}'")
                    if original_question_simple != decoded_question_simple:
                        print(f"  - Question Mismatch:")
                        print(f"    Original: '{original_item['question']}'")
                        print(f"    Decoded:  '{decoded_question}'")
            else:
                # --- Verification for TEST items (check question ONLY) ---
                decoded_question = tokenizer.decode(generated_item['text_segment'], skip_special_tokens=True)
                original_question_simple = original_item['question'].replace(" ", "")
                decoded_question_simple = decoded_question.replace(" ", "")

                if original_question_simple == decoded_question_simple:
                    matches += 1
                else:
                    mismatches += 1
                    print(f"\n❌ MISMATCH FOUND on QID: {qid} (line {line_num})")
                    print(f"  - Reason: Question Mismatch")
                    print(f"    Original: '{original_item['question']}'")
                    print(f"    Decoded:  '{decoded_question}'")
    
    # --- 4. PRINT SUMMARY ---
    print("\n--- ✅ VERIFICATION COMPLETE ---")
    print(f"Total Items Checked: {items_checked}")
    print(f"Correct Matches: {matches}")
    print(f"Mismatches: {mismatches}")
    if mismatches == 0:
        print("\n🎉 Success! All entries in the generated file match the original source.")
    else:
        print("\nFound errors. Please review the mismatch details above.")

if __name__ == "__main__":
    verify_files()