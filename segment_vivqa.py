import py_vncorenlp
import json
import os


def segment_vivqa_item(item, segmenter):
    # Segment the question
    item['question'] = segmenter.word_segment(item['question'])
    
    # Segment the context
    item['answer'] = segmenter.word_segment(item['answer'])

    return data

# Automatically download VnCoreNLP components from the original repository
# and save them in some local machine folder
py_vncorenlp.download_model(save_dir='/root/projects/exp1/vncorenlp')

# Load the word and sentence segmentation component
rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir='/root/projects/exp1/vncorenlp')

text = "Ông Nguyễn Khắc Chúc  đang làm việc tại Đại học Quốc gia Hà Nội. Bà Lan, vợ ông Chúc, cũng làm việc tại đây."

output = rdrsegmenter.word_segment(text)

# Test the output
print("Word Segmentation Output:")
print(output)

vivqa_path = '/root/projects/exp1/data/vivqa'
vivqa_data_path = os.path.join(vivqa_path, 'annotations')

# Load the JSON files
train_file = os.path.join(vivqa_data_path, 'train.json')
dev_file = os.path.join(vivqa_data_path, 'test.json')

with open(train_file, 'r', encoding='utf-8') as f:
    train_data = json.load(f)
with open(dev_file, 'r', encoding='utf-8') as f:
    dev_data = json.load(f)

print("Train Data Sample:")
print(train_data[0])
print("Dev Data Sample:")
print(dev_data[0])

for file in [train_file, dev_file]:
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for item in data:
        item = segment_vivqa_item(item, rdrsegmenter)

    # Wrap data inside "annotations"
    data = {
        "annotations": data
    }
    
    # Save the segmented data
    with open(file.replace('.json', '_segmented.json'), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


