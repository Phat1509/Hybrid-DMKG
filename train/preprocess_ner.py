import json
from datasets import Dataset
from transformers import DistilBertTokenizerFast

# 1. Đọc file NER
with open('../train/ner_train.json', 'r') as f:
    data = json.load(f)

dataset = Dataset.from_dict({
    "tokens": [item["tokens"] for item in data],
    "labels": [item["labels"] for item in data]
})

dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_valid = dataset['train'].train_test_split(test_size=0.1, seed=42)
dataset['train'] = train_valid['train']
dataset['dev'] = train_valid['test']

# 2. Định nghĩa nhãn ENT
label_list = ["O", "ENT"]
label2id = {l: i for i, l in enumerate(label_list)}

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')

max_length = 500  

def preprocess_function(example):
    tokenized_input = tokenizer(
        example["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding='max_length', 
        max_length=max_length 
    )
    word_ids = tokenized_input.word_ids()
    label_ids = []
    previous_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)
        elif word_idx != previous_word_idx:
            label_ids.append(label2id[example["labels"][word_idx]])
        else:
            label_ids.append(-100)
        previous_word_idx = word_idx
    
    label_ids += [-100] * (len(tokenized_input["input_ids"]) - len(label_ids))
    label_ids = label_ids[:len(tokenized_input["input_ids"])]
    tokenized_input["labels"] = label_ids
    return tokenized_input

encoded_dataset = dataset.map(preprocess_function, batched=False)

# 3. Lưu dữ liệu
encoded_dataset.save_to_disk('./encoded_dataset')
print("Xử lý dữ liệu NER thành công! Đã lưu tại ./encoded_dataset")
