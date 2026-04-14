import json
from datasets import Dataset
from transformers import DistilBertTokenizerFast

# 加载原始数据
with open('../train/nre_train.json', 'r') as f:
    data = json.load(f)

dataset = Dataset.from_dict({
    "tokens": [item["tokens"] for item in data],
    "labels": [item["labels"] for item in data]
})

# 先划分 test
dataset = dataset.train_test_split(test_size=0.2, seed=42)
# 再从 train 中划分 dev
train_valid = dataset['train'].train_test_split(test_size=0.1, seed=42)
dataset['train'] = train_valid['train']
dataset['dev'] = train_valid['test']

label_list = ["O", "REL"]
label2id = {l: i for i, l in enumerate(label_list)}

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')

max_length = 500  # 设置最大长度，建议与模型输入保持一致

def preprocess_function(example):
    tokenized_input = tokenizer(
        example["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding='max_length',  # 使用max_length方式padding
        max_length=max_length  # 统一为500
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
    # 对label_ids做padding到max_length（与input_ids长度一致）
    label_ids += [-100] * (len(tokenized_input["input_ids"]) - len(label_ids))
    label_ids = label_ids[:len(tokenized_input["input_ids"])]
    tokenized_input["labels"] = label_ids
    return tokenized_input

encoded_dataset = dataset.map(preprocess_function, batched=False)

# 保存预处理后的数据集
encoded_dataset.save_to_disk('./RE_dataset')
print("数据预处理完成并已保存到 ./RE_dataset")