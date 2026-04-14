import json
import re

def align_labels(tokens, entity):
    labels = ["O"] * len(tokens)
    entity_tokens = entity.split()
    for i in range(len(tokens) - len(entity_tokens) + 1):
        if tokens[i:i+len(entity_tokens)] == entity_tokens:
            for j in range(len(entity_tokens)):
                labels[i+j] = "ENT"
            break
    return labels

with open('../train/final_train_data.json', 'r') as f:
    data = json.load(f)

ner_data = []
for item in data:
    if "question" not in item or "entity_labels" not in item:
        continue
    question = item["question"]
    entity = item["entity_labels"]
    # 如果结尾有问号，将问号前加空格
    question = re.sub(r'(\w)(\?)$', r'\1 ?', question)
    tokens = question.split()
    labels = align_labels(tokens, entity)
    ner_data.append({"tokens": tokens, "labels": labels})

with open('../train/ner_train.json', 'w') as f:
    json.dump(ner_data, f, ensure_ascii=False, indent=2)