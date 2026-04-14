import json
with open('../train/ner_train.json') as f:
    data = json.load(f)
print(type(data))  # 应该是 list
for i, item in enumerate(data):
    if not isinstance(item, dict):
        print(f"Row {i} is not a dict:", item)
    if "tokens" not in item or "labels" not in item:
        print(f"Row {i} missing keys:", item)