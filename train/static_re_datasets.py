import json
from collections import Counter

with open("./nre_train.json", "r") as f:
    data = json.load(f)

total = len(data)
tokens_lens = [len(item.get("tokens", [])) for item in data]
labels_rel_count = [
    sum(1 for label in item.get("labels", []) if label == "REL")
    for item in data
]
avg_tokens_len = sum(tokens_lens) / total if total else 0
avg_rel = sum(labels_rel_count) / total if total else 0

print(f"总样本数: {total}")
print(f"tokens 平均长度: {avg_tokens_len:.2f}")
print(f"labels 中 REL 的平均个数: {avg_rel:.2f}")
print("tokens 长度分布:", Counter(tokens_lens)) 