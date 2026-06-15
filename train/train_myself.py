%%writefile train_myself.py
import json
from datasets import Dataset, load_from_disk
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification, Trainer, TrainingArguments
from seqeval.metrics import f1_score
import torch

label_list = ["O", "REL"]
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for l, i in label2id.items()}

# 加载预处理后的数据
encoded_dataset = load_from_disk('./RE_dataset')

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')

model = DistilBertForTokenClassification.from_pretrained(
    'distilbert-base-cased',
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)


def compute_metrics(p):
    predictions, labels = p
    predictions = predictions.argmax(axis=-1)
    true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return {"f1": f1_score(true_labels, true_predictions)}


training_args = TrainingArguments(
    output_dir='./results_relation/',
    eval_strategy='epoch',
    learning_rate=2e-5,
    save_strategy='epoch',
    per_device_train_batch_size=16,   # Giữ ở mức 16 để tránh nghẽn mạch GPU song song
    per_device_eval_batch_size=16,    # Giữ ở mức 16
    num_train_epochs=10,
    weight_decay=0.01,
    logging_steps=10,
    save_steps=500,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    fp16=False
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset['train'],
    eval_dataset=encoded_dataset['dev'],  
    processing_class=tokenizer,
    compute_metrics=compute_metrics
)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please run on a machine with GPU.")

device = torch.device("cuda")
print("Using device:", device)
model.to(device)

trainer.train()
dev_results = trainer.evaluate()  

# 最终在test集上评估
test_results = trainer.evaluate(encoded_dataset['test'])

# 打印test集预测结果
predictions_output = trainer.predict(encoded_dataset['test'])
predictions = predictions_output.predictions.argmax(axis=-1)
labels = predictions_output.label_ids

# 收集所有预测结果
all_results = []
test_dataset = encoded_dataset['test']
for i, (pred, label) in enumerate(zip(predictions, labels)):
    pred_labels = [id2label[p] for (p, l) in zip(pred, label) if l != -100]
    true_labels = [id2label[l] for l in label if l != -100]
    raw_text = test_dataset[i]['tokens'] if 'tokens' in test_dataset.features else test_dataset[i].get('text', '')
    print(f"Sample {i}:")
    print("  Raw:  ", raw_text)
    print("  True: ", true_labels)
    print("  Pred: ", pred_labels)
    all_results.append({
        "sample_id": i,
        "raw_text": raw_text,
        "true_labels": true_labels,
        "pred_labels": pred_labels
    })
print("Dev集评估结果:", dev_results)
print("Test集评估结果:", test_results)

# 保存为 JSON 文件
with open('./results_relation/prediction_results.json', 'w', encoding='utf-8') as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)
model.save_pretrained('./results_relation/my_best_model_relation_judge')
tokenizer.save_pretrained('./results_relation/my_best_model_relation_judge')