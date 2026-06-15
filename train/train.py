%%writefile train.py
import json
from datasets import Dataset, load_from_disk
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification, Trainer, TrainingArguments
from seqeval.metrics import f1_score
import torch

label_list = ["O", "ENT"]
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for l, i in label2id.items()}

# 加载预处理后的数据
encoded_dataset = load_from_disk('./encoded_dataset')

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
    output_dir='./results_entity_judge',
    eval_strategy='epoch',
    learning_rate=2e-5,
    save_strategy='epoch',
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_steps=200,        
    save_steps=500,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset['train'],
    eval_dataset=encoded_dataset['test'],
    processing_class=tokenizer,    # Đã sửa lỗi ở đây!
    compute_metrics=compute_metrics
)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please run on a machine with GPU.")

num_gpus = torch.cuda.device_count()
print(f"Detected {num_gpus} GPU(s).")

trainer.train()
results = trainer.evaluate()
print(results)

model.save_pretrained('./results_entity_judge/best_model_entity_judge')
tokenizer.save_pretrained('./results_entity_judge/best_model_entity_judge')