from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import TrainingArguments,AutoModelForSequenceClassification,Trainer

task='mrpc'
checkpoint = "~/scratch/akera/vision_language/output/checkpoint-epoch0003/"
# checkpoint = './pre_trained_model'
train_batch_size = 128
eval_batch_size = 128


raw_datasets = load_dataset("glue",task)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

if task == 'sst2':
    def tokenize_function(example):
        return tokenizer(example["sentence"], truncation=True)

elif task == 'mrpc':
    def tokenize_function(example):
        return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

elif task == 'qqp':
    def tokenize_function(example):
        return tokenizer(example["question1"], example["question2"], truncation=True)
elif task == 'stsb':
    def tokenize_function(example):
        return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

import numpy as np
from datasets import load_metric

def compute_metrics(eval_preds):
    metric = load_metric("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments("test-trainer",
                per_device_train_batch_size = train_batch_size,
                per_device_eval_batch_size = eval_batch_size)


# trainer = Trainer(
#     model,
#     training_args,
#     train_dataset=tokenized_datasets["train"],
#     eval_dataset=tokenized_datasets["validation"],
#     data_collator=data_collator,
#     tokenizer=tokenizer,
# )

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
trainer.train()

