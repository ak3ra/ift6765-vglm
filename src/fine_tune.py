from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import TrainingArguments,AutoModelForSequenceClassification,Trainer

task='sst2'
checkpoint = "bert-base-uncased"
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


training_args = TrainingArguments("test-trainer",
                per_device_train_batch_size = train_batch_size,
                per_device_eval_batch_size = eval_batch_size)




trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()


python ../run_glue.py \\n  --model_name_or_path bert-base-cased \\n  --task_name stsb \\n  --do_train \\n  --do_eval \\n  --max_seq_length 512 \\n  --per_device_train_batch_size 16 \\n  --learning_rate 2e-5 \\n  --num_train_epochs 3 \\n  --output_dir bert-base-cased-finetuned-stsb \\n  --push_to_hub \\n  --hub_strategy all_checkpoints \\n  --logging_strategy epoch \\n  --save_strategy epoch \\n  --evaluation_strategy epoch \\n```
