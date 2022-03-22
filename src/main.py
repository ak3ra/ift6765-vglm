import copy
import torch
import os
import torch
import logging

from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch import nn
from typing import Dict, List, Tuple
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from tqdm import tqdm, trange
from transformers import (
    AdamW,
    BertTokenizer,
    get_linear_schedule_with_warmup,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from data import CoLDataset
from model import CoLBertConfig, SimpleBertForMaskedLM_Vis, mask_tokens
import wandb
wandb.init(project="vglm")

print('test')
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

def main():


    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataset=CoLDataset('./vokenization/data/wiki103-cased/wiki.train.raw', 'bert-base-uncased', tokenizer, block_size=126)

    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)


    config = CoLBertConfig.from_pretrained('./vokenization/vlm/configs/bert-6L-512H.json', cache_dir='./test',voken_dim=1024)
    model = SimpleBertForMaskedLM_Vis(config=config,tokenizer=tokenizer)
    model.to(device)

    global_step = 0
    epochs_trained = 0
    num_train_epochs = 40
    mlm_probability =0.15
    warmup_steps = 10000
    gradient_accumulation_steps = 2
    t_total = num_train_epochs*gradient_accumulation_steps
    max_grad_norm = 1.0
    adam_epsilon = 1e-6
    output_path = "./output"
    shuffle = True
    mlm = True

    train_sampler = RandomSampler(
            train_dataset
        )

    train_dataloader=DataLoader(
        train_dataset, shuffle=False, num_workers=0,
        batch_size=64, pin_memory=True, sampler=train_sampler
    )

        # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                    lr=2e-4,
                    eps=adam_epsilon
                    )

    scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
        )

    model.zero_grad()
    train_iterator = trange(
            epochs_trained, int(num_train_epochs), desc="Epoch"
        )
    for epoch in train_iterator:
        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()       # Support of accumulating gradients
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")

        for step, batch in enumerate(epoch_iterator):

            inputs, labels = mask_tokens(batch, tokenizer, mlm_probability) if mlm_probability else (batch, batch)
            inputs = inputs.to(device)
            labels = labels.to(device)
            # If some of the input is padded, then the attention mask is needed
            attention_mask = (inputs != tokenizer.pad_token_id)         # word_tokens --> 1, pad_token --> 0
            if attention_mask.all():
                attention_mask = None

            model.train()
            outputs = model(inputs,
                            attention_mask=attention_mask,
                            masked_lm_labels=labels) if mlm_probability else model(inputs, labels=labels)
            loss = outputs  # model outputs are always tuple in transformers (see doc)

            loss.backward()
            optimizer.step()
            model.zero_grad()
            # logger.info(" Training loss : %0.4f" % (loss.item()))

            tr_loss += loss.item()
            print(loss.item())
            if (step + 1) % gradient_accumulation_steps == 0:
                if max_grad_norm > 0.:
                    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
        wandb.log({'training loss': loss.item()})



        checkpoint_name = "checkpoint-epoch%04d" % epoch
        save_model(checkpoint_name, model, tokenizer, optimizer, scheduler,output_path)
        evaluate(model,tokenizer,mlm)


def save_model(name, model, tokenizer, optimizer, scheduler,output_path):
    # Save model checkpoint
    output_dir = os.path.join(output_path, name)
    os.makedirs(output_dir, exist_ok=True)
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # torch.save(os.path.join(output_dir, "training_args.bin"))
    logger.info("Saving model checkpoint to %s", output_dir)

def evaluate(model, tokenizer,mlm=True, prefix="") -> Dict:

    eval_dataset=CoLDataset('./vokenization/data/wiki103-cased/wiki.valid.raw', 'bert-base-uncased', tokenizer, block_size=126)
    eval_batch_size = 32

    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size, collate_fn=collate
    )

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        inputs, labels = mask_tokens(batch, tokenizer) if mlm else (batch, batch)
        inputs = inputs.to(device)
        labels = labels.to(device)
        # If some of the input is padded, then the attention mask is needed
        attention_mask = (inputs != tokenizer.pad_token_id)  # word_tokens --> 1, pad_token --> 0
        if attention_mask.all():
            attention_mask = None

        with torch.no_grad():
            outputs = model(inputs, attention_mask=attention_mask, masked_lm_labels=labels) if mlm else model(inputs, labels=labels)
            lm_loss = outputs
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss)).item()

    result = {"perplexity": perplexity}

    wandb.log({'eval loss': eval_loss})
    wandb.log({'perplexity': perplexity})

    return result

if __name__ == '__main__':
    main()
