import copy
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm, trange
from transformers import (
    AdamW,
    BertTokenizer,
)

from data import CoLDataset
from model import CoLBertConfig, SimpleBertForMaskedLM_Vis, mask_tokens
import wandb 
wandb.init(project="vglm")


def main():

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset=CoLDataset('./vokenization/data/wiki103-cased/wiki.test.raw', 'bert-base-uncased', tokenizer, block_size=512)
    # train_dataloader = DataLoader(
    #         dataset,batch_size=2
    #     )

    train_dataloader=DataLoader(
        dataset, shuffle=False, num_workers=0,
        batch_size=2, pin_memory=True
    )

    config = CoLBertConfig.from_pretrained('./vokenization/vlm/configs/bert-6L-512H.json', cache_dir='./test',voken_dim=1024)
    model = SimpleBertForMaskedLM_Vis(config=config,tokenizer=tokenizer)
    global_step = 0
    epochs_trained = 0
    num_train_epochs = 1
    mlm_probability =0.15
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
            # inputs = inputs.to(args.device)
            # labels = labels.to(args.device)
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
            wandb.log({'training loss': loss.item()})

            tr_loss += loss.item()
            print(loss.item())
            # if (step + 1) % args.gradient_accumulation_steps == 0:
            #     if args.max_grad_norm > 0.:
            #         if args.fp16:
            #             total_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            #         else:
            #             total_norm =torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            #     optimizer.step()
            #     scheduler.step()  # Update learning rate schedule
            #     model.zero_grad()
            #     global_step += 1

    torch.save(model.state_dict(), 'trained_model.pth')
if __name__ == '__main__':
    main()
