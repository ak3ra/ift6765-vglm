import copy
import os
import random
import logging
import h5py
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
import torch.nn.functional as F

from tqdm import tqdm, trange

from transformers import (
    AutoConfig,
    AutoModelWithLMHead,
    AutoModel,
    AutoModelForMaskedLM,
    BertConfig,
    BertForMaskedLM,

)


from torch.nn import CrossEntropyLoss
BertLayerNorm = torch.nn.LayerNorm


def mask_tokens(inputs, tokenizer,mlm_probability=0.15):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


class CoLBertConfig(BertConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.voken_size = None
        self.num_class = None
        self.voken_dim = None
        self.do_voken_cls = False
        self.do_voken_reg = False
        # self.do_voken_ctr = False
        self.shared_head = False
        self.verbose = False


def getVisFeature(input_ids):
  batch_size = input_ids.shape[0]
  block_size = input_ids.shape[1]
  return torch.ones(batch_size,block_size,1024)

def getVisLabels(input_ids):
   batch_size = input_ids.shape[0]
   block_size = input_ids.shape[1]
   return torch.ones(batch_size,block_size,dtype=torch.long)


class BertVLMRegressionHead(nn.Module):
    """Bert Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = torch.nn.Linear(config.hidden_size, config.voken_dim, bias=True)

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = F.gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x

class BertVLMClassificationHead(nn.Module):
    """Bert Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.num_class, bias=True)

        if config.verbose:
            print(f"Visual Object Classification Head: Build model with num_class {config.num_class}")

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = F.gelu(x)
        x = self.layer_norm(x)
        x = self.decoder(x)

        return x


class SimpleBertForMaskedLM_Vis(BertForMaskedLM):

    def __init__(self, config,tokenizer):
        super().__init__(config)
        self.tokenizer = tokenizer
        self.visual_reg_head = BertVLMRegressionHead(config)
        #self.visual_cls_head = BertVLMClassificationHead(config)
        self.visual_cls_loss_fct = nn.CrossEntropyLoss()
        self.voken_reg_loss_fct = nn.SmoothL1Loss(reduction='none')

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            masked_lm_labels=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            lm_labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        sequence_output = outputs[0] # token embeddings

        # visual_features = getVisFeature(input_ids)
        # voken_predictions = self.visual_reg_head(sequence_output)
        # voken_reg_loss = self.voken_reg_loss_fct(voken_predictions, visual_features)
        # voken_reg_loss=voken_reg_loss.sum(-1).mean()

        # obj_prediction_scores = self.visual_cls_head(sequence_output)
        # obj_labels = getVisLabels(input_ids)
        # visual_cls_loss = self.visual_cls_loss_fct(obj_prediction_scores.view(-1,10),obj_labels.view(-1))

        prediction_scores = self.cls(sequence_output)
        loss_fct = CrossEntropyLoss()
        token_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))

        print(token_loss)
        return token_loss



class BertForMaskedVisLan(nn.Module):

    def __init__(self, model_checkpoint,config,tokenizer):
        super(BertForMaskedVisLan,self).__init__()
        self.bert = AutoModel.from_pre_trained(model_checkpoint)
        self.tokenizer = tokenizer

        self.do_voken_cls = config.do_voken_cls
        self.do_voken_reg = config.do_voken_reg

        self.token_cls_loss_fct = CrossEntropyLoss()


        if config.do_voken_reg:
            self.visual_reg_head = BertVLMRegressionHead(config)
            self.voken_reg_loss_fct = nn.SmoothL1Loss(reduction='none')
        
        if config.do_voken_cls:
            self.visual_cls_head = BertVLMClassificationHead(config)
            self.visual_cls_loss_fct = nn.CrossEntropyLoss()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            masked_lm_labels=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            lm_labels=None,
            voken_labels=None,
            voken_features=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        sequence_output = outputs[0] # token embeddings

        voken_loss = 0.0
        if self.do_voken_cls:
            assert voken_labels is not None
            voken_scores = self.visual_cls_head(sequence_output)
            voken_cls_loss = self.voken_cls_loss_fct(voken_scores.view(-1, self.config.voken_size), voken_labels.view(-1))
            voken_loss += voken_cls_loss

        if self.do_voken_reg:
            assert voken_features is not None
            assert voken_labels is not None

            # visual_features = getVisFeature(input_ids)
            voken_predictions = self.visual_reg_head(sequence_output)
            voken_reg_loss = self.voken_reg_loss_fct(voken_predictions, voken_features)
            voken_reg_loss=voken_reg_loss.sum(-1).mean()
            # voken_prediction = self.visual_reg_head(sequence_output)

            # # Get the mask and pre-trained features
            # voken_label_mask = (voken_labels != -100)               # Get a mask of [0, 1, 1, ...., 1, 0], [b, len]
            # safe_voken_labels = voken_labels.clone()
            # safe_voken_labels[~voken_label_mask] = 0
            # voken_feats = self.voken_feat_emb(safe_voken_labels)         # [b, len] --> [b, len, f]

            # # Loss
            # voken_reg_loss = self.voken_reg_loss_fct(voken_prediction, voken_feats)   # [b, len, f]

            # # [b, l, f] * ([b,l] --> [b, l, 1]) = [b, l, f]
            # voken_reg_loss = (voken_reg_loss * voken_label_mask.float().unsqueeze(-1))

            # # [b, l, f] --sum-> [b, l] --mean-> [1,]
            # voken_reg_loss = voken_reg_loss.sum(-1).mean()

            voken_loss += voken_reg_loss

        prediction_scores = self.cls(sequence_output)
        loss_fct = CrossEntropyLoss()
        token_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))

        # print(token_loss)
        return token_loss,voken_loss
