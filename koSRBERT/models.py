import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertForMaskedLM, BertForSequenceClassification, BertConfig
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

class NUGModel(torch.nn.Module):
    def __init__(self, model_name, num_labels):
        super(NUGModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.drop = torch.nn.Dropout(p=0.3)
        self.out = torch.nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_output = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        sequence_output = self.drop(sequence_output)
        logits = self.out(sequence_output)  # (batch_size, seq_len, num_labels)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            # 마스킹된 위치에 대한 로스만 계산
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, logits.size(-1))
            active_labels = torch.where(
                active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            loss = loss_fct(active_logits, active_labels)
        return loss, logits


class MURModel(nn.Module):
    def __init__(self, model_name):
        super(MURModel, self).__init__()
        self.bert = BertForMaskedLM.from_pretrained(model_name)
        self.loss_fct = nn.CrossEntropyLoss()  # Cross Entropy Loss 정의

    def forward(self, input_ids, attention_mask, masked_indices=None, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs.logits

        loss = None
        if labels is not None and masked_indices is not None:
            # 마스킹된 위치에 대해서만 손실 계산
            active_logits = logits[masked_indices]
            active_labels = labels[masked_indices]
            loss = self.loss_fct(active_logits.view(-1, self.bert.config.vocab_size), active_labels.view(-1))
        return loss, logits

class DORNModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super(DORNModel, self).__init__()
        self.config = BertConfig.from_pretrained(model_name, num_labels=num_labels)
        self.bert = BertForSequenceClassification.from_pretrained(model_name, config=self.config)
        self.loss_fct = nn.KLDivLoss(reduction='batchmean')  # KL Divergence Loss 정의

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        loss = None
        if labels is not None:
            # 로짓을 로그 확률로 변환
            log_probs = F.log_softmax(logits, dim=-1)
            # 라벨을 원-핫 인코딩으로 변환
            true_probs = F.one_hot(labels, num_classes=self.config.num_labels).float()
            # true_probs의 크기를 log_probs와 일치시키기 위해 view 사용
            true_probs = true_probs.view_as(log_probs)
            # KLDiv 손실 계산
            loss = self.loss_fct(log_probs, true_probs)

        return loss, logits

class koSRBERT(nn.Module):
    def __init__(self, model_name, nug, mur, dorn, hidden_size, num_layers, num_attention_heads, role_vocab_size=2):
        super(koSRBERT, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.role_embeddings = nn.Embedding(role_vocab_size, self.bert.config.hidden_size)
        self.nug = nug
        self.mur = mur
        self.dorn = dorn
        self.context_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.bert.config.hidden_size, nhead=num_attention_heads),
            num_layers=num_layers
        )
        self.final_classifier = nn.Linear(hidden_size * 3, 2)  # 이진 분류를 위해 출력 크기 2로 설정

    def forward(self, input_ids, attention_mask, role_ids, orders, masked_indices=None, order_labels=None, target_ids=None):
        batch_size, num_orders, seq_len = input_ids.size()
        input_ids = input_ids.view(batch_size * num_orders, seq_len)
        attention_mask = attention_mask.view(batch_size * num_orders, seq_len)
        role_ids = role_ids.view(batch_size * num_orders)
        
        if target_ids is not None:
            target_ids = target_ids.view(batch_size * num_orders, seq_len)

        if order_labels is not None:
            order_labels = order_labels.view(batch_size * num_orders)
        
        token_embeddings = self.bert.embeddings(input_ids)
        role_embeddings = self.role_embeddings(role_ids).unsqueeze(1).expand_as(token_embeddings)
        combined_embeddings = token_embeddings + role_embeddings
        
        outputs = self.bert(inputs_embeds=combined_embeddings, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state.view(batch_size, num_orders, seq_len, -1)[:, :, 0, :]

        # NUG loss and logits
        nug_loss, nug_logits = self.nug(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids)

        # MUR loss and logits
        mur_loss, mur_logits = self.mur(input_ids=input_ids, attention_mask=attention_mask, masked_indices=masked_indices, labels=target_ids)
        
        # DORN loss and logits
        dorn_loss, dorn_logits = self.dorn(input_ids=input_ids, attention_mask=attention_mask, labels=order_labels)

        nug_cls_logits = nug_logits.view(batch_size, num_orders, -1)[:, 0, :]
        mur_cls_logits = mur_logits.view(batch_size, num_orders, -1)[:, 0, :]
        dorn_cls_logits = dorn_logits.view(batch_size, num_orders, -1)[:, 0, :]

        combined_outputs = torch.cat((nug_cls_logits, mur_cls_logits, dorn_cls_logits), dim=2)
        
        context_output = self.context_encoder(combined_outputs)
        
        final_logits = self.final_classifier(context_output)
        final_probs = F.softmax(final_logits, dim=-1)  # 소프트맥스 적용

        return final_probs, nug_loss, mur_loss, dorn_loss

    def compute_loss(self, logits, sr_labels, orders, nug_loss, mur_loss, dorn_loss):
        loss_fct = nn.CrossEntropyLoss()
        logits = logits.view(-1, 2)  # 이진 분류를 위해 2로 설정
        sr_labels = sr_labels.view(-1)
        
        # CrossEntropyLoss 적용
        loss = loss_fct(logits, sr_labels)

        # order 값이 -1인 경우 무시하고 손실 계산
        active_loss = orders.view(-1) != -1
        active_nug_loss = nug_loss.view(-1)[active_loss]
        active_mur_loss = mur_loss.view(-1)[active_loss]
        active_dorn_loss = dorn_loss.view(-1)[active_loss]

        total_loss = loss + active_nug_loss.mean() + active_mur_loss.mean() + active_dorn_loss.mean()
        return total_loss
