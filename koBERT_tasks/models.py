
import torch
import torch.nn as nn
from transformers import BertModel, BertForMaskedLM, BertForSequenceClassification, BertConfig
from kobert_tokenizer import KoBERTTokenizer
import sentencepiece as spm

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

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        logits = outputs.logits
        return loss, logits

class DORNModel(torch.nn.Module):
    def __init__(self, model_name, num_labels):
        super(DORNModel, self).__init__()
        self.config = BertConfig.from_pretrained(model_name, num_labels=num_labels)
        self.bert = BertForSequenceClassification.from_pretrained(model_name, config=self.config)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        return logits

class EmbeddingEnhancedModel(nn.Module):
    def __init__(self, model_name, tf_idf_words, bert_class):
        super(EmbeddingEnhancedModel, self).__init__()
        self.bert = bert_class.from_pretrained(model_name)
        self.tokenizer = KoBERTTokenizer.from_pretrained(model_name)
        self.tf_idf_words = tf_idf_words
        self.update_sentencepiece_model()

        # 기존 임베딩 가중치 로드
        old_embeddings = self.bert.get_input_embeddings().weight.data

        # 추가 서브워드를 포함하는 새로운 임베딩 층 생성
        new_vocab_size = old_embeddings.size(0) + len(tf_idf_words)
        new_embedding_dim = old_embeddings.size(1)

        # 기존 임베딩으로 새로운 임베딩 층 초기화
        new_embeddings = torch.nn.Embedding(new_vocab_size, new_embedding_dim)
        new_embeddings.weight.data[:old_embeddings.size(0)] = old_embeddings

        # 새로운 단어 임베딩 초기화 (랜덤 가중치 부여)
        new_word_embeddings = torch.nn.init.normal_(torch.empty(len(tf_idf_words), new_embedding_dim))
        new_embeddings.weight.data[old_embeddings.size(0):] = new_word_embeddings

        # 기존 임베딩 층을 새로운 임베딩 층으로 교체
        self.bert.set_input_embeddings(new_embeddings)

        # 토크나이저 업데이트
        self.tokenizer.add_tokens(tf_idf_words)
        self.bert.resize_token_embeddings(len(self.tokenizer))

    def update_sentencepiece_model(self):
        # SentencePiece 모델 로드
        sp_model_path = self.tokenizer.pretrained_vocab_files_map['vocab_file']['skt/kobert-base-v1']
        sp = spm.SentencePieceProcessor()
        sp.Load(sp_model_path)

        # 새로운 서브워드 추가
        new_vocab_size = len(sp) + len(self.tf_idf_words)
        with open('tf_idf_words.txt', 'w') as f:
            for word in self.tf_idf_words:
                f.write(f"{word}\n")

        spm.SentencePieceTrainer.Train(
            f'--input=tf_idf_words.txt --model_prefix=new_vocab --vocab_size={new_vocab_size} --user_defined_symbols={",".join(self.tf_idf_words)}'
        )

        # 새롭게 생성된 SentencePiece 모델 로드
        new_sp = spm.SentencePieceProcessor()
        new_sp.Load('new_vocab.model')
        self.tokenizer.sp = new_sp

    def forward(self, input_ids, attention_mask, labels=None):
        raise NotImplementedError("Each subclass must implement this method.")

class NUGEModel(EmbeddingEnhancedModel):
    def __init__(self, model_name, num_labels, tf_idf_words):
        super(NUGEModel, self).__init__(model_name, tf_idf_words, BertModel)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, num_labels)

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
            loss_fct = nn.CrossEntropyLoss()
            # 마스킹된 위치에 대한 로스만 계산
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, logits.size(-1))
            active_labels = torch.where(
                active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            loss = loss_fct(active_logits, active_labels)
        return loss, logits

class MUREModel(EmbeddingEnhancedModel):
    def __init__(self, model_name, tf_idf_words):
        super(MUREModel, self).__init__(model_name, tf_idf_words, BertForMaskedLM)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        logits = outputs.logits
        return loss, logits

class DORNEModel(EmbeddingEnhancedModel):
    def __init__(self, model_name, num_labels, tf_idf_words):
        super(DORNEModel, self).__init__(model_name, tf_idf_words, BertForSequenceClassification)
        self.config = BertConfig.from_pretrained(model_name, num_labels=num_labels)
        self.bert = BertForSequenceClassification.from_pretrained(model_name, config=self.config)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        loss = outputs.loss if labels is not None else None
        return loss, logits
    
