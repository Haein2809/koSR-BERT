import torch
from torch.utils.data import Dataset
import random

class NUGDataset(Dataset):
    def __init__(self, dataframe, tokenizer, text_col, max_len, mask_prob=0.15):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text_col = text_col
        self.max_len = max_len
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = str(self.data.iloc[index][self.text_col])

        encoding = self.tokenizer(
            text,
            padding='max_length',
            max_length=self.max_len,
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        labels = input_ids.clone()

        # 마스킹할 위치 선택
        rand = torch.rand(input_ids.shape)
        mask_arr = (rand < self.mask_prob) & (input_ids != self.tokenizer.pad_token_id) & (input_ids != self.tokenizer.cls_token_id) & (input_ids != self.tokenizer.sep_token_id)

        input_ids[mask_arr] = self.tokenizer.mask_token_id

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

class MURDataset(Dataset):
    def __init__(self, dataframe, tokenizer, text_col, max_len, mask_prob=0.15):
        self.tokenizer = tokenizer
        self.dataframe = dataframe
        self.text_col = text_col
        self.max_len = max_len
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        text = self.dataframe.iloc[idx][self.text_col]
        
        # 토큰화 및 인코딩
        encoding = self.tokenizer(
            text,
            padding='max_length',
            max_length=self.max_len,
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        # 토큰 마스킹
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, self.mask_prob)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # 마스킹되지 않은 토큰은 -100으로 설정
        input_ids[masked_indices] = self.tokenizer.mask_token_id

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'masked_indices': masked_indices
        }

class DORNDataset(Dataset):
    def __init__(self, df, session_col, text_col, order_col, tokenizer, max_length, num_labels):
        self.session_col = session_col
        self.text_col = text_col
        self.order_col = order_col
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_labels = num_labels

        # 세션을 정의하고 패딩/트렁케이팅 처리
        self.sessions = df.groupby(session_col)[text_col].apply(list).reset_index(name='session')
        self.sessions['session'] = self.sessions['session'].apply(self.pad_or_truncate)

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx):
        session = self.sessions.iloc[idx]['session']
        tokenized = self.tokenizer(
            session,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        input_ids = tokenized['input_ids']  # [num_labels, seq_length]
        attention_mask = tokenized['attention_mask']  # [num_labels, seq_length]
        labels = torch.tensor(range(len(session)))  # [num_labels]

        items = {
            'input_ids': input_ids.squeeze(0),  # squeeze to remove batch dimension
            'attention_mask': attention_mask.squeeze(0),  # squeeze to remove batch dimension
            'labels': labels
        }

        return items

    def pad_or_truncate(self, session):
        if len(session) > self.num_labels:
            return session[:self.num_labels]
        else:
            return session + [''] * (self.num_labels - len(session))
    
