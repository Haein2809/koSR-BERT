
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
    def __init__(self, dataframe, tokenizer, text_col, max_len, masking_prob=0.2):
        self.tokenizer = tokenizer
        self.dataframe = dataframe
        self.text_col = text_col
        self.max_len = max_len
        self.masking_prob = masking_prob

        # [MASK] 토큰 ID를 설정
        self.mask_token_id = self.tokenizer.convert_tokens_to_ids('[MASK]')

        # 세션별로 데이터프레임을 그룹화하여 마스킹할 인덱스 선택
        self.masked_indices = self._select_masked_indices()

    def _select_masked_indices(self):
        masked_indices = set()
        grouped = self.dataframe.groupby('id')

        for session_id, group in grouped:
            total_messages = len(group)
            mask_count = max(1, round(total_messages * self.masking_prob))
            mask_indices = random.sample(list(group.index), mask_count)
            masked_indices.update(mask_indices)

        return masked_indices

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        text = row[self.text_col]

        encoding = self.tokenizer(
            text,
            padding='max_length',
            max_length=self.max_len,
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze()
        input_ids_nomask = input_ids.clone()
        attention_mask = encoding['attention_mask'].squeeze()
        target_ids = input_ids.clone()

        # 마스킹 작업
        if idx in self.masked_indices:
            for i in range(len(input_ids)):
                if input_ids[i] not in [
                    self.tokenizer.cls_token_id,
                    self.tokenizer.sep_token_id,
                    self.tokenizer.pad_token_id
                ]:
                    rand = random.random()
                    if rand < 0.8:
                        input_ids[i] = self.mask_token_id
                    elif rand < 0.9:
                        input_ids[i] = random.randint(0, self.tokenizer.vocab_size - 1)
                else:
                    target_ids[i] = -100
        else:
            target_ids[:] = -100

        return {
            'input_ids_nomask': input_ids_nomask,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': target_ids
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
    