import torch
from torch.utils.data import Dataset
import pandas as pd

class koSRBERTDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, max_order, text_col, role_col, order_col, session_col, class_col, mask_prob=0.15):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_order = max_order
        self.text_col = text_col
        self.role_col = role_col
        self.order_col = order_col
        self.session_col = session_col
        self.class_col = class_col
        self.mask_prob = mask_prob

        self.sessions = dataframe.groupby(self.session_col).apply(lambda x: {
            'texts': x[self.text_col].tolist(),
            'orders': x[self.order_col].tolist(),
            'sr_labels': x[self.class_col].tolist(),
            'role_ids': x[self.role_col].tolist()
        }).tolist()

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, index):
        session = self.sessions[index]
        texts = session['texts']
        orders = session['orders']
        sr_labels = session['sr_labels']
        role_ids = session['role_ids']

        if len(texts) > self.max_order:
            texts = texts[:self.max_order]
            orders = orders[:self.max_order]
            sr_labels = sr_labels[:self.max_order]
            role_ids = role_ids[:self.max_order]
        else:
            texts += [""] * (self.max_order - len(texts))
            orders += [-1] * (self.max_order - len(orders))
            sr_labels += [0] * (self.max_order - len(sr_labels))
            role_ids += [0] * (self.max_order - len(role_ids))

        inputs = self.tokenizer(
            texts,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        # target_ids 생성
        target_ids = input_ids.clone()
        target_ids[target_ids == self.tokenizer.pad_token_id] = -100

        # Bernoulli 분포를 사용하여 마스킹할 위치 결정
        bernoulli_dist = torch.distributions.Bernoulli(self.mask_prob)
        mask = (bernoulli_dist.sample(input_ids.shape).bool()) & (input_ids != self.tokenizer.pad_token_id)
        input_ids[mask] = self.tokenizer.mask_token_id

        # masked_indices 생성
        masked_indices = mask

        orders = torch.tensor(orders, dtype=torch.long)
        sr_labels = torch.tensor(sr_labels, dtype=torch.float)
        role_ids = torch.tensor(role_ids, dtype=torch.long)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'sr_labels': sr_labels,
            'role_ids': role_ids,
            'orders': orders,
            'target_ids': target_ids,
            'masked_indices': masked_indices
        }
