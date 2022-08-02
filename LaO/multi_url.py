from torch.utils.data import Dataset
import torch

import pandas as pd
import numpy as np
import os

class URLIdentificationDataset(Dataset):

    class_to_id = {
        'benign': 0,
        'defacement': 1,
        'malware': 2,
        'phishing': 3,
        'spam': 4
    }

    def __init__(self, root, tokenizer = None, max_len=512):

        self.tokenizer = tokenizer
        self.max_len = max_len

        self.data, self.label = self.process_data(root)

    def process_data(self, root):
        urls = []
        labels = []

        for cls in self.class_to_id.keys():
            f_path = os.path.join(root, cls + '_dataset.csv')
            df = pd.read_csv(f_path, header=None, names=["url"])

            urls.extend(list(df['url']))
            labels.extend([self.class_to_id[cls]] * len(list(df['url'])))

        return urls, labels

    def __getitem__(self, idx):
        description = str(self.data[idx])
        targets = int(self.label[idx])

        data = self.tokenizer.encode_plus(
            description,
            max_length=self.max_len,
            pad_to_max_length=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors='pt'
        )

        return {
            'input_ids': data['input_ids'].squeeze(),
            'attention_mask': data['attention_mask'].squeeze(),
            'token_type_ids': data['token_type_ids'].squeeze(),
            'labels': targets
        }

    def __len__(self):
        return len(self.label)