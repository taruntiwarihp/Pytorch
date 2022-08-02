from torch.utils.data import Dataset
import torch

import pandas as pd
import numpy as np

class PhishingURL(Dataset):

    class_to_id = {
        'good': 0,
        'bad': 1,
    }

    def __init__(self, f_path, tokenizer = None, max_len=512):

        self.tokenizer = tokenizer
        self.max_len = max_len

        self.data, self.label = self.process_data(f_path)

    def process_data(self, f_path):
        df_temp = pd.read_csv(f_path)
        df = df_temp[['URL', 'Label']]
        df = df.astype(str)

        df.dropna(inplace=True)
        df = df.astype(str)

        df = df[df['Label'].isin(list(self.class_to_id.keys()))]

        df['label'] = df.Label.map(self.class_to_id)

        # df = df.assign(len_txt=df.URL.apply(lambda x: len(x)))
        # df = df[df.len_txt < self.max_len]
        # del df['len_txt']

        df_final = df.copy()
        df_final = df_final.reindex(np.random.permutation(df_final.index))

        return df_final['URL'].values, df_final['label'].values

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
        return self.label.shape[0]

class PhishingURLChar(Dataset):

    class_to_id = {
        'good': 0,
        'bad': 1,
    }

    def __init__(self, f_path, tokenizer = None, max_len=512):

        self.tokenizer = tokenizer
        self.max_len = max_len

        self.data, self.label = self.process_data(f_path)

    def process_data(self, f_path):
        df_temp = pd.read_csv(f_path)
        df = df_temp[['URL', 'Label']]
        df = df.astype(str)

        df.dropna(inplace=True)
        df = df.astype(str)

        df = df[df['Label'].isin(list(self.class_to_id.keys()))]

        df['label'] = df.Label.map(self.class_to_id)

        # df = df.assign(len_txt=df.URL.apply(lambda x: len(x)))
        # df = df[df.len_txt < self.max_len]
        # del df['len_txt']

        df_final = df.copy()
        df_final = df_final.reindex(np.random.permutation(df_final.index))

        return df_final['URL'].values, df_final['label'].values

    def __getitem__(self, idx):
        description = str(self.data[idx])
        targets = int(self.label[idx])

        char_tokens = []

        for ct in description:
            char_tokens.extend(self.tokenizer.tokenize(ct))

        char_tokens = char_tokens[:(self.max_len - 2)]
        char_tokens = ["[CLS]"] + char_tokens + ["[SEP]"]
        segment_ids = [0] * len(char_tokens)


        input_ids = self.tokenizer.convert_tokens_to_ids(char_tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        input_mask = [1] * len(char_tokens)

        # Zero-pad up to the sequence length.
        padding_length = self.max_len - len(input_mask)

        input_ids += [0] * padding_length
        input_mask += [0] * padding_length
        segment_ids += [0] * padding_length

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(input_mask, dtype=torch.long),
            'token_type_ids': torch.tensor(segment_ids, dtype=torch.long),
            'labels': torch.tensor(targets, dtype=torch.long)
        }

    def __len__(self):
        return self.label.shape[0]
