from torch.utils.data import Dataset
import torch

import pandas as pd
import re
import numpy as np

class Twitchdataset(Dataset):

    class_to_id = {
        'GG' : 0, 
        'QS' : 1, 
        'JK' : 2,
    }
    def __init__(
        self,
        root='data/chats_data_sampled_combined.xlsx - Combined_round_4k.csv',
        tokenizer = None,
        chunk_len = 10,
        overlap_len = 5,
    ):
        self.tokenizer = tokenizer
        self.chunk_len = chunk_len
        self.overlap_len = overlap_len

        self.data, self.label = self.process_data(root)

    def process_data(self, root):
        df_temp = pd.read_csv(root)
        df = df_temp[['Message', 'Finalized']]
        # df['Message'] = df['Message'].astype(str)
        df = df.astype(str)
        # df = df.rename(columns={"Message":"text"})
        df.dropna(inplace=True)

        df = df.astype(str)

        df = df[df['Finalized'].isin(list(self.class_to_id.keys()))]

        df['label'] = df.Finalized.map(self.class_to_id)

        df_final = df.copy()
        df_final = df_final.reindex(np.random.permutation(df_final.index))
        df_final['text'] = df_final.Message.apply(self.clean_txt)

        return df_final['text'].values, df_final['label'].values

    def clean_txt(self, text):
        
        text = re.sub("'", "", text)
        text = re.sub("(\\W)+", " ", text)
        return text

    def long_terms_tokenizer(self, data_tokenize, targets):
        long_terms_token = []
        input_ids_list = []
        attention_mask_list = []
        token_type_ids_list = []
        targets_list = []

        previous_input_ids = data_tokenize["input_ids"].reshape(-1)
        previous_attention_mask = data_tokenize["attention_mask"].reshape(-1)
        previous_token_type_ids = data_tokenize["token_type_ids"].reshape(-1)
        remain = data_tokenize.get("overflowing_tokens").reshape(-1)
        targets = torch.tensor(targets, dtype=torch.int)

        input_ids_list.append(previous_input_ids)
        attention_mask_list.append(previous_attention_mask)
        token_type_ids_list.append(previous_token_type_ids)
        targets_list.append(targets)

        if (remain.nelement() != 0):
            remain = torch.tensor(remain, dtype=torch.long)
            idxs = range(len(remain)+self.chunk_len)
            idxs = idxs[(self.chunk_len-self.overlap_len-2)
                         ::(self.chunk_len-self.overlap_len-2)]
            input_ids_first_overlap = previous_input_ids[-(
                self.overlap_len+1):-1]
            start_token = torch.tensor([101], dtype=torch.long)
            end_token = torch.tensor([102], dtype=torch.long)

            for i, idx in enumerate(idxs):
                if i == 0:
                    input_ids = torch.cat(
                        (input_ids_first_overlap, remain[:idx]))
                elif i == len(idxs):
                    input_ids = remain[idx:]
                elif previous_idx >= len(remain):
                    break
                else:
                    input_ids = remain[(previous_idx-self.overlap_len):idx]

                previous_idx = idx

                nb_token = len(input_ids)+2
                attention_mask = torch.ones(self.chunk_len, dtype=torch.long)
                attention_mask[nb_token:self.chunk_len] = 0
                token_type_ids = torch.zeros(self.chunk_len, dtype=torch.long)
                input_ids = torch.cat((start_token, input_ids, end_token))
                if self.chunk_len-nb_token > 0:
                    padding = torch.zeros(
                        self.chunk_len-nb_token, dtype=torch.long)
                    input_ids = torch.cat((input_ids, padding))

                input_ids_list.append(input_ids)
                attention_mask_list.append(attention_mask)
                token_type_ids_list.append(token_type_ids)
                targets_list.append(targets)

        return {
            'ids': input_ids_list,  # torch.tensor(ids, dtype=torch.long),
            # torch.tensor(mask, dtype=torch.long),
            'mask': attention_mask_list,
            # torch.tensor(token_type_ids, dtype=torch.long),
            'token_type_ids': token_type_ids_list,
            'targets': targets_list,
            'len': [torch.tensor(len(targets_list), dtype=torch.long)]
        }

    def __getitem__(self, idx):
        """  Return a single tokenized sample at a given positon [idx] from data"""

        description = str(self.data[idx])
        targets = int(self.label[idx])
        data = self.tokenizer.encode_plus(
            description,
            max_length=self.chunk_len,
            pad_to_max_length=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_overflowing_tokens=True,
            return_tensors='pt')

        long_token = self.long_terms_tokenizer(data, targets)
        return long_token

    def __len__(self):
        """ Return data length """
        return self.label.shape[0]
    
class EqualizeTwitchDataset(Dataset):

    class_to_id = {
        'GG' : 0, 
        'QS' : 1, 
        'JK' : 2,
    }

    def __init__(
        self,
        root='data/chats_data_sampled_combined.xlsx - Combined_round_4k.csv',
        k = None,
        nK = None,
        model_type='robert',
        mode='train',
        tokenizer = None,
        chunk_len = None,
        overlap_len = None,
        max_length=None,
    ):

        self.model_type = model_type
        self.tokenizer = tokenizer
        self.chunk_len = chunk_len
        self.overlap_len = overlap_len
        self.max_length = max_length

        self.data, self.label = self.process_data(root, k, nK, mode)

    def process_data(self, root, k, nK=10, mode='train'):
        df_temp = pd.read_csv(root)
        df = df_temp[['Message', 'Finalized']]
        df = df.astype(str)
        df.dropna(inplace=True)

        df = df.astype(str)

        df = df[df['Finalized'].isin(list(self.class_to_id.keys()))]

        df['label'] = df.Finalized.map(self.class_to_id)

        df_final = df.copy()
        df_final = df_final.reindex(np.random.permutation(df_final.index))
        df_final['text'] = df_final.Message.apply(self.clean_txt)

        # Apply Equal split
        df_train = {
            'text':[],
            'label':[],
        }
        df_val = {
            'text':[],
            'label':[],
        }
        
        df_grouped = df_final.groupby('label')

        for i in self.class_to_id.values():
            df_class = df_grouped.get_group(i)
            # df_gg = df_gg.sample(frac=1).reset_index(drop=True)
            val_group_len = int(len(df_class) / nK)
            df_val['text'].append(df_class['text'].values[val_group_len*k:val_group_len*(k+1)])
            df_val['label'].append(df_class['label'].values[val_group_len*k:val_group_len*(k+1)])

            df_train['text'].append(df_class['text'].values[:val_group_len*k])
            df_train['text'].append(df_class['text'].values[val_group_len*(k+1):])
            df_train['label'].append(df_class['label'].values[:val_group_len*k])
            df_train['label'].append(df_class['label'].values[val_group_len*(k+1):])

        if mode=='train':
            return np.concatenate(df_train['text']), np.concatenate(df_train['label'])
        else:
            return np.concatenate(df_val['text']), np.concatenate(df_val['label'])

    def clean_txt(self, text):
        
        text = re.sub("'", "", text)
        text = re.sub("(\\W)+", " ", text)
        return text

    def long_terms_tokenizer(self, data_tokenize, targets):
        long_terms_token = []
        input_ids_list = []
        attention_mask_list = []
        token_type_ids_list = []
        targets_list = []

        previous_input_ids = data_tokenize["input_ids"].reshape(-1)
        previous_attention_mask = data_tokenize["attention_mask"].reshape(-1)
        previous_token_type_ids = data_tokenize["token_type_ids"].reshape(-1)
        remain = data_tokenize.get("overflowing_tokens").reshape(-1)
        targets = torch.tensor(targets, dtype=torch.int)

        input_ids_list.append(previous_input_ids)
        attention_mask_list.append(previous_attention_mask)
        token_type_ids_list.append(previous_token_type_ids)
        targets_list.append(targets)

        if (remain.nelement() != 0):
            remain = torch.tensor(remain, dtype=torch.long)
            idxs = range(len(remain)+self.chunk_len)
            idxs = idxs[(self.chunk_len-self.overlap_len-2)
                         ::(self.chunk_len-self.overlap_len-2)]
            input_ids_first_overlap = previous_input_ids[-(
                self.overlap_len+1):-1]
            start_token = torch.tensor([101], dtype=torch.long)
            end_token = torch.tensor([102], dtype=torch.long)

            for i, idx in enumerate(idxs):
                if i == 0:
                    input_ids = torch.cat(
                        (input_ids_first_overlap, remain[:idx]))
                elif i == len(idxs):
                    input_ids = remain[idx:]
                elif previous_idx >= len(remain):
                    break
                else:
                    input_ids = remain[(previous_idx-self.overlap_len):idx]

                previous_idx = idx

                nb_token = len(input_ids)+2
                attention_mask = torch.ones(self.chunk_len, dtype=torch.long)
                attention_mask[nb_token:self.chunk_len] = 0
                token_type_ids = torch.zeros(self.chunk_len, dtype=torch.long)
                input_ids = torch.cat((start_token, input_ids, end_token))
                if self.chunk_len-nb_token > 0:
                    padding = torch.zeros(
                        self.chunk_len-nb_token, dtype=torch.long)
                    input_ids = torch.cat((input_ids, padding))

                input_ids_list.append(input_ids)
                attention_mask_list.append(attention_mask)
                token_type_ids_list.append(token_type_ids)
                targets_list.append(targets)

        return {
            'ids': input_ids_list,  # torch.tensor(ids, dtype=torch.long),
            # torch.tensor(mask, dtype=torch.long),
            'mask': attention_mask_list,
            # torch.tensor(token_type_ids, dtype=torch.long),
            'token_type_ids': token_type_ids_list,
            'targets': targets_list,
            'len': [torch.tensor(len(targets_list), dtype=torch.long)]
        }

    def __getitem__(self, idx):
        """  Return a single tokenized sample at a given positon [idx] from data"""

        description = str(self.data[idx])
        targets = int(self.label[idx])

        if self.model_type == 'robert':
            data = self.tokenizer.encode_plus(
                description,
                max_length=self.chunk_len,
                pad_to_max_length=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_token_type_ids=True,
                return_overflowing_tokens=True,
                return_tensors='pt')

            long_token = self.long_terms_tokenizer(data, targets)
            return long_token

        elif self.model_type == 'bert':
            data = self.tokenizer.encode_plus(
                description,
                max_length=self.max_length,
                pad_to_max_length=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_token_type_ids=True,
                return_tensors='pt')
            
            return data, np.array(targets)

        elif self.model_type == 'lstm':
            data, seq_len = self.tokenizer.encode(
                description,
                max_len = self.max_length,
            )

            return data, np.array(targets), seq_len


    def __len__(self):
        """ Return data length """
        return self.label.shape[0]