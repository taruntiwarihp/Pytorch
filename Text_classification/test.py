# # import warnings
# # warnings.filterwarnings("ignore", category=DeprecationWarning)
# # warnings.filterwarnings("ignore", category=UserWarning)
# # warnings.filterwarnings("ignore", category=FutureWarning)

# # import pandas as pd
# # from collections import Counter
# # import matplotlib.pyplot as plt
# # from transformers import BertTokenizer
# # from sklearn.preprocessing import LabelEncoder

# # df = pd.read_csv('data/chats_data_sampled_combined.xlsx - Combined_round_2k.csv')
# # print(df.columns)

# # df_final = df[['Message', 'Finalized']]
# # df_final.dropna(inplace=True)

# # # df_final.drop(df_final.loc[df_final['Finalized'] not in ['JK', 'GG', 'QS']].index, inplace=True)
# # # df_final = df_final[df_final.Finalized not in ['JK', 'GG', 'QS']]
# # mask = df_final['Finalized'].isin(['JK', 'GG', 'QS'])
# # df_final = df_final[mask]

# # df_final = df_final.astype(str)

# # # print(df_final.isnull().sum())

# # print(set(df_final['Finalized']))
# # print(Counter(df_final['Finalized']))

# # class_to_id = {
# #     'GG' : 0, 
# #     'QS' : 1, 
# #     'JK' : 2,
# # }

# # print(class_to_id)

# # df_final['label'] = df_final.Finalized.map(class_to_id)

# # print(df_final)




# # # print(df['Message'])

# # # df_final = df_final.assign(len_txt=df.Message.apply(lambda x: len(str(x).split())))
# # # df_filt = df_final[df_final.len_txt > 30]

# # # print(df_filt)

# # # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# # # msg = list(df_filt['Message'])[0]
# # # data = tokenizer.encode_plus(
# # #     msg,
# # #     max_length=64,
# # #     # pad_to_max_length=True,
# # #     padding='max_length',
# # #     add_special_tokens=True,
# # #     return_attention_mask=True,
# # #     return_token_type_ids=True,
# # #     return_overflowing_tokens=True,
# # #     return_tensors='pt'
# # # )

# # # print(data)


# # print('############################')

# # from dataset.dataset import Twitchdataset
# # import torch
# # from torch.utils.data import DataLoader
# # from model import Base_Model
# # from sklearn.model_selection import KFold

# # bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# # data = Twitchdataset(tokenizer=bert_tokenizer)

# # # for i in range(10):
# # #     print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$                   ', i)
# # #     print(data[i])
# # # print(data.label)

# # def my_collate(batches):
# #     # return batches
# #     return [{key: torch.stack(value) for key, value in batch.items()} for batch in batches]


# # train_data_loader=DataLoader(
# #     data,
# #     batch_size=8,
# #     collate_fn=my_collate)

# # kfold = KFold(n_splits=10, shuffle=True)

# # for fold, (train_ids, test_ids) in enumerate(kfold.split(data)):

# #     print(test_ids)

# # # batch = next(iter(train_data_loader))

# # # model = Base_Model(n_class=3).cuda()

# # # device="cuda"

# # # ids = [data["ids"] for data in batch]
# # # mask = [data["mask"] for data in batch]
# # # token_type_ids = [data["token_type_ids"] for data in batch]
# # # targets = [data["targets"] for data in batch]
# # # lengt = [data['len'] for data in batch]

# # # ids = torch.cat(ids)
# # # mask = torch.cat(mask)
# # # token_type_ids = torch.cat(token_type_ids)
# # # targets = torch.cat(targets)
# # # lengt = torch.cat(lengt)

# # # ids = ids.to(device, dtype=torch.long)
# # # mask = mask.to(device, dtype=torch.long)
# # # token_type_ids = token_type_ids.to(device, dtype=torch.long)
# # # targets = targets.to(device, dtype=torch.long)

# # # outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)

# from torch.utils.data import DataLoader
# from torch.utils.data.sampler import SubsetRandomSampler
# from model import Base_Model, RoBERT_Model
# from dataset.dataset import Twitchdataset
# from utils.logging import create_logger, get_model_summary
# from utils.utils import my_collate, prepare_batch_for_model
# from transformers import get_linear_schedule_with_warmup, AdamW, BertTokenizer
# from sklearn.model_selection import KFold
# import torch
# import numpy as np
# from tqdm import tqdm

# from sklearn.metrics import accuracy_score, f1_score, precision_score

# bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
# dataset = Twitchdataset(tokenizer=bert_tokenizer)

# kfold = KFold(n_splits=10, shuffle=True)

# for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):

#     train_sampler = SubsetRandomSampler(train_ids)

#     train_data_loader=DataLoader(
#         dataset,
#         batch_size=8,
#         sampler=train_sampler,
#         collate_fn=my_collate,
#         drop_last=True)


#     child_model=torch.load('weights/base_model.pt')
#     model=RoBERT_Model(bertFineTuned=list(child_model.children())[0], lstm_hidden_layer = 100, n_class = 3).to("cuda")

#     batch = next(iter(train_data_loader))

#     ids, mask, token_type_ids, targets, lengt = prepare_batch_for_model(batch)

#     print(lengt)

#     outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids, lengt=lengt)
#     out_child = child_model(ids=ids, mask=mask, token_type_ids=token_type_ids)
#     print(ids.shape, outputs.shape, out_child.shape, targets.shape)
#     print(targets)

#     # new_target = torch.tensor([tm[0].item() for tm in targets.split_with_sizes(lengt)], dtype=torch.long, device='cuda')
#     loss_fun = torch.nn.CrossEntropyLoss()

#     loss = loss_fun(outputs, targets)

#     print(loss)

#     fin_targets = []
#     fin_outputs = []

#     epoch = 1

#     with torch.no_grad():
#         with tqdm(train_data_loader, unit="batch") as tloader:
#             for batch in tloader:

#                 tloader.set_description("Epoch {}".format(epoch))
#                 ids, mask, token_type_ids, targets, lengt = prepare_batch_for_model(batch)

#                 outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids, lengt=lengt)
#                 tloader.set_postfix(loss=loss.item())

#                 _, pred = torch.max(outputs.data, 1)

#                 fin_targets.append(targets.cpu().detach().numpy())
#                 fin_outputs.append(pred.cpu().detach().numpy())

#     target = np.concatenate(fin_targets)
#     predicted = np.concatenate(fin_outputs)

#     print(target.shape)
#     print(predicted.shape)

#     print(target)
#     print(predicted)

#     acc = accuracy_score(predicted, target)
#     precision = precision_score(predicted, target, average='micro')

#     print(acc, precision)

#     break

# from dataset.dataset import Twitchdataset, EqualizeTwitchDataset
# from transformers import get_linear_schedule_with_warmup, AdamW, BertTokenizer
# import numpy as np
# from torch.utils.data import DataLoader
# from model.bert import BERT_Model
# from torch import nn
# from utils.utils import prepare_batch_for_bert_model

# root = 'data/chats_data_sampled_combined.xlsx - Combined_round_4k.csv'
# fold = 7
# kfold = 10
# chunk_len = 10
# overlap_len = 5
# max_length=128
# model_type = 'bert'
# bert_config = 'bert-base-uncased'

# bert_tokenizer = BertTokenizer.from_pretrained(bert_config, do_lower_case=True)

# train_set = EqualizeTwitchDataset(
#     root=root,
#     k = fold,
#     nK=kfold,
#     mode='train',
#     model_type = model_type,
#     tokenizer=bert_tokenizer,
#     max_length=max_length,
# )

# data_loader = DataLoader(train_set, batch_size=4, shuffle=True)

# batch = next(iter(data_loader))

# print(batch[0].keys(), batch[1])

# model = BERT_Model(n_class=3).to("cuda")

# ids, mask, token_type_ids, targets = prepare_batch_for_bert_model(batch)

# print(ids.shape, mask.shape, token_type_ids.shape)
# out = model(ids, mask, token_type_ids)

# print(out.shape)

# loss_fn = nn.CrossEntropyLoss()

# loss = loss_fn(out, targets)

# print(loss)

# import numpy as np


# vocab,embeddings = [],[]
# with open('dataset/glove/glove.6B.300d.txt','rt') as fi:
#     full_content = fi.read().strip().split('\n')
# for i in range(len(full_content)):
#     i_word = full_content[i].split(' ')[0]
#     i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
#     vocab.append(i_word)
#     embeddings.append(i_embeddings)

# vocab_npa = np.array(vocab)
# embs_npa = np.array(embeddings)

# vocab_npa = np.insert(vocab_npa, 0, '<pad>')
# vocab_npa = np.insert(vocab_npa, 1, '<unk>')

# pad_emb_npa = np.zeros((1,embs_npa.shape[1]))   #embedding for '<pad>' token.
# unk_emb_npa = np.mean(embs_npa,axis=0,keepdims=True)    #embedding for '<unk>' token.

# #insert embeddings for pad and unk tokens at top of embs_npa.
# embs_npa = np.vstack((pad_emb_npa,unk_emb_npa,embs_npa))

# print(embs_npa.shape)

# print(vocab_npa[:10])


from dataset.tokenizer import GloveTokenizer
from dataset.dataset import EqualizeTwitchDataset
from torch.utils.data import DataLoader
from model.lstm import RNNBaseMoel
from torch import nn

glove = GloveTokenizer()

data = EqualizeTwitchDataset(model_type='lstm', k=1, nK=10, tokenizer=glove, max_length=512)

data_loader = DataLoader(data, batch_size=4)
data, label, seq_len = next(iter(data_loader))

print(data.shape, label, seq_len)

model = RNNBaseMoel(
    pretrained_embeddings=glove.embeddings,
    freeze_embeddings=True,
    hidden_dim=100,
    n_class=3,
    n_layers=4,
    dropout=0.2,
)

out = model(data, seq_len)

print(out.shape)

loss_fun = nn.CrossEntropyLoss()

loss = loss_fun(out, label)

print(loss)