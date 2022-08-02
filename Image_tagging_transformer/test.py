# from transformers import BasicTokenizer, BertTokenizer

# # basic_tokenizer = BasicTokenizer(do_lower_case=True)

# tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=True)

# tokens = 'nobell.it/70ffb52d079109dca5664cce6f317373782/login.SkyPe.com/en/cgi-bin/verification/login/70ffb52d079109dca5664cce6f317373/index.php?cmd=_profile-ach&outdated_page_tmpl=p/gen/failed-to-load&nav=0.5.1&login_access=1322408526'
# # tokens = str(tokens)


# # tokens_a = basic_tokenizer.tokenize(tokens)

# # # tokens_a = [str(s) for s in tokens_a]

# # # print(tokens_a)
# tokens_new = []

# for tk_ in tokens:
#     tokens_new.extend(tokenizer.tokenize(tk_))
# # tokens_new = tokenizer.tokenize(tokens_a)

# # tokens_alt = tokenizer.tokenize(tokens)

# print(tokens_new)

# # print(tokens_alt)

# # input_ids = tokenizer.convert_tokens_to_ids(tokens_new)

# # print(input_ids)

# # from datasets.phishing_url import PhishingURL
# # from models import BertSeqClassification
# # from transformers import BertTokenizer

# # from torch.utils.data import DataLoader

# # tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=True)

# # dataset = PhishingURL(
# #     f_path='data/phishing_site_urls.csv',
# #     tokenizer=tokenizer
# # )
# # data_loader = DataLoader(dataset, batch_size=4)

# # batch = next(iter(data_loader))

# # print(batch['ids'].shape)

# # sample = {
# #     'input_ids': batch['ids'],
# #     'attention_mask': batch['mask'],
# #     'token_type_ids': batch['token_type_ids'],
# #     'labels': batch['target'],
# # }

# # model = BertSeqClassification('bert-large-uncased', 2)

# # # outputs = model(**sample)

# # # print(outputs)

# from datasets.phishing_url import PhishingURLChar, PhishingURL
from multi_url import URLIdentificationDataset
from transformers import BasicTokenizer, BertTokenizer
from torch.utils.data import random_split, SequentialSampler, RandomSampler, DataLoader

from tqdm import tqdm
import torch

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=True)

# dataset = PhishingURL(
#     'data/phishing_site_urls.csv',
#     tokenizer = tokenizer
# )

dataset = URLIdentificationDataset(
    'data/URL',
    tokenizer = tokenizer
)



# tr, vl = random_split(dataset, [50000, len(dataset)-50000])
# batch = dataset[0]

train_sampler = RandomSampler(dataset)

train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=4)

batch = next(iter(train_dataloader))

input_ids = batch['input_ids']
token_type_ids = batch['token_type_ids']
attention_mask = batch['attention_mask']
labels = batch['labels']

print(input_ids.shape, token_type_ids.shape, labels.shape)

# embedding = torch.nn.Embedding(30522, 16)

# emb_out = embedding(input_ids)
# emb_out = torch.transpose(emb_out, 1, 2)

# print(emb_out.shape)

# conv = torch.nn.Conv1d(16, 32, kernel_size=1, bias=True)

# conv_out = conv(emb_out)

# print(conv_out.shape)

# convolved, _ = torch.max(conv_out, dim=-1)

# print(convolved.shape)

from models.embeddings import CharacterEmbedding
from models.bert_classifier import CharacterBertModel

from transformers import BertForSequenceClassification

# embd = CharacterEmbedding(1024)

# position_embeddings = torch.nn.Embedding(512, 1024)

# seq_length = input_ids.shape[1]

# position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
# position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

# print(seq_length, position_ids.shape)

model = CharacterBertModel().to('cuda:0')

# bert_out = model(
#     input_ids=input_ids, 
#     attention_mask=attention_mask,
#     token_type_ids=token_type_ids,
#     labels=labels
# )

# print(bert_out)
# print(bert_out[0], bert_out[1])

# batch = {k: v.to(device='cuda:0', non_blocking=True) for k, v in batch.items()}

# bo = model(**batch)

# print(bo)

# _, pred = torch.max(bo[1], 1)
# print(pred)

# a = []

# for s in tqdm(dataset):
#     a.append(s['ids'])

# b = torch.cat(a)

# print(torch.unique(b), torch.unique(b).shape)

# from tqdm import tqdm
# import pandas as pd
# from tqdm import tqdm

# ll = []
# max_len = 512

# def func(x):

#     data = tokenizer.encode_plus(
#         x, max_length=max_len,
#         pad_to_max_length=False, add_special_tokens=True,
#         return_tensors='pt' 
#     )

#     return data['input_ids'].shape[1]


# df = pd.read_csv('data/phishing_site_urls.csv')
# df = df.assign(len_txt=df.URL.apply(lambda x: func(x)))
# df.to_csv('data/phishing_site_urls_token_len.csv')
# for l in tqdm(list(df['URL'])):
#     data = tokenizer.encode_plus(
#         l,
#         max_length=max_len,
#         pad_to_max_length=False,
#         add_special_tokens=True,
#         return_attention_mask=True,
#         return_token_type_ids=True,
#         return_tensors='pt'
#     )

#     ll.append(data['input_ids'].shape[1])

# print(sorted(list(set(ll))))
