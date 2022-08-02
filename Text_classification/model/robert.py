from torch import nn
from transformers import BertModel
import torch

class RoBERT_Model(nn.Module):

    def __init__(self, bertFineTuned, lstm_hidden_layer, n_class):
        super(RoBERT_Model, self).__init__()
        self.lstm_hidden_layer = lstm_hidden_layer
        self.bertFineTuned = bertFineTuned
        self.lstm = nn.LSTM(768, lstm_hidden_layer, num_layers=1, bidirectional=False)
        self.out = nn.Linear(lstm_hidden_layer, n_class)

    def forward(self, ids, mask, token_type_ids, lengt):

        output = self.bertFineTuned(
            ids, attention_mask=mask, token_type_ids=token_type_ids)
        chunks_emb = output['pooler_output'].split_with_sizes(lengt)

        seq_lengths = torch.LongTensor([x for x in map(len, chunks_emb)])

        batch_emb_pad = nn.utils.rnn.pad_sequence(
            chunks_emb, padding_value=-91, batch_first=True)
        batch_emb = batch_emb_pad.transpose(0, 1)  # (B,L,D) -> (L,B,D)
        lstm_input = nn.utils.rnn.pack_padded_sequence(
            batch_emb, seq_lengths.cpu().numpy(), batch_first=False, enforce_sorted=False)

        packed_output, (h_t, h_c) = self.lstm(lstm_input, )  # (h_t, h_c))
        h_t = h_t.view(-1, self.lstm_hidden_layer)

        return self.out(h_t)