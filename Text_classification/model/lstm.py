from turtle import forward
from torch import nn
import torch

class RNNBaseMoel(nn.Module):

    def __init__(self, pretrained_embeddings, freeze_embeddings, hidden_dim, n_class, n_layers, dropout):

        super(RNNBaseMoel, self).__init__()

        self.vocab_size = pretrained_embeddings.shape[0]
        self.embedding_dim = pretrained_embeddings.shape[1]

        self.embedding = nn.Embedding.from_pretrained(
            torch.from_numpy(pretrained_embeddings).float(),
            freeze=freeze_embeddings
            )

        self.lstm = torch.nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
            )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, n_class)

    def forward(self, ids, seq_len):

        embed_out = self.embedding(ids)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embed_out, seq_len, batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        # output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))

        return self.fc(hidden)