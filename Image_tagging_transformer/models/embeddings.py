import torch
from torch import nn
from torch.nn import functional as F

import numpy as np

class HighWay(nn.Module):

    def __init__(self, input_dim, num_layers=1):
        super().__init__()

        self.input_dim = input_dim

        self.layers = nn.ModuleList([
            nn.Linear(input_dim, input_dim * 2) for _ in range(num_layers)
        ])

    def forward(self, inputs):
        current_input = inputs
        for layer in self.layers:
            projected_input = layer(current_input)
            linear_part = current_input
            nonlinear_part, gate = projected_input.chunk(2, dim=-1)
            nonlinear_part = F.relu(nonlinear_part)
            gate = torch.sigmoid(gate)
            current_input = gate * linear_part + (1 - gate) * nonlinear_part
        
        return current_input

class CharacterEmbedding(nn.Module):

    def __init__(self, output_dim, vocab_dim=30522, emb_dim=16, n_highway=2):
        super().__init__()

        self.emb_dim = emb_dim
        self.n_highway = n_highway

        self.embedding = nn.Embedding(vocab_dim, emb_dim)

        filters = [
            [1, 32], [1, 32], [3, 64], [3, 128], 
            [5, 256], [5, 512], [7, 1024]
        ]
        n_filters = sum(f[1] for f in filters)
        self.n_filters = n_filters

        conv_blocks = []

        for (width, num) in filters:
            conv = nn.Conv1d(
                in_channels=emb_dim, out_channels=num,
                kernel_size=width, bias=True, padding=int((width -1)/2)
            )
            conv_blocks.append(conv)

        self.conv_blocks = nn.ModuleList(conv_blocks)

        self.highway = HighWay(n_filters, n_highway)
        self._init_highway()

        self.projection = nn.Linear(n_filters, output_dim, bias=True)

    def _init_highway(self):
        for k in range(self.n_highway):
            self.highway.layers[k].bias[self.n_filters:].data.fill_(1)

    def forward(self, input_ids):

        character_embedding = self.embedding(input_ids)

        character_embedding = torch.transpose(character_embedding, 1, 2)

        token_embedding = []
        for conv in self.conv_blocks:
            convolved = conv(character_embedding)
            # print('conv bM ', convolved.shape)
            # convolved, _ = torch.max(convolved, dim=-1)
            convolved = F.relu(convolved)
            convolved = torch.transpose(convolved, 1, 2)

            token_embedding.append(convolved)

        token_embedding = torch.cat(token_embedding, dim=-1)

        # apply the highway layers
        token_embedding = self.highway(token_embedding)

        # final projection
        token_embedding = self.projection(token_embedding)

        return token_embedding