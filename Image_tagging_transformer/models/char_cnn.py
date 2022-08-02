import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# from utils.character_cnn import CharacterMapper, CharacterIndexer

class HighWay(nn.Module):

    def __init__(
        self, input_dim, num_layers=1, activation=F.relu
    ):
        super().__init__()

        self.input_dim = input_dim
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, input_dim * 2) for _ in range(num_layers)
        ])

        elf.activations = activation

        for layer in self.layers:
            layer.bias[input_dim:].data.fill_(1)


    def forward(self, inputs):
        current_input = inputs
        for layer in self.layers:
            projected_input = layer(current_input)
            linear_part = current_input
            nonlinear_part, gate = projected_input.chunk(2, dim=-1)
            nonlinear_part = self.activation(nonlinear_part)
            gate = torch.sigmoid(gate)
            current_input = gate * linear_part + (1 - gate) * nonlinear_part

        return current_input

class CharacterCNN(nn.Module):

    def __init__(self, output_dim, requires_grad):
        super().__init__()

        self._options = {
            'char_cnn': {
                'activation': 'relu',
                'filters': [
                    [1, 32],
                    [2, 32],
                    [3, 64],
                    [4, 128],
                    [5, 256],
                    [6, 512],
                    [7, 1024]
                    ],
                'n_highway': 2,
                'embedding': {'dim': 16},
                'n_characters': 262,
                'max_characters_per_token': 50
            }
        }

        self.output_dim = output_dim
        self.requires_grad = requires_grad

        self._init_weights()


    def _init_weights(self):
        self._init_char_embedding()
        self._init_cnn_weights()
        self._init_highway()
        self._init_projection()

    def _init_char_embedding(self):
        weights = np.zeros(
            (
                self._options["char_cnn"]["n_characters"] + 1,
                self._options["char_cnn"]["embedding"]["dim"]
            ),
            dtype="float32")
        weights[-1, :] *= 0.  # padding
        self._char_embedding_weights = torch.nn.Parameter(
            torch.FloatTensor(weights), requires_grad=self.requires_grad
        )

    def _init_cnn_weights(self):
        cnn_options = self._options["char_cnn"]
        filters = cnn_options["filters"]
        char_embed_dim = cnn_options["embedding"]["dim"]

        convolutions = []
        for i, (width, num) in enumerate(filters):
            conv = torch.nn.Conv1d(
                in_channels=char_embed_dim, out_channels=num,
                kernel_size=width, bias=True)
            conv.weight.requires_grad = self.requires_grad
            conv.bias.requires_grad = self.requires_grad
            convolutions.append(conv)
            self.add_module("char_conv_{}".format(i), conv)
        self._convolutions = convolutions

    def _init_highway(self):
        # the highway layers have same dimensionality as the number of cnn filters
        cnn_options = self._options["char_cnn"]
        filters = cnn_options["filters"]
        n_filters = sum(f[1] for f in filters)
        n_highway = cnn_options["n_highway"]

        self._highways = HighWay(n_filters, n_highway, activation=torch.nn.functional.relu)
        for k in range(n_highway):
            # The AllenNLP highway is one matrix multplication with concatenation of
            # transform and carry weights.
            self._highways.layers[k].weight.requires_grad = self.requires_grad
            self._highways.layers[k].bias.requires_grad = self.requires_grad

    def _init_projection(self):
        cnn_options = self._options["char_cnn"]
        filters = cnn_options["filters"]
        n_filters = sum(f[1] for f in filters)
        self._projection = torch.nn.Linear(n_filters, self.output_dim, bias=True)
        self._projection.weight.requires_grad = self.requires_grad
        self._projection.bias.requires_grad = self.requires_grad

    def get_output_dim(self):
        return self.output_dim

    def forward(self, inputs):
        """
        Parameters
        ----------
        inputs: ``torch.Tensor``
            Shape ``(batch_size, sequence_length, 50)`` of character ids representing the
            current batch.
        Returns
        -------
        embeddings: ``torch.Tensor``
            Shape ``(batch_size, sequence_length, embedding_dim)`` tensor with context
            insensitive token representations.
        """
        # Add BOS/EOS
        mask = ((inputs > 0).long().sum(dim=-1) > 0).long()
        #character_ids_with_bos_eos, mask_with_bos_eos = add_sentence_boundary_token_ids(
        #    inputs, mask, self._beginning_of_sentence_characters, self._end_of_sentence_characters
        #)
        character_ids_with_bos_eos, mask_with_bos_eos = inputs, mask

        # the character id embedding
        max_chars_per_token = self._options["char_cnn"]["max_characters_per_token"]
        # (batch_size * sequence_length, max_chars_per_token, embed_dim)
        character_embedding = torch.nn.functional.embedding(
            character_ids_with_bos_eos.view(-1, max_chars_per_token), self._char_embedding_weights
        )

        # run convolutions
        cnn_options = self._options["char_cnn"]
        if cnn_options["activation"] == "tanh":
            activation = torch.tanh
        elif cnn_options["activation"] == "relu":
            activation = torch.nn.functional.relu
        else:
            raise Exception("Unknown activation")

        # (batch_size * sequence_length, embed_dim, max_chars_per_token)
        character_embedding = torch.transpose(character_embedding, 1, 2)
        convs = []
        for i in range(len(self._convolutions)):
            conv = getattr(self, "char_conv_{}".format(i))
            convolved = conv(character_embedding)
            # (batch_size * sequence_length, n_filters for this width)
            convolved, _ = torch.max(convolved, dim=-1)
            convolved = activation(convolved)
            convs.append(convolved)

        # (batch_size * sequence_length, n_filters)
        token_embedding = torch.cat(convs, dim=-1)

        # apply the highway layers (batch_size * sequence_length, n_filters)
        token_embedding = self._highways(token_embedding)

        # final projection  (batch_size * sequence_length, embedding_dim)
        token_embedding = self._projection(token_embedding)

        # reshape to (batch_size, sequence_length, embedding_dim)
        batch_size, sequence_length, _ = character_ids_with_bos_eos.size()

        return token_embedding.view(batch_size, sequence_length, -1)