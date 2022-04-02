

import logging

import torch
import torch.nn as nn
import numpy as np
import math


logger = logging.getLogger(__name__)

class PositionalEncoding2D(nn.Module):

    def __init__(self, channels, device):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        self.device = device
        channels = int(np.ceil(channels/4)*2)
        self.channels = channels
        inv_freq = 1. / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer('inv_freq', inv_freq)
        self.cache_shape = None
        self.enc_cache = None

    def _from_cache(self, tensor):
        shape = list(tensor.size())[1:]
        if shape == self.cache_shape and self.enc_cache is not None:
            return self.enc_cache
        else:
            batch_size, x, y, orig_ch = tensor.shape
            pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
            pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
            sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
            sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
            emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(1)
            emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
            emb = torch.zeros((x, y, self.channels * 2), device=tensor.device).type(tensor.type())
            emb[:, :, :self.channels] = emb_x
            emb[:, :, self.channels:2 * self.channels] = emb_y
            self.enc_cache = emb
            self.cache_shape = shape
        return self.enc_cache

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")
        batch_size, x, y, orig_ch = tensor.shape
        emb = self._from_cache(tensor)
        return tensor + emb[None, :, :, :orig_ch].expand(batch_size, -1, -1, -1)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TwoHapDecoder(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 100)
        self.fc_hap0 = nn.Linear(100, out_dim)
        self.fc_hap1 = nn.Linear(100, out_dim)
        self.elu = nn.ELU()
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        x = self.elu(self.fc1(x))
        x0 = self.softmax(self.fc_hap0(x))
        x1 = self.softmax(self.fc_hap1(x))
        return torch.stack((x0, x1), dim=1)


class VarTransformer(nn.Module):

    def __init__(self, read_depth, feature_count, out_dim, nhead=6, d_hid=256, embed_dim_factor=40, n_encoder_layers=2, p_dropout=0.1, device='cpu'):
        super().__init__()
        self.device=device
        self.embed_dim = nhead * embed_dim_factor
        self.fc1_hidden = 12

        self.fc1 = nn.Linear(feature_count, self.fc1_hidden)
        self.fc2 = nn.Linear(read_depth * self.fc1_hidden, self.embed_dim)

        self.pos_encoder = PositionalEncoding(self.embed_dim, p_dropout)
        #self.pos_encoder = PositionalEncoding2D(self.fc1_hidden, self.device)
        encoder_layers = nn.TransformerEncoderLayer(self.embed_dim, nhead, d_hid, p_dropout, batch_first=True, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_encoder_layers)
        self.decoder = TwoHapDecoder(self.embed_dim, out_dim)
        self.elu = torch.nn.ELU()

    def forward(self, src):

        # r contains a 1 if the read base matches the reference base, 0 otherwise
        #r = (src[:, :, :, 0:4] * src[:, :, 0:1, 0:4]).sum(dim=-1) # x[:, :, 0:1..] is the reference seq
        #src = torch.cat((src, r.unsqueeze(-1)), dim=3).to(self.device)


        src = self.elu(self.fc1(src))
        #src = self.pos_encoder(src) # For 2D encoding we have to do this before flattening, right?
        src = src.flatten(start_dim=2)
        src = self.elu(self.fc2(src))
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)

        return output


