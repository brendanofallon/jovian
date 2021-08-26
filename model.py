
import torch
import torch.nn as nn
import math


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
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, out_dim)
        self.fc_vaf = nn.Linear(128, 1)
        self.elu = nn.ELU()
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        x = self.elu(self.fc1(x))
        x1 = self.softmax(self.fc2(x))
        # VAF stuff is VERY experimental - x has shape [batch, seqlen (in general this is variable), 4] - no linear
        # layer can be operate over the seq_dim since it can change from run to run, so instead we just sum it??
        x_vaf = torch.sigmoid(self.fc_vaf(x).sum(dim=1))
        return x1, x_vaf


class VarTransformer(nn.Module):

    def __init__(self, in_dim, out_dim, nhead=6, d_hid=256, n_encoder_layers=2, p_dropout=0.1):
        super().__init__()
        self.embed_dim = nhead * 24
        self.fc1 = nn.Linear(in_dim, self.embed_dim)
        self.pos_encoder = PositionalEncoding(self.embed_dim, p_dropout)
        encoder_layers = nn.TransformerEncoderLayer(self.embed_dim, nhead, d_hid, p_dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_encoder_layers)
        self.decoder = TwoHapDecoder(self.embed_dim, out_dim)
        self.elu = torch.nn.ELU()

    def forward(self, src):
        src = self.elu(self.fc1(src))
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output