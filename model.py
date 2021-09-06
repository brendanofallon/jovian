
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
        self.fc1 = nn.Linear(in_dim, 256)
        self.fc2 = nn.Linear(256, out_dim)
        self.fc_vaf = nn.Linear(256, 1)
        self.elu = nn.ELU()
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        x = self.elu(self.fc1(x))
        x1 = self.softmax(self.fc2(x))
        # VAF stuff is VERY experimental - x has shape [batch, seqlen (in general this is variable), 4] - no linear
        # layer can be operate over the seq_dim since it can change from run to run, so instead we just sum it?? (Mean works better than sum)
        x_vaf = torch.sigmoid(self.fc_vaf(x).mean(dim=1))
        return x1, x_vaf


class VarTransformer(nn.Module):

    def __init__(self, in_dim, out_dim, nhead=6, d_hid=256, n_encoder_layers=2, p_dropout=0.1):
        super().__init__()
        self.embed_dim = nhead * 20
        self.fc1 = nn.Linear(in_dim, 200)
        self.fc2 = nn.Linear(200, self.embed_dim)
        self.pos_encoder = PositionalEncoding(self.embed_dim, p_dropout)
        encoder_layers = nn.TransformerEncoderLayer(self.embed_dim, nhead, d_hid, p_dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_encoder_layers)
        self.decoder = TwoHapDecoder(self.embed_dim, out_dim)
        self.elu = torch.nn.ELU()

    def forward(self, src):
        src = src.flatten(start_dim=2)
        src = self.elu(self.fc1(src))
        src = self.elu(self.fc2(src))
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output


class AltPredictor(nn.Module):

    def __init__(self, readnum, feats):
        super().__init__()
        self.fc1 = nn.Linear(feats + 4 + 5 + 1, 5)
        self.fc2 = nn.Linear(5, 5)
        self.fc3 = nn.Linear(5, 1)
        #         self.cn = nn.Conv2d(in_channels=feats+1, out_channels=2, stride=1, kernel_size=(readnum,1), padding=(0,0))
        self.l1 = nn.Linear(feats + 4 + 1, 5)
        self.l2 = nn.Linear(5, 5)
        self.l3 = nn.Linear(5, 5)

        self.elu = torch.nn.ELU()

    def forward(self, x, emit=False):
        # r is 1 if base is a match with the ref sequence, otherwise 0
        r = (x[:, :, :, 0:4] * x[:, :, 0:1, 0:4]).sum(dim=-1)  # x[:, :, 0:1..] is the reference seq
        z = torch.cat((r.unsqueeze(-1), x[:, :, :, :]), dim=3)

        # Compute the mean base usage across reads at every position, and join this with the features list for everything
        b = (x[:, :, :, 0:4]).mean(dim=2)
        bx = torch.cat((b.unsqueeze(2).expand(-1, -1, x.shape[2], -1), z), dim=3)

        # Take the features for every read pos, then compute read-level features by taking a mean of the features across the reads
        y = self.elu(self.l1(bx))
        y = self.elu(self.l2(y) + y)
        y = self.elu(self.l3(y) + y)
        ymean = y.mean(dim=2).unsqueeze(2)
        z = torch.cat((ymean.expand(-1, -1, x.shape[2], -1), bx), dim=3)

        # Take mean read level features (after reduction) and compute change this is an alt read
        x = self.elu(self.fc1(z)).squeeze(-1)
        x = self.elu(self.fc2(x) + x)
        x = 5 * self.elu(self.fc3(
            x))  # Important that this is an ELU and not ReLU activation since it must be able to produce negative values, and the factor of three makes them even more negative which translates into something closer to zero after sigmoid()
        x = torch.sigmoid(x.max(dim=1)[0]).squeeze(-1)
        return x


class VarTransformerAltMask(nn.Module):

    def __init__(self, readdepth, feature_count, out_dim, nhead=6, d_hid=256, n_encoder_layers=2, p_dropout=0.1):
        super().__init__()
        self.embed_dim = nhead * 20
        self.altpredictor = AltPredictor(readdepth, feature_count)
        self.fc1 = nn.Linear(readdepth * (feature_count+1), self.embed_dim)
        self.pos_encoder = PositionalEncoding(self.embed_dim, p_dropout)
        encoder_layers = nn.TransformerEncoderLayer(self.embed_dim, nhead, d_hid, p_dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_encoder_layers)
        self.decoder = TwoHapDecoder(self.embed_dim, out_dim)
        self.elu = torch.nn.ELU()

    def forward(self, src):
        altmask = self.altpredictor(src)
        # aex = altmask.unsqueeze(-1).unsqueeze(-1)
        # fullmask = aex.expand(src.shape[0], src.shape[2], src.shape[1],
        #                       src.shape[3]).transpose(1, 2)
        # src = src * fullmask

        src = src.flatten(start_dim=2)
        src = self.elu(self.fc1(src))
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output
