

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
        self.fc1 = nn.Linear(in_dim, 400)
        self.fc2 = nn.Linear(400, out_dim)
        self.fc_vaf = nn.Linear(400, 1)
        self.elu = nn.ELU()
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        x = self.elu(self.fc1(x))
        x1 = self.softmax(self.fc2(x))
        # VAF stuff is VERY experimental - x has shape [batch, seqlen (in general this is variable), 4] - no linear
        # layer can be operate over the seq_dim since it can change from run to run, so instead we just sum it?? (Mean works better than sum)
        # x_vaf = torch.sigmoid(self.fc_vaf(x).mean(dim=1))
        return x1


class VarTransformer(nn.Module):

    def __init__(self, read_depth, feature_count, out_dim, nhead=6, d_hid=256, n_encoder_layers=2, p_dropout=0.1):
        super().__init__()
        self.embed_dim = nhead * 20
        self.fc1 = nn.Linear(read_depth * feature_count, 200)
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
    
    def __init__(self, readnum, feats, device):
        super().__init__()
        self.device = device
        self.g_hidden = 20
        self.g_output = 10
        self.l_hidden = 10
        self.l_output = 10
        self.conv_output = 10
        self.f_hidden = 10
        
        self.g1 = nn.Linear(feats + 1, self.g_hidden)
        self.g2 = nn.Linear(self.g_hidden, self.g_output)
        self.l1 = nn.Linear(feats + 1 + self.g_output, self.l_hidden)
        self.l2 = nn.Linear(self.l_hidden, self.l_output)
        self.conv1 = nn.Conv2d(in_channels=feats +1 +self.l_output +self.g_output, out_channels=self.conv_output, kernel_size=(1,10), padding='same')
        self.fc1 = nn.Linear(self.conv_output + feats +1, self.f_hidden)
        self.fc2 = nn.Linear(self.f_hidden, 1)
        self.elu = torch.nn.ELU()

    def forward(self, x, emit=False):
        x = x.float().to(self.device)
        # r is 1 if base is a match with the ref sequence, otherwise 0
        r = (x[:, :, :, 0:4] * x[:, :, 0:1, 0:4]).sum(dim=-1) # x[:, :, 0:1..] is the reference seq
        z0 = torch.cat((r.unsqueeze(-1), x[:, :, :, :]), dim=3).to(self.device)
#         z = torch.cat((x[:, :, 0:1, 0:4].expand(-1, -1, x.shape[2], -1), x), dim=3).to(device)

        # Compute the mean of each raw feature usage across reads at every position, and join this with the features list for everything
        b1 = self.elu(self.g1(z0)) 
        b2 = self.elu(self.g2(b1)).mean(dim=2)

#         b = z0.mean(dim=2)
#         b[:, :, 0:4] = b[:, :, 0:4] / (b[:, :, 0:4].sum(dim=-1).unsqueeze(-1) + 0.001)
        bx = torch.cat((b2.unsqueeze(2).expand(-1, -1, x.shape[2], -1), z0), dim=3).to(self.device)
       
        # Take the features for every read pos, then compute read-level features by taking a mean of the features across the reads
        y = self.elu(self.l1(bx))
        y = self.elu(self.l2(y))
        ymean = y.mean(dim=2).unsqueeze(2)
        z = torch.cat((ymean.expand(-1, -1, x.shape[2], -1), bx), dim=3).to(self.device)
#         print(f"z: {z.shape}")
       
        cx = self.elu(self.conv1(z.transpose(3,1))).transpose(1,3)  # Features must be in second dimension (we use it as 'channels')
        cxz = torch.cat((cx, z0), dim=3).to(self.device)

        
        x = self.elu(self.fc1(cxz)).squeeze(-1)
        x = 6 * self.elu(self.fc2(x) ) # Important that this is an ELU and not ReLU activation since it must be able to produce negative values, and the factor of 4 makes them even more negative which translates into something closer to zero after sigmoid()
        x = torch.sigmoid(x.max(dim=1)[0]).squeeze(-1)
        return x



class VarTransformerAltMask(nn.Module):

    def __init__(self, read_depth, feature_count, out_dim, nhead=6, d_hid=256, n_encoder_layers=2, p_dropout=0.1, altpredictor_sd=None, train_altpredictor=False, device='cpu'):
        super().__init__()
        self.device=device
        self.embed_dim = nhead * 20
        self.train_altpredictor = train_altpredictor
        self.altpredictor = AltPredictor(read_depth, feature_count, device)
        if altpredictor_sd is not None:
            logger.info(f"Loading altpredictor statedict from {altpredictor_sd}, training={train_altpredictor}")
            self.altpredictor.load_state_dict(torch.load(altpredictor_sd))
        else:
            logger.info(f"Not loading altpredictor params, initializing randomly, training={train_altpredictor}")

        if train_altpredictor == False and altpredictor_sd is None:
            raise ValueError("Turning off training for altpredictor but NOT supplying a state_dict for it seems crazy (the params will be randomly initialized but never trained)")

        self.first_hidden_dim = 5
        self.fc1 = nn.Linear(feature_count, self.first_hidden_dim)
        self.fc2 = nn.Linear(read_depth * self.first_hidden_dim, self.embed_dim)

        # self.pos_encoder = PositionalEncoding(self.embed_dim, p_dropout)
        self.pos_encoder = PositionalEncoding2D(self.first_hidden_dim, self.device)
        encoder_layers = nn.TransformerEncoderLayer(self.embed_dim, nhead, d_hid, p_dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_encoder_layers)
        self.decoder = TwoHapDecoder(self.embed_dim, out_dim)
        self.elu = torch.nn.ELU()

    def forward(self, src):
        if self.train_altpredictor:
            altmask = self.altpredictor(src).to(self.device)
        else:
            with torch.no_grad():
                altmask = self.altpredictor(src).to(self.device)

        # Force the first read in each batch to have weight = 1, because this is the reference read but we _dont_ want to mask it
        predicted_altmask = torch.cat((torch.ones(src.shape[0], 1).to(self.device), altmask[:, 1:]), dim=1)
        predicted_altmask = predicted_altmask.clamp(0.001, 1.0)
        aex = predicted_altmask.unsqueeze(-1).unsqueeze(-1)
        fullmask = aex.expand(src.shape[0], src.shape[2], src.shape[1],
                              src.shape[3]).transpose(1, 2).to(self.device)

        src = src * fullmask
        src = self.elu(self.fc1(src))
        src = self.pos_encoder(src) # Should happen *before* any sort of flattening.. but after first linear layer??
        src = self.elu(self.fc2(src.flatten(start_dim=2)))
        output = self.transformer_encoder(src)
        output = self.decoder(output)

        pred_vaf = altmask.mean(dim=1)
        return output, pred_vaf
