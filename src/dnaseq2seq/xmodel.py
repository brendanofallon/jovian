
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math

import util
from dnaseq2seq.model import PositionalEncoding2D, PositionalEncoding, VarTransformer

from x_transformers import TransformerWrapper, Encoder, Decoder, AutoregressiveWrapper


class Identity(nn.Module):

    def forward(self, x, *args, **kwargs):
        return x

def identity(x, *args, **kwargs):
    return x

class XVarTransformer(nn.Module):

    def __init__(self,
                 read_depth,
                 feature_count,
                 kmer_dim,
                 encoder_attention_heads=6,
                 decoder_attention_heads=6,
                 d_ff=1024,
                 embed_dim_factor=40,
                 n_encoder_layers=3,
                 n_decoder_layers=3,
                 p_dropout=0.1,
                 device='cpu'):
        super().__init__()

        # For quantization aware training - see https://pytorch.org/tutorials/recipes/quantization.html
        self.device = device
        self.read_depth = read_depth
        self.kmer_dim = kmer_dim
        self.embed_dim = encoder_attention_heads * embed_dim_factor
        self.fc1_hidden = 12

        self.fc1 = nn.Linear(feature_count, self.fc1_hidden)
        self.fc2 = nn.Linear(self.read_depth * self.fc1_hidden, self.embed_dim)

        self.converter = nn.Linear(self.embed_dim, self.kmer_dim)
        self.pos_encoder = PositionalEncoding2D(self.fc1_hidden, self.device)
        self.tgt_pos_encoder = PositionalEncoding(self.kmer_dim, batch_first=True, max_len=500).to(self.device)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=encoder_attention_heads,
            dim_feedforward=d_ff,
            dropout=p_dropout,
            batch_first=True,
            activation='gelu')
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers=n_encoder_layers)

        # decoder_layers = nn.TransformerDecoderLayer(
        #     d_model=self.kmer_dim,
        #     nhead=decoder_attention_heads,
        #     dim_feedforward=d_ff,
        #     dropout=p_dropout,
        #     batch_first=True,
        #     activation='gelu')


        normalmodel0 = TransformerWrapper(
            num_tokens=util.FEATURE_DIM,
            max_seq_len=500,
            attn_layers=Decoder(
                dim=self.kmer_dim,
                depth=n_decoder_layers,
                heads=decoder_attention_heads,
                attn_dropout=0.1,
                cross_attend=True,
            ),
        )
        normalmodel0.token_emb = Identity()  # Identity func, but takes arbitrary args that do nothing
        normalmodel0.pos_emb = Identity()
        normalmodel1 = TransformerWrapper(
            num_tokens=util.FEATURE_DIM,
            max_seq_len=500,
            attn_layers=Decoder(
                dim=self.kmer_dim,
                depth=n_decoder_layers,
                heads=decoder_attention_heads,
                attn_dropout=0.1,
                cross_attend=True,
            ),
        )
        normalmodel1.token_emb = Identity()
        normalmodel1.pos_emb = Identity()

        self.decoder0 = NoLossAutoregressiveWrapper(normalmodel0)
        self.decoder1 = NoLossAutoregressiveWrapper(normalmodel1)

        self.softmax = nn.LogSoftmax(dim=-1)
        self.elu = torch.nn.ELU()

    def encode(self, src):
        src = self.elu(self.fc1(src))
        src = self.pos_encoder(src)  # For 2D encoding we have to do this before flattening, right?
        src = src.flatten(start_dim=2)
        src = self.elu(self.fc2(src))
        mem = self.encoder(src)
        return mem

    def decode(self, mem, tgt, tgt_mask, tgt_key_padding_mask=None):
        mem_proj = self.converter(mem)
        tgt0 = self.tgt_pos_encoder(tgt[:, 0, :, :])
        tgt1 = self.tgt_pos_encoder(tgt[:, 1, :, :])

        # assert tgt_mask.shape[0] == tgt.shape[0], "Mask must have batch dimension equal to batch size"
        # if tgt_mask.dtype != torch.bool:
        #     tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_mask.shape[1], dtype=torch.bool).unsqueeze(0).expand(tgt.shape[0], -1, -1).to(self.device)
        # The magic of DataParallel mistakenly modifies the first dimension of the tgt mask when running on multi-GPU setups
        # This hack just forces it to be a square again
        # if tgt_mask.shape[0] != tgt_mask.shape[1]:
        #     tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_mask.shape[1]).to(self.device)
            #logger.info(f"Forcing tgt mask shapre to be {tgt_mask.shape}, input enc shape is: {mem.shape}")
        h0, _ = self.decoder0(tgt0, context=mem_proj)
        h1, _ = self.decoder1(tgt1, context=mem_proj)
        h0 = self.softmax(h0)
        h1 = self.softmax(h1)
        return torch.stack((h0, h1), dim=1)

    def forward(self, src, tgt, tgt_mask, tgt_key_padding_mask=None):
        mem = self.encode(src)
        result = self.decode(mem, tgt, tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        return result


class NoLossAutoregressiveWrapper(AutoregressiveWrapper):
    """
    Normal AutoregressiveWrapper computes loss as part of the forward pass?
    This just returns the logits and cache
    """

    def forward(self, x, return_outputs = False, **kwargs):
        seq, ignore_index, add_attn_z_loss = x.shape[1], self.ignore_index, self.add_attn_z_loss

        # inp, _ = x[:, :-1], x[:, 1:]
        # inp = torch.where(inp == ignore_index, self.pad_value, inp)

        if self.mask_prob > 0.:
            rand = torch.randn(x.shape, device=x.device)
            rand[:, 0] = -torch.finfo(rand.dtype).max  # first token should not be masked out
            num_mask = min(int(seq * self.mask_prob), seq - 1)
            indices = rand.topk(num_mask, dim=-1).indices
            mask = ~torch.zeros_like(x).scatter(1, indices, 1.).bool()
            kwargs.update(self_attn_kv_mask=mask)

        logits, cache = self.net(
            x,
            return_intermediates=True,
            return_attn_z_loss=add_attn_z_loss,
            **kwargs
        )

        return logits, cache

def main():
    # dec = Decoder(
    #     dim=128,
    #     depth=4,
    #     heads=8,
    #     attn_dropout=0.1,
    #     cross_attend=True,
    # )
    # normalmodel = TransformerWrapper(
    #     num_tokens=100,
    #     max_seq_len=128,
    #     attn_layers=dec,
    # )
    # normal_autregwrap = AutoregressiveWrapper(normalmodel)
    # context = torch.rand(3, 13, 128)
    # x = torch.randint(0, 100, (3,5))
    # result = normal_autregwrap.generate(x, seq_len=4, temperature=0, context=context)
    # print(result.shape)
    # return

    xmodel = XVarTransformer(read_depth=150,
                           feature_count=10,
                           kmer_dim=util.FEATURE_DIM,  # Number of possible kmers
                           n_encoder_layers=3,
                           n_decoder_layers=3,
                           embed_dim_factor=40,
                           encoder_attention_heads=8,
                           decoder_attention_heads=4,
                           d_ff=128,
                           device='cpu')

    nmodel = VarTransformer(read_depth=150,
                           feature_count=10,
                           kmer_dim=util.FEATURE_DIM,  # Number of possible kmers
                           n_encoder_layers=3,
                           n_decoder_layers=3,
                           embed_dim_factor=40,
                           encoder_attention_heads=8,
                           decoder_attention_heads=4,
                           d_ff=128,
                           device='cpu')

    x = torch.rand(3, 2, 150, 10)
    tgtkmers = F.one_hot(torch.randint(0, util.FEATURE_DIM, (3, 2, 150)), num_classes=util.FEATURE_DIM)

    tgt_mask = nn.Transformer.generate_square_subsequent_mask(150, dtype=torch.bool)
    y = xmodel(x, tgtkmers, tgt_mask)
    print(y.shape)
    print(y[0, 0, 0, 0:10])

    y2 = nmodel(x, tgtkmers, tgt_mask)
    print(y2.shape)
    print(y2[0, 0, 0, 0:10])


if __name__=="__main__":
    main()

