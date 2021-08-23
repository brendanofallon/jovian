#!/usr/bin/env python

import torch
import torch.nn as nn
import logging
import pandas as pd
from collections import defaultdict
import pysam
from datetime import datetime

import yaml

from bam import target_string_to_tensor, encode_with_ref
from model import VarTransformer
import sim


logging.basicConfig(format='[%(asctime)s]  %(name)s  %(levelname)s  %(message)s',
                    datefmt='%m-%d %H:%M',
                    level=logging.INFO) # handlers=[RichHandler()])
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda:0") if hasattr(torch, 'cuda') and torch.cuda.is_available() else torch.device("cpu")


class ReadLoader:

    def __init__(self, src, tgt):
        self.src = src
        self.tgt = tgt

    def __len__(self):
        return self.src.shape[0]

    def iter_once(self, batch_size):
        offset = 0
        while offset < self.src.shape[0]:
            yield self.src[offset:offset + batch_size, :, :, :], self.tgt[offset:offset + batch_size, :, :]
            offset += batch_size


class SimLoader:

    def __init__(self):
        self.batches_in_epoch = 10

    def iter_once(self, batch_size):
        for i in range(self.batches_in_epoch):
            src, tgt = sim.make_mixed_batch(batch_size, seqlen=100, readsperbatch=100, readlength=70, error_rate=0.02, clip_prob=0.01)
            yield src.to(DEVICE), tgt.to(DEVICE)



def load_from_csv(bampath, refpath, csv, max_reads_per_aln):
    refgenome = pysam.FastaFile(refpath)
    bam = pysam.AlignmentFile(bampath)
    for i, row in pd.read_csv(csv).iterrows():
        if row.status == 'FP':
            encoded, refseq, altseq = encode_with_ref(str(row.chrom), row.pos, row.ref, row.ref, bam, refgenome, max_reads_per_aln)
        else:
            encoded, refseq, altseq = encode_with_ref(str(row.chrom), row.pos, row.ref, row.alt, bam, refgenome, max_reads_per_aln)

        minseqlen = min(len(refseq), len(altseq))
        reft = target_string_to_tensor(refseq[0:minseqlen])
        altt = target_string_to_tensor(altseq[0:minseqlen])
        tgt = torch.stack((reft, altt))
        yield encoded, tgt, row.status


def trim_pileuptensor(src, tgt, width):
    if src.shape[0] < width:
        z = torch.zeros(width - src.shape[0], src.shape[1], src.shape[2])
        src = torch.cat((src, z))
    else:
        src = src[0:width, :, :]
    if tgt.shape[1] < width:
        t = torch.zeros(tgt.shape[0], width - tgt.shape[1])
        tgt = torch.cat((tgt, t), dim=1)
    else:
        tgt = tgt[:, 0:width]

    return src, tgt


def make_loader(bampath, refpath, csv, max_to_load=1e9, max_reads_per_aln=100):
    allsrc = []
    alltgt = []
    count = 0
    seq_len = 150
    status_counter = defaultdict(int)
    for enc, tgt, status in load_from_csv(bampath, refpath, csv, max_reads_per_aln=max_reads_per_aln):
        status_counter[status] += 1
        src, tgt = trim_pileuptensor(enc, tgt, seq_len)
        assert src.shape[0] == seq_len, f"Src tensor #{count} had incorrect shape after trimming, found {src.shape[0]} but should be {seq_len}"
        assert tgt.shape[1] == seq_len, f"Tgt tensor #{count} had incorrect shape after trimming, found {tgt.shape[1]} but should be {seq_len}"
        allsrc.append(src)
        alltgt.append(tgt)
        count += 1
        if count % 100 == 0:
            logger.info(f"Loaded {count} tensors from {csv}")
        if count == max_to_load:
            logger.info(f"Stopping tensor load after {max_to_load}")
            break
    logger.info(f"Loaded {count} tensors from {csv}")
    return ReadLoader(torch.stack(allsrc).to(DEVICE), torch.stack(alltgt).long().to(DEVICE))



def sort_by_ref(seq, reads):
    """
    Sort read tensors by how closely they match the sequence seq.
    :param seq:
    :param reads: Tensor of encoded reads, with dimension [batch, pos, read, feature]
    :return: New tensor containing the same read tensors, but sorted
    """
    results = []
    for batch in range(reads.shape[0]):
        w = reads[batch, :, :, 0:4].sum(dim=-1)
        t = reads[batch, :, :, 0:4].argmax(dim=-1)
        matchsum = (t == (seq[batch, 0, :].repeat(reads.shape[2], 1).transpose(0,1)*w).long()).sum(dim=0)
        results.append(reads[batch, :, torch.argsort(matchsum), :])
    return torch.stack(results)


def train_epoch(model, optimizer, criterion, loader, batch_size):
    """
    Train for one epoch, which is defined by the loader but usually involves one pass over all input samples
    :param model: Model to train
    :param optimizer: Optimizer to update params
    :param criterion: Loss function
    :param loader: Provides training data
    :param batch_size:
    :return: Sum of losses over each batch, plus fraction of matching bases for ref and alt seq
    """
    epoch_loss_sum = 0
    for unsorted_src, tgt in loader.iter_once(batch_size):
        src = sort_by_ref(tgt, unsorted_src)
        optimizer.zero_grad()

        tgt_seq1 = tgt[:, 0, :]
        tgt_seq2 = tgt[:, 1, :]

        seq1preds, seq2preds = model(src.flatten(start_dim=2))

        loss = criterion(seq1preds.flatten(start_dim=0, end_dim=1), tgt_seq1.flatten())
        loss += 2 * criterion(seq2preds.flatten(start_dim=0, end_dim=1), tgt_seq2.flatten())

        with torch.no_grad():
            matches1 = (torch.argmax(seq1preds.flatten(start_dim=0, end_dim=1),
                                     dim=1) == tgt_seq1.flatten()).float().mean()
            matches2 = (torch.argmax(seq2preds.flatten(start_dim=0, end_dim=1),
                                     dim=1) == tgt_seq2.flatten()).float().mean()

        loss.backward(retain_graph=True)
        optimizer.step()
        epoch_loss_sum += loss.detach().item()

    return epoch_loss_sum, matches1.item(), matches2.item()


def train(epochs, dataloader, max_read_depth=25, feats_per_read=6, init_learning_rate=0.001, statedict=None, model_dest=None):
    in_dim = max_read_depth * feats_per_read
    model = VarTransformer(in_dim=in_dim, out_dim=4, nhead=5, d_hid=200, n_encoder_layers=2).to(DEVICE)
    if statedict is not None:
        logger.info(f"Initializing model with state dict {statedict}")
        model.load_state_dict(torch.load(statedict))
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=init_learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    batch_size = 15
    try:
        for epoch in range(epochs):
            starttime = datetime.now()
            loss, refmatch, altmatch = train_epoch(model, optimizer, criterion, dataloader, batch_size=batch_size)
            elapsed = datetime.now() - starttime
            logger.info(f"Epoch {epoch} Secs: {elapsed.total_seconds():.2f} loss: {loss:.4f} Ref match: {refmatch:.4f}  altmatch: {altmatch:.4f} ")
            scheduler.step()

        logger.info(f"Training completed after {epoch} epochs")
    except KeyboardInterrupt:
        if model_dest is not None:
            logger.info(f"Saving model state dict to {model_dest}")
            torch.save(model.to('cpu').state_dict(), model_dest)


def load_train_conf(confyaml):
    logger.info(f"Loading configuration from {confyaml}")
    conf = yaml.safe_load(open(confyaml).read())
    assert 'reference' in conf, "Expected 'reference' entry in training configuration"
    assert 'data' in conf, "Expected 'data' entry in training configuration"
    return conf


def main(confyaml="train.yaml"):
    logger.info(f"Found torch device: {DEVICE}")
    conf = load_train_conf(confyaml)
    #loader = make_loader(conf['data'][0]['bam'],
    #                      conf['reference'],
    #                      conf['data'][0]['labels'],
    #                      max_to_load=1000,
    #                      max_reads_per_aln=100)
    loader = SimLoader()
    train(50, loader, max_read_depth=100, statedict="saved.model", model_dest="saved.model")


if __name__ == "__main__":
    main()
