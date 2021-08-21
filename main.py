import numpy as np
import torch
import torch.nn as nn
from rich.logging import RichHandler
from torch import optim
import torch.nn.functional as F
import math
import copy
import logging
import pandas as pd
from collections import defaultdict
import pysam

from bam import target_string_to_tensor, encode_with_ref
from model import VarTransformer


logging.basicConfig(format='%(message)s', level=logging.INFO, handlers=[RichHandler()])
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
            yield self.src[offset:offset + batch_size, : :, :], self.tgt[offset:offset + batch_size, :, :]
            offset += batch_size


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
    if src.shape[0] >= width:
        return src[0:width, :, :], tgt[:, 0:width]
    elif src.shape[0] < width:
        z = torch.zeros(width - src.shape[0], src.shape[1], src.shape[2])
        t = torch.zeros(tgt.shape[0], width - tgt.shape[1])
        return torch.cat((src, z)), torch.cat((tgt, t))


def make_loader(bampath, refpath, csv, max_to_load=1e9, max_reads_per_aln=100):
    allsrc = []
    alltgt = []
    count = 0
    status_counter = defaultdict(int)
    for enc, tgt, status in load_from_csv(bampath, refpath, csv, max_reads_per_aln=max_reads_per_aln):
        status_counter[status] += 1
        src, tgt = trim_pileuptensor(enc, tgt, 150)
        allsrc.append(src)
        alltgt.append(tgt)
        count += 1
        if count % 100 == 0:
            logger.info(f"Loaded {count} tensors from {csv}")
        if count == max_to_load:
            logger.info(f"Stopping tensor load after {max_to_load}")
            break
    logger.info(f"Loaded {count} tensors from {csv}")
    return ReadLoader(torch.stack(allsrc), torch.stack(alltgt))



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


def train(epochs, dataloader, max_read_depth=25, feats_per_read=6, init_learning_rate=0.001, statedict=None):
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
    for epoch in range(epochs):
        loss, refmatch, altmatch = train_epoch(model, optimizer, criterion, dataloader, batch_size=batch_size)
        logger.info(f"Epoch {epoch} loss: {loss:.4f} Ref match: {refmatch:.4f}  altmatch: {altmatch:.4f}")
        scheduler.step()


def main():
    logger.info(f"Found torch device: {DEVICE}")
    loader = make_loader("/Volumes/Share/genomics/onccn_15_car641/bam/roi.bam",
                         "/Volumes/Share/genomics/reference/human_g1k_v37_decoy_phiXAdaptr.fasta",
                         "/Volumes/Share/genomics/onccn_15_car641/tp_fp_fn.csv",
                         max_to_load=100,
                         max_reads_per_aln=100)
    train(5, loader, max_read_depth=100)


if __name__ == "__main__":
    main()