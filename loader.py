
import logging
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import pysam

from bam import target_string_to_tensor, encode_with_ref
import sim

logger = logging.getLogger(__name__)


class ReadLoader:
    """
    The simplest loader, this one just has a src and tgt tensor and iterates over them
    Assumes first index in each the batch index
    """

    def __init__(self, src, tgt, device):
        assert src.shape[0] == tgt.shape[0]
        self.src = src
        self.tgt = tgt
        self.device = device

    def __len__(self):
        return self.src.shape[0]

    def iter_once(self, batch_size):
        offset = 0
        while offset < self.src.shape[0]:
            yield self.src[offset:offset + batch_size, :, :, :].to(self.device), self.tgt[offset:offset + batch_size, :, :].to(self.device)
            offset += batch_size


class WeightedLoader:

    def __init__(self, src, tgt, weights, device):
        assert len(weights) == src.shape[0]
        assert src.shape[0] == tgt.shape[0]
        self.src = src
        self.tgt = tgt
        weights = np.array(weights)
        self.weights = weights / weights.sum()
        self.device = device

    def __len__(self):
        return self.src.shape[0]

    def iter_once(self, batch_size):
        iterations = self.src.shape[0] // batch_size
        count = 0
        while count < iterations:
            count += 1
            idx = np.random.choice(range(self.src.shape[0]), size=batch_size, replace=True, p=self.weights)
            yield self.src[idx, :, :, :].to(self.device), self.tgt[idx, :, :].to(self.device)



class MultiLoader:
    """
    Combines multiple loaders into a single loader
    """

    def __init__(self, loaders):
        self.loaders = loaders

    def iter_once(self, batch_size):
        for loader in self.loaders:
            for src, tgt in loader.iter_once(batch_size):
                yield src, tgt


class SimLoader:

    def __init__(self, device, seqlen, readsperbatch, readlength, error_rate, clip_prob):
        self.batches_in_epoch = 10
        self.device = device
        self.seqlen = seqlen
        self.readsperbatch = readsperbatch
        self.readlength = readlength
        self.error_rate = error_rate
        self.clip_prob = clip_prob

    def iter_once(self, batch_size):
        for i in range(self.batches_in_epoch):
            src, tgt = sim.make_mixed_batch(batch_size,
                                            seqlen=self.seqlen,
                                            readsperbatch=self.readsperbatch,
                                            readlength=self.readlength,
                                            error_rate=self.error_rate,
                                            clip_prob=self.clip_prob)
            yield src.to(self.device), tgt.to(self.device)



def load_from_csv(bampath, refpath, csv, max_reads_per_aln):
    """
    Generator for encoded pileups, reference, and alt sequences obtained from a
    alignment (BAM) and labels csv file
    :param bampath: Path to input BAM file
    :param refpath: Path to ref genome
    :param csv: CSV file containing CSRA, variant status (TP, FP, FN..) and type (SNV, del, ins, etc)
    :param max_reads_per_aln: Max reads to import for each pileup
    :return:
    """
    refgenome = pysam.FastaFile(refpath)
    bam = pysam.AlignmentFile(bampath)
    num_ok_errors = 20

    for i, row in pd.read_csv(csv).iterrows():
        try:
            if row.status == 'FP':
                encoded, refseq, altseq = encode_with_ref(str(row.chrom), row.pos, row.ref, row.ref, bam, refgenome, max_reads_per_aln)
            else:
                encoded, refseq, altseq = encode_with_ref(str(row.chrom), row.pos, row.ref, row.alt, bam, refgenome, max_reads_per_aln)

            minseqlen = min(len(refseq), len(altseq))
            reft = target_string_to_tensor(refseq[0:minseqlen])
            altt = target_string_to_tensor(altseq[0:minseqlen])
        except Exception as ex:
            logger.warning(f"Error encoding position {row.chrom}:{row.pos} for bam: {bampath}, skipping it: {ex}")
            num_ok_errors -= 1
            if num_ok_errors < 0:
                raise ValueError(f"Too many errors for {bampath}, quitting!")
            else:
                continue

        if minseqlen != encoded.shape[0]:
            encoded = encoded[0:minseqlen, :, :]
        tgt = torch.stack((reft, altt))
        yield encoded, tgt, row.status, row.vtype

# t = torch.arange(100).unsqueeze(1).unsqueeze(1).unsqueeze(1)
# weights = 100 - np.arange(100)
# wl = WeightedLoader(t, t, weights, 'cpu')
#
# for src, tgt in wl.iter_once(10):
#     print(src)
