
import logging
import scipy.stats as stats
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import pysam

import bwasim
from bam import target_string_to_tensor, encode_with_ref
import sim

logger = logging.getLogger(__name__)


class ReadLoader:
    """
    The simplest loader, this one just has a src and tgt tensor and iterates over them
    Assumes first index in each is the batch index
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
    """
    A fancier loader that has a sampling weight associated with each element
    Elements with higher weights get sampled more frequently
    """

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


class PregenLoader:

    def __init__(self, device, datadir, src_prefix="src", tgt_prefix="tgt"):
        self.device = device
        self.datadir = Path(datadir)
        self.src_prefix = src_prefix
        self.tgt_prefix = tgt_prefix
        self.pathpairs = self._find_files()

    def _find_files(self):
        allsrc = list(self.datadir.glob(self.src_prefix + "*"))
        alltgt = list(self.datadir.glob(self.tgt_prefix + "*"))
        pairs = []
        for src in allsrc:
            suffix = src.name.split("_")[-1]
            found = False
            for tgt in alltgt:
                tsuf = tgt.name.split("_")[-1]
                if tsuf == suffix:
                    if found:
                        raise ValueError(f"Uh oh, found multiple matches for suffix {suffix}!")
                    found = True
                    pairs.append((src, tgt))
            if not found:
                raise ValueError(f"Uh oh, didn't find any match for suffix {suffix}!")
        return pairs

    def iter_once(self, batch_size):
        for src, tgt in self.pathpairs:
            yield torch.load(src, map_location=self.device), torch.load(tgt, map_location=self.device), None, None



class MultiLoader:
    """
    Combines multiple loaders into a single loader
    """

    def __init__(self, loaders):
        self.loaders = loaders

    def iter_once(self, batch_size):
        for loader in self.loaders:
            for src, tgt in loader.iter_once(batch_size):
                yield src, tgt, None, None


class BWASimLoader:

    def __init__(self, device, regions, refpath, readsperpileup, readlength, error_rate, clip_prob):
        self.batches_in_epoch = 10
        self.regions = bwasim.load_regions(regions)
        self.refpath = refpath
        self.device = device
        self.readsperpileup = readsperpileup
        self.readlength = readlength
        self.error_rate = error_rate
        self.clip_prob = clip_prob
        self.sim_data = []


    def iter_once(self, batch_size):
        if len(self.sim_data):
            self.sim_data = self.sim_data[1:] # Trim off oldest batch - but only once per epoch

        for i in range(self.batches_in_epoch):
            if len(self.sim_data) <= i:
                src, tgt, vaftgt, altmask = bwasim.make_batch(batch_size,
                                                              self.regions,
                                                              self.refpath,
                                                              numreads=self.readsperpileup,
                                                              readlength=self.readlength,
                                                              vaf_func=bwasim.betavaf,
                                                              var_funcs=None,
                                                              error_rate=self.error_rate,
                                                              clip_prob=self.clip_prob)

                self.sim_data.append((src, tgt, vaftgt, altmask))
            src, tgt, vaftgt, altmask = self.sim_data[i]
            yield src.to(self.device), tgt.to(self.device), vaftgt.to(self.device), altmask.to(self.device)


class SimLoader:

    def __init__(self, device, seqlen, readsperbatch, readlength, error_rate, clip_prob):
        self.batches_in_epoch = 10
        self.device = device
        self.seqlen = seqlen
        self.readsperbatch = readsperbatch
        self.readlength = readlength
        self.error_rate = error_rate
        self.clip_prob = clip_prob
        self.sim_data = []


    def iter_once(self, batch_size):
        if len(self.sim_data):
            self.sim_data = self.sim_data[1:] # Trim off oldest batch - but only once per epoch

        for i in range(self.batches_in_epoch):
            if len(self.sim_data) <= i:
                src, tgt, vaftgt, altmask = sim.make_mixed_batch(batch_size,
                                            seqlen=self.seqlen,
                                            readsperbatch=self.readsperbatch,
                                            readlength=self.readlength,
                                            error_rate=self.error_rate,
                                            clip_prob=self.clip_prob)
                self.sim_data.append((src, tgt, vaftgt, altmask))
            src, tgt, vaftgt, altmask = self.sim_data[-1]
            yield src.to(self.device), tgt.to(self.device), vaftgt.to(self.device), altmask.to(self.device)



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
