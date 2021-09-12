
import logging
import random
from pathlib import Path
from collections import defaultdict

import scipy.stats as stats
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import pysam

import bwasim
from bam import target_string_to_tensor, encode_with_ref, encode_and_downsample
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
            yield self.src[offset:offset + batch_size, :, :, :].to(self.device), self.tgt[offset:offset + batch_size, :, :].to(self.device), None, None
            offset += batch_size


class LazyLoader:
    """
    A loader that doesn't load anything until iter_once() is called, and doesn't save anything in memory
    This will be pretty slow but good if we can't fit all of the data into memory
    Useful for 'pre-gen' where we just want to iterate over everything once and save it to a file
    """

    def __init__(self, bamlabelpairs, reference, reads_per_pileup, samples_per_pos):
        self.bamlabels = bamlabelpairs
        self.reference = reference
        self.reads_per_pileup = reads_per_pileup
        self.samples_per_pos = samples_per_pos

    def iter_once(self, batch_size):
        for bam, labels in self.bamlabels:
            logger.info(f"Encoding tensors from {labels}")
            for src, tgt in encode_chunks(bam,
                                          self.reference,
                                          labels,
                                          batch_size,
                                          self.reads_per_pileup,
                                          self.samples_per_pos,
                                          max_to_load=1e9):
                yield src, tgt, None, None


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


def trim_pileuptensor(src, tgt, width):
    """
    Trim or zero-pad the sequence dimension of src and target (first dimension of src, second of target)
    so they're equal to width
    :param src: Data tensor of shape [sequence, read, features]
    :param tgt: Target tensor of shape [haplotype, sequence]
    :param width: Size to trim to
    """
    assert src.shape[0] == tgt.shape[-1], f"Unequal src and target lengths ({src.shape[0]} vs. {tgt.shape[-1]}), not sure how to deal with this :("
    if src.shape[0] < width:
        z = torch.zeros(width - src.shape[0], src.shape[1], src.shape[2])
        src = torch.cat((src, z))
        t = torch.zeros(tgt.shape[0], width - tgt.shape[1])
        tgt = torch.cat((tgt, t), dim=1)
    else:
        start = src.shape[0] // 2 - width // 2
        src = src[start:start+width, :, :]
        tgt = tgt[:, start:start+width]

    return src, tgt


def load_from_csv(bampath, refpath, csv, max_reads_per_aln, samples_per_pos):
    """
    Generator for encoded pileups, reference, and alt sequences obtained from a
    alignment (BAM) and labels csv file
    :param bampath: Path to input BAM file
    :param refpath: Path to ref genome
    :param csv: CSV file containing CSRA, variant status (TP, FP, FN..) and type (SNV, del, ins, etc)
    :param max_reads_per_aln: Max reads to import for each pileup
    :param samples_per_pos: Number of samples to create for a given position
    :return: Generator
    """
    refgenome = pysam.FastaFile(refpath)
    bam = pysam.AlignmentFile(bampath)
    num_ok_errors = 20

    for i, row in pd.read_csv(csv).iterrows():
        if row.status == 'FP':
            altseq = row.ref
        else:
            altseq = row.alt

        try:
            for encoded, refseq, altseq in encode_and_downsample(str(row.chrom),
                                                                 row.pos,
                                                                 row.ref,
                                                                 altseq,
                                                                 bam,
                                                                 refgenome,
                                                                 max_reads_per_aln,
                                                                 samples_per_pos):


                minseqlen = min(len(refseq), len(altseq))
                tgt = target_string_to_tensor(altseq[0:minseqlen])
                if minseqlen != encoded.shape[0]:
                    encoded = encoded[0:minseqlen, :, :]

                yield encoded, tgt, row.status, row.vtype

        except Exception as ex:
            logger.warning(f"Error encoding position {row.chrom}:{row.pos} for bam: {bampath}, skipping it: {ex}")
            num_ok_errors -= 1
            if num_ok_errors < 0:
                raise ValueError(f"Too many errors for {bampath}, quitting!")
            else:
                continue



def encode_chunks(bampath, refpath, csv, chunk_size, max_reads_per_aln, samples_per_pos, max_to_load=1e9):
    allsrc = []
    alltgt = []
    count = 0
    seq_len = 300
    logger.info(f"Creating new data loader from {bampath}")
    for enc, tgt, status, vtype in load_from_csv(bampath, refpath, csv, max_reads_per_aln=max_reads_per_aln, samples_per_pos=samples_per_pos):
        src, tgt = trim_pileuptensor(enc, tgt.unsqueeze(0), seq_len)
        src = ensure_dim(src, seqlen, max_reads_per_aln)
        assert src.shape[0] == seq_len, f"Src tensor #{count} had incorrect shape after trimming, found {src.shape[0]} but should be {seq_len}"
        assert tgt.shape[1] == seq_len, f"Tgt tensor #{count} had incorrect shape after trimming, found {tgt.shape[1]} but should be {seq_len}"
        allsrc.append(src)
        alltgt.append(tgt)
        count += 1
        if count % 100 == 0:
            logger.info(f"Loaded {count} tensors from {csv}")
        if count == max_to_load:
            logger.info(f"Stopping tensor load after {max_to_load}")
            yield torch.stack(allsrc), torch.stack(alltgt).long()
            break

        if len(allsrc) >= chunk_size:
            yield torch.stack(allsrc), torch.stack(alltgt).long()
            allsrc = []
            alltgt = []

    if len(allsrc):
        yield torch.stack(allsrc), torch.stack(alltgt).long()
    logger.info(f"Done loading {count} tensors from {csv}")



def make_loader(bampath, refpath, csv, max_reads_per_aln, samples_per_pos, max_to_load=1e9):
    allsrc = []
    alltgt = []
    count = 0
    seq_len = 150
    logger.info(f"Creating new data loader from {bampath}")
    counter = defaultdict(int)
    classes = []
    for enc, tgt, status, vtype in load_from_csv(bampath, refpath, csv, max_reads_per_aln=max_reads_per_aln, samples_per_pos=samples_per_pos):
        label_class = "-".join((status, vtype))
        classes.append(label_class)
        counter[label_class] += 1
        src, tgt = trim_pileuptensor(enc, tgt.unsqueeze(0), seq_len)
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
    logger.info("Class breakdown is: " + " ".join(f"{k}={v}" for k,v in counter.items()))
    weights = np.array([1.0 / counter[c] for c in classes])
    return WeightedLoader(torch.stack(allsrc), torch.stack(alltgt).long(), weights, DEVICE)



def make_multiloader(inputs, refpath, threads, max_to_load, max_reads_per_aln, samples_per_pos):
    """
    Create multiple ReadLoaders in parallel for each element in Inputs
    :param inputs: List of (BAM path, labels csv) tuples
    :param threads: Number of threads to use
    :param max_reads_per_aln: Max number of reads for each pileup
    :return: List of loaders
    """
    results = []
    if len(inputs) == 1:
        logger.info(
            f"Loading training data for {len(inputs)} sample with 1 processe (max to load = {max_to_load})")
        bam = inputs[0][0]
        labels_csv = inputs[0][1]
        return make_loader(bam, refpath, labels_csv, max_to_load=max_to_load, max_reads_per_aln=max_reads_per_aln, samples_per_pos=samples_per_pos)
    else:
        logger.info(f"Loading training data for {len(inputs)} samples with {threads} processes (max to load = {max_to_load})")
        with mp.Pool(processes=threads) as pool:
            for bam, labels_csv in inputs:
                result = pool.apply_async(make_loader, (bam, refpath, labels_csv, max_reads_per_aln, samples_per_pos, max_to_load))
                results.append(result)
            pool.close()
            return MultiLoader([l.get(timeout=2*60*60) for l in results])