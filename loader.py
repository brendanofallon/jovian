
import logging
import random
import math
from pathlib import Path
from collections import defaultdict
import gzip
import io

import scipy.stats as stats
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import pysam

import bwasim
from bam import target_string_to_tensor, encode_with_ref, encode_and_downsample, ensure_dim
import sim
import util

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
            for src, tgt, vaftgt in encode_chunks(bam,
                                          self.reference,
                                          labels,
                                          batch_size,
                                          self.reads_per_pileup,
                                          self.samples_per_pos,
                                          max_to_load=1e9):
                yield src, tgt, vaftgt, None


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

    def __init__(self, device, datadir, src_prefix="src", tgt_prefix="tgt", vaftgt_prefix="vaftgt"):
        self.device = device
        self.datadir = Path(datadir)
        self.src_prefix = src_prefix
        self.tgt_prefix = tgt_prefix
        self.vaftgt_prefix = vaftgt_prefix
        self.pathpairs = self._find_files()
        self.cache = {} # Maps filenames to data, experimental
        self.max_cache_size = 2000 # ? No idea what makes sense here
        logger.info(f"Found {len(self.pathpairs)} batches in {datadir}")
        if not self.pathpairs:
            raise ValueError(f"Could not find any files in {datadir}")

    def retain_val_samples(self, fraction):
        """
        Remove a fraction of the samples from this loader and return them
        :returns : List of (src, tgt) PATH tuples (not loaded tensors)
        """
        num_to_retain = int(math.ceil(len(self.pathpairs) * fraction))
        val_samples = random.sample(self.pathpairs, num_to_retain)
        newdata = []
        for sample in self.pathpairs:
            if sample not in val_samples:
                newdata.append(sample)
        self.pathpairs = newdata
        return val_samples

    def _find_tgt(self, suffix, files):
        found = None
        for tgt in files:
            tsuf = tgt.name.split("_")[-1]
            if tsuf == suffix:
                if found:
                    raise ValueError(f"Uh oh, found multiple matches for suffix {suffix}!")
                found = tgt
        if found is None:
            raise ValueError(f"Could not find matching tgt file for {suffix}")
        return found

    def _find_files(self):
        """
        Match up all src / tgt files and store them as tuples in a list
        """
        allsrc = list(self.datadir.glob(self.src_prefix + "*"))
        alltgt = list(self.datadir.glob(self.tgt_prefix + "*"))
        allvaftgt = list(self.datadir.glob(self.vaftgt_prefix + "*"))
        pairs = []
        for src in allsrc:
            suffix = src.name.split("_")[-1]
            tgt = self._find_tgt(suffix, alltgt)
            vaftgt = self._find_tgt(suffix, allvaftgt)
            pairs.append((src, tgt, vaftgt))
        return pairs


    def item(self, path):
        """
        Return a tensor from the given path, doing a cache lookup first to see if we alrady have the data
        Cached data is gzipped, so decompress it when returning
        If the cache is smaller then max cache size, add the new data (still compressed) to the cache
        """
        if not str(path).endswith('.gz'):
            # If data is not compressed, don't try to store it in RAM. This is mostly for legacy support
            return torch.load(path, map_location=self.device)

        if path in self.cache:
            return torch.load(io.BytesIO(gzip.decompress(self.cache[path])), map_location=self.device)
        elif len(self.cache) < self.max_cache_size:
            with open(path, 'rb') as fh:
                data = fh.read()
                self.cache[path] = data
                return torch.load(io.BytesIO(gzip.decompress(data)), map_location=self.device)
        else:
            return util.unzip_load(path, self.device)

    def iter_once(self, batch_size):
        for src, tgt, vaftgt in self.pathpairs:
            yield self.item(src), self.item(tgt), self.item(vaftgt), None




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
        t = torch.zeros(tgt.shape[0], width - tgt.shape[1]).long()
        tgt = torch.cat((tgt, t), dim=1)
    else:
        start = src.shape[0] // 2 - width // 2
        src = src[start:start+width, :, :]
        tgt = tgt[:, start:start+width]

    return src, tgt


def assign_class_indexes(rows):
    """
    Iterate over every row in rows and create a list of class indexes for each element
    , where class index is currently row.vtype-row.status. So a class will be something like
    snv-TP or small_del-FP, and each class gets a unique number
    :returns : List of class indexes across rows
    """
    classcount = 0
    classes = defaultdict(int)
    idxs = []
    for row in rows:
        clz = f"{row.vtype}-{row.status}" # Could imagine putting VAF here too - maybe low / medium / high vaf?
        if clz in classes:
            idx = classes[clz]
        else:
            idx = classcount
            classcount += 1
            classes[clz] = idx
        idxs.append(idx)
    return idxs


def resample_classes(classes, rows, vals_per_class=100):
    """
    Randomly sample each class so that there are 'vals_per_class' members of each class
    then return a list of class assignments and the selected rows
    """
    byclass = defaultdict(list)
    for clz, row in zip(classes, rows):
        byclass[clz].append(row)

    result_rows = []
    result_classes = []
    for clz, classrows in byclass.items():
        # If there are MORE instances on this class than we want, grab a random sample of them
        if len(classrows) > vals_per_class:
            rows = random.sample(classrows, vals_per_class)
            result_rows.extend(rows)
            result_classes.extend([clz] * vals_per_class)
        else:
            # If there are FEWER instances of the class than we want, then loop over them
            # repeatedly - no random sampling since that might ignore a few of the precious few samples we do have
            for i in range(vals_per_class):
                result_classes.append(clz)
                idx = i % len(classrows)
                result_rows.append(classrows[idx])
    return result_classes, result_rows


def upsample_labels(rows, vals_per_class):
    """
    Generate class assignments for each element in rows (see assign_class_indexes),
     then create a new list of rows that is "upsampled" to normalize class frequencies
     The new list will be multiple times larger then the old list and will contain repeated
     elements of the low frequency classes
    :param rows: List of elements with vtype and status attributes (probably read in from a labels CSV file)
    :param vals_per_class: Number of instances to retain for each class
    :returns: List of rows, with less frequent rows included multiple times to help normalize frequencies
    """
    label_idxs = np.array(assign_class_indexes(rows))
    label_idxs, rows = resample_classes(label_idxs, rows, vals_per_class=vals_per_class)
    classes, counts = np.unique(label_idxs, return_counts=True)

    freqs = 1.0 / (np.min(1.0/counts) * counts)
    results = []
    for clz, row in zip(label_idxs, rows):
        freq = min(10, int(freqs[clz]))
        for reps in range(freq):
            results.append(row)

    random.shuffle(results)
    return results


def load_from_csv(bampath, refpath, csv, max_reads_per_aln, samples_per_pos, vals_per_class):
    """
    Generator for encoded pileups, reference, and alt sequences obtained from a
    alignment (BAM) and labels csv file. This performs some class normalized by upsampling / downsampling
    rows from the CSV based on "class", which is something like "snv-TP" or "ins-FP" (defined by assign_class_indexes
    function), may include VAF at some point?
    :param bampath: Path to input BAM file
    :param refpath: Path to ref genome
    :param csv: CSV file containing C/S/R/A, variant status (TP, FP, FN..) and type (SNV, del, ins, etc)
    :param max_reads_per_aln: Max reads to import for each pileup
    :param samples_per_pos: Max number of samples to create for a given position when read resampling
    :return: Generator
    """
    refgenome = pysam.FastaFile(refpath)
    bam = pysam.AlignmentFile(bampath)

    labels = [l for _, l in pd.read_csv(csv).iterrows()]
    upsampled_labels = upsample_labels(labels, vals_per_class=vals_per_class)
    logger.info(f"Will save {len(upsampled_labels)} with up to {samples_per_pos} samples per site from {csv}")
    for row in upsampled_labels:
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

                yield encoded, tgt, row.status, row.vtype, row.exp_vaf

        except Exception as ex:
            logger.warning(f"Error encoding position {row.chrom}:{row.pos} for bam: {bampath}, skipping it: {ex}")



def encode_chunks(bampath, refpath, csv, chunk_size, max_reads_per_aln, samples_per_pos, max_to_load=1e9):
    """
    Generator for creating batches of src, tgt (data and label) tensors from a CSV file

    :param bampath: Path to BAM file
    :param refpath: Path to human genome reference fasta file
    :param csv: Path to labels CSV file
    :param chunk_size: Number of samples / instances to include in one generated tensor
    :param max_reads_per_aln: Max read depth per tensor (defines index 2 of tensor)
    :param samples_per_pos: Randomly resample reads this many times, max, per position
    :returns : Generator of src, tgt tensor tuples
    """
    allsrc = []
    alltgt = []
    alltgtvaf = []
    count = 0
    vals_per_class = 500
    seq_len = 300
    logger.info(f"Creating new data loader from {bampath}")
    for enc, tgt, status, vtype, vaf in load_from_csv(bampath, refpath, csv, max_reads_per_aln=max_reads_per_aln, samples_per_pos=samples_per_pos, vals_per_class=vals_per_class):
        src, tgt = trim_pileuptensor(enc, tgt.unsqueeze(0), seq_len)
        src = ensure_dim(src, seq_len, max_reads_per_aln)
        assert src.shape[0] == seq_len, f"Src tensor #{count} had incorrect shape after trimming, found {src.shape[0]} but should be {seq_len}"
        assert tgt.shape[1] == seq_len, f"Tgt tensor #{count} had incorrect shape after trimming, found {tgt.shape[1]} but should be {seq_len}"
        allsrc.append(src)
        alltgt.append(tgt)
        alltgtvaf.append(vaf)
        count += 1
        if count % 100 == 0:
            logger.info(f"Loaded {count} tensors from {csv}")
        if count == max_to_load:
            logger.info(f"Stopping tensor load after {max_to_load}")
            yield torch.stack(allsrc), torch.stack(alltgt).long(), torch.tensor(alltgtvaf)
            break

        if len(allsrc) >= chunk_size:
            yield torch.stack(allsrc), torch.stack(alltgt).long(), torch.tensor(alltgtvaf)
            allsrc = []
            alltgt = []
            alltgtvaf = []

    if len(allsrc):
        yield torch.stack(allsrc), torch.stack(alltgt).long(), torch.tensor(alltgtvaf)
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
