
import logging
logger = logging.getLogger(__name__)

import random
import math
from pathlib import Path
from collections import defaultdict
from itertools import chain
import gzip
import lz4.frame
from datetime import datetime


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

    def __init__(self, bam, csv, reference, reads_per_pileup, samples_per_pos, vals_per_class):
        self.bam = bam
        self.csv = csv
        self.reference = reference
        self.reads_per_pileup = reads_per_pileup
        self.samples_per_pos = samples_per_pos
        self.vals_per_class = vals_per_class

    def iter_once(self, batch_size):
        logger.info(f"Encoding tensors from {self.bam} and {self.csv}")
        for src, tgt, vaftgt, varsinfo, posflag in encode_chunks(self.bam,
                                          self.reference,
                                          self.csv,
                                          batch_size,
                                          self.reads_per_pileup,
                                          self.samples_per_pos,
                                          self.vals_per_class,
                                          max_to_load=1e9):
                yield src, tgt, vaftgt, varsinfo, posflag


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


def decomp_single(path):
    if str(path).endswith('.lz4'):
        with open(path, 'rb') as fh:
            return torch.load(io.BytesIO(lz4.frame.decompress(fh.read())), map_location='cpu')
    else:
        return torch.load(path, map_location='cpu')


def decompress_multi(paths, threads):
    """
    Read & decompress all of the items in the paths with lz4, then load them into Tensors
    but keep them on the CPU
    :returns : List of Tensors (all on CPU)
    """
    start = datetime.now()
    with mp.Pool(threads) as pool:
        result = pool.map(decomp_single, paths)
        elapsed = datetime.now() - start
        logger.info(f"Decompressed {len(result)} items in {elapsed.total_seconds():.3f} seconds ({elapsed.total_seconds()/len(result):.3f} secs per item)")
        return result


class PregenLoader:

    def __init__(self, device, datadir, threads, max_decomped_batches=10, src_prefix="src", tgt_prefix="tgt", vaftgt_prefix="vaftgt"):
        """
        Create a new loader that reads tensors from a 'pre-gen' directory
        :param device: torch.device
        :param datadir: Directory to read data from
        :param threads: Max threads to use for decompressing data
        :param max_decomped_batches: Maximum number of batches to decompress at once. Increase this on machines with tons of RAM
        """
        self.device = device
        self.datadir = Path(datadir)
        self.src_prefix = src_prefix
        self.tgt_prefix = tgt_prefix
        self.vaftgt_prefix = vaftgt_prefix
        self.posflag_prefix = "posflag"
        self.pathpairs = util.find_files(self.datadir, self.src_prefix, self.tgt_prefix, self.vaftgt_prefix, self.posflag_prefix)
        self.threads = threads
        self.max_decomped = max_decomped_batches # Max number of decompressed items to store at once - increasing this uses more memory, but allows increased parallelization
        logger.info(f"Creating PreGen data loader with {self.threads} threads")
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
        logger.info(f"Number of batches left for training: {len(self.pathpairs)}")
        return val_samples


    def iter_once(self, batch_size):
        """
        Make one pass over the training data, in this case all of the files in the 'data dir'
        Training data is compressed and on disk, which makes it slow. To increase performance we
        load / decomp several batches in parallel, then train over each decompressed batch
        sequentially
        :param batch_size: The number of samples in a minibatch.
        """
        src, tgt, vaftgt, posflag = [], [], [], []
                    
        for i in range(0, len(self.pathpairs), self.max_decomped):
            paths = self.pathpairs[i:i+self.max_decomped]
            decomped = decompress_multi(chain.from_iterable(paths), self.threads)

            for j in range(0, len(decomped), 4):
                src.append(decomped[j])
                tgt.append(decomped[j+1])
                vaftgt.append(decomped[j+2])
                posflag.append(decomped[j+3])

            total_size = sum([s.shape[0] for s in src])
            if total_size < batch_size:
                # We need to decompress more data to make a batch
                continue

            # Make a big tensor.
            src_t = torch.cat(src, dim=0)
            tgt_t = torch.cat(tgt, dim=0)
            vaftgt_t = torch.cat(vaftgt, dim=0)
            posflag_t = torch.cat(posflag, dim=0)

            nbatch = total_size // batch_size
            remain = total_size % batch_size

            # Slice the big tensors for batches
            for n in range(0, nbatch):
                start = n * batch_size
                end = (n + 1) * batch_size
                yield (
                    src_t[start:end].to(self.device).float(),
                    tgt_t[start:end].to(self.device).long(),
                    vaftgt_t[start:end].to(self.device),
                    posflag_t[start:end].to(self.device),
                ) 

            if remain:
                # The remaining data points will be in next batch. 
                src = [src_t[nbatch * batch_size:]]
                tgt = [tgt_t[nbatch * batch_size:]]
                vaftgt = [vaftgt_t[nbatch * batch_size:]]
                posflag = [posflag_t[nbatch * batch_size:]]
            else:
                src, tgt, vaftgt, posflag = [], [], [], []

        if len(src) > 0:
            # We need to yield the last batch.
            yield (
                torch.cat(src, dim=0).to(self.device).float(),
                torch.cat(tgt, dim=0).to(self.device).long(),
                torch.cat(vaftgt, dim=0).to(self.device),
                torch.cat(posflag, dim=0).to(self.device),
            )


class ShorteningLoader:
    """
    This loader shortens the sequence dimension (dimension 1 of src, 2 of tgt) to seq_len
    It should be used to wrap another loader that actually does the loading
    For instance:

        loader = ShorteningLoader(PregenLoader(...), seq_len=100)

    """
    def __init__(self, wrapped_loader, seq_len):
        self.wrapped_loader = wrapped_loader
        self.seq_len = seq_len

    def iter_once(self, batch_size):
        for src, tgt, vaftgt, _ in self.wrapped_loader.iter_once(batch_size):
            start = src.shape[1] // 2 - self.seq_len // 2
            end = src.shape[1] // 2 + self.seq_len //2
            yield src[:, start:end, :, :], tgt[:, :, start:end],  vaftgt, None


class ShufflingLoader:
    """
    This loader shuffles the reads (dimension 2 of src), excluding the 0th element i.e. ref read.
    It should be used to wrap another loader that actually does the loading
    For instance:

        loader = ShufflingLoader(PregenLoader(...))

    """
    def __init__(self, wrapped_loader):
        self.wrapped_loader = wrapped_loader

    def iter_once(self, batch_size):
        for src, tgt, vaftgt, _ in self.wrapped_loader.iter_once(batch_size):
            src = src[:, :, torch.randperm(src.shape[2])[1:], :]
            yield src, tgt, vaftgt, None


class DownsamplingLoader:
    """
    This loader downsamples the reads (dimension 2 of src) by setting entire read to 0,
    excluding the 0th element i.e. ref read.
    The number of reads to be downsampled is obtained from the binomial distribution
    where each reads has a prob p=0.01 of getting dropped out.
    This loader should be used by wrapping another loader that actually does the loading
    For instance:

        loader = DownsamplingLoader(PregenLoader(...), prob_of_read_being_dropped=0.01)

    """
    def __init__(self, wrapped_loader, prob_of_read_being_dropped):
        self.wrapped_loader = wrapped_loader
        self.prob_of_read_being_dropped = prob_of_read_being_dropped


    def iter_once(self, batch_size):
        for src, tgt, vaftgt, _ in self.wrapped_loader.iter_once(batch_size):
            num_reads_to_drop = stats.binom(n=src.shape[2], p=self.prob_of_read_being_dropped).rvs(src.shape[0])
            for idx in range(src.shape[0]):
                logger.debug(f"{num_reads_to_drop[idx]} reads to be dropped out of total {src.shape[2]} reads in batch: {idx}")
                read_index_to_drop = random.sample(list(range(src.shape[2])[1:]), num_reads_to_drop[idx])
                logger.debug(f"Reads at batch id {idx} and read index {read_index_to_drop} are being dropped, excluding ref read at 0th position.")
                src[idx, :, read_index_to_drop, :] = 0
            yield src, tgt, vaftgt, None


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


def trim_pileuptensor(src, tgt, posflag, width):
    """
    Trim or zero-pad the sequence dimension of src and target (first dimension of src, second of target)
    so they're equal to width
    :param src: Data tensor of shape [sequence, read, features]
    :param tgt: Target tensor of shape [haplotype, sequence]
    :param width: Size to trim to
    """
    assert src.shape[0] == tgt.shape[-1], f"Unequal src and target lengths ({src.shape[0]} vs. {tgt.shape[-1]}), not sure how to deal with this :("
    assert tgt.shape == posflag.shape
    if src.shape[0] < width:
        z = torch.zeros(width - src.shape[0], src.shape[1], src.shape[2], dtype=src.dtype)
        src = torch.cat((src, z))
        t = torch.zeros(tgt.shape[0], width - tgt.shape[1]).long()
        tgt = torch.cat((tgt, t), dim=1)
        posflag = torch.cat((posflag, torch.zeros(tgt.shape[0], width - posflag.shape[1]).long()), dim=1)
    else:
        start = src.shape[0] // 2 - width // 2
        src = src[start:start+width, :, :]
        tgt = tgt[:, start:start+width]
        posflag = posflag[start:start+width]

    return src, tgt, posflag


def assign_class_indexes(rows):
    """
    Iterate over every row in rows and create a list of class indexes for each element
    , where class index is currently row.vtype-row.status. So a class will be something like
    snv-TP or small_del-FP, and each class gets a unique number
    :returns : 1. List of class indexes across rows.
               2. A dictionary keyed by class index and valued by class names.
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
    class_names = {v: k for k, v in classes.items()}
    return idxs, class_names


def resample_classes(classes, class_names, rows, vals_per_class=100):
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
        logger.info(f"Variant class {class_names[clz]} contains {len(classrows)} variants")
        if len(classrows) > vals_per_class:
            logger.info(f"Randomly sampling {vals_per_class} variants for variant class {class_names[clz]}")
            rows = random.sample(classrows, vals_per_class)
            result_rows.extend(rows)
            result_classes.extend([clz] * vals_per_class)
        else:
            # If there are FEWER instances of the class than we want, then loop over them
            # repeatedly - no random sampling since that might ignore a few of the precious few samples we do have
            logger.info(f"Loop over {len(classrows)} variants repeatedly to generate {vals_per_class} {class_names[clz]} variants")
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
    label_idxs, label_names = assign_class_indexes(rows)
    label_idxs, rows = resample_classes(
        np.array(label_idxs), label_names, rows, vals_per_class=vals_per_class
    )
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

    labels = [l for _, l in pd.read_csv(csv, dtype={'chrom': str, 'filters': str}).iterrows()]
    upsampled_labels = upsample_labels(labels, vals_per_class=vals_per_class)
    logger.info(f"Will save {len(upsampled_labels)} with up to {samples_per_pos} samples per site from {csv}")
    for row in upsampled_labels:
        if row.status == 'FP':
            altseq = row.ref
        else:
            altseq = row.alt

        try:
            for encoded, refseq, altseq, posflag in encode_and_downsample(str(row.chrom),
                                                                 row.pos,
                                                                 row.ref,
                                                                 altseq,
                                                                 bam,
                                                                 refgenome,
                                                                 max_reads_per_aln,
                                                                 samples_per_pos):


                minseqlen = min(len(refseq), len(altseq))
                tgt = target_string_to_tensor(altseq[0:minseqlen])
                posflag = posflag[0:minseqlen]
                if minseqlen != encoded.shape[0]:
                    encoded = encoded[0:minseqlen, :, :]

                yield encoded, tgt, row, posflag

        except Exception as ex:
            logger.warning(f"Error encoding position {row.chrom}:{row.pos} for bam: {bampath}, skipping it: {ex}")



def encode_chunks(bampath, refpath, csv, chunk_size, max_reads_per_aln, samples_per_pos, vals_per_class, max_to_load=1e9):
    """
    Generator for creating batches of src, tgt (data and label) tensors from a CSV file

    :param bampath: Path to BAM file
    :param refpath: Path to human genome reference fasta file
    :param csv: Path to labels CSV file
    :param chunk_size: Number of samples / instances to include in one generated tensor
    :param max_reads_per_aln: Max read depth per tensor (defines index 2 of tensor)
    :param samples_per_pos: Randomly resample reads this many times, max, per position
    :param vals_per_class: 
    :returns : Generator of src, tgt tensor tuples
    """
    allsrc = []
    alltgt = []
    alltgtvaf = []
    allposflag = []
    varsinfo = []
    count = 0
    seq_len = 300
    logger.info(f"Creating new data loader from {bampath}, vals_per_class: {vals_per_class}")
    for enc, tgt, row, posflag in load_from_csv(bampath, refpath, csv, max_reads_per_aln=max_reads_per_aln, samples_per_pos=samples_per_pos, vals_per_class=vals_per_class):
        status, vtype, vaf = row.status, row.vtype, row.exp_vaf
        src, tgt, posflag = trim_pileuptensor(enc, tgt.unsqueeze(0), posflag.unsqueeze(0), seq_len)
        src = ensure_dim(src, seq_len, max_reads_per_aln)
        assert src.shape[0] == seq_len, f"Src tensor #{count} had incorrect shape after trimming, found {src.shape[0]} but should be {seq_len}"
        assert tgt.shape[1] == seq_len, f"Tgt tensor #{count} had incorrect shape after trimming, found {tgt.shape[1]} but should be {seq_len}"
        assert posflag.shape[1] == seq_len, f"position flag tensor did not have length equal to seq len"
        allsrc.append(src)
        alltgt.append(tgt)
        alltgtvaf.append(vaf)
        allposflag.append(posflag)
        varsinfo.append([f"{row.chrom}", f"{row.pos}", row.ref, row.alt, f"{vaf}"])
        count += 1
        if count % 100 == 0:
            logger.info(f"Loaded {count} tensors from {csv}")
        if count == max_to_load:
            logger.info(f"Stopping tensor load after {max_to_load}")
            yield torch.stack(allsrc).char(), torch.stack(alltgt).long(), torch.tensor(alltgtvaf), varsinfo, torch.stack(allposflag).char()
            break

        if len(allsrc) >= chunk_size:
            yield torch.stack(allsrc).char(), torch.stack(alltgt).short(), torch.tensor(alltgtvaf), varsinfo, torch.stack(allposflag).char()
            allsrc = []
            alltgt = []
            alltgtvaf = []
            varsinfo = []
            allposflag = []

    if len(allsrc):
        yield torch.stack(allsrc).char(), torch.stack(alltgt).long(), torch.tensor(alltgtvaf), varsinfo, torch.stack(allposflag).char()

    logger.info(f"Done loading {count} tensors from {csv}")



def make_loader(bampath, refpath, csv, max_reads_per_aln, samples_per_pos, max_to_load=1e9):
    allsrc = []
    alltgt = []
    count = 0
    seq_len = 150
    logger.info(f"Creating new data loader from {bampath}")
    counter = defaultdict(int)
    classes = []
    for enc, tgt, row in load_from_csv(bampath, refpath, csv, max_reads_per_aln=max_reads_per_aln, samples_per_pos=samples_per_pos):
        status, vtype = row.status, row.vtype
        label_class = "-".join((status, vtype))
        classes.append(label_class)
        counter[label_class] += 1
        src, tgt, posflag = trim_pileuptensor(enc, tgt.unsqueeze(0), posflag, seq_len)
        assert src.shape[0] == seq_len, f"Src tensor #{count} had incorrect shape after trimming, found {src.shape[0]} but should be {seq_len}"
        assert tgt.shape[1] == seq_len, f"Tgt tensor #{count} had incorrect shape after trimming, found {tgt.shape[1]} but should be {seq_len}"
        assert posflag.shape
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
