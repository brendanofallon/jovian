
import logging

import phaser

logger = logging.getLogger(__name__)

import random
import math
from pathlib import Path
from collections import defaultdict
from itertools import chain
import gzip
import lz4.frame
from datetime import datetime
import traceback as tb

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

    def __init__(self, bam, bed, vcf, reference, reads_per_pileup, samples_per_pos, vals_per_class):
        self.bam = bam
        self.bed = bed
        self.vcf = vcf
        self.reference = reference
        self.reads_per_pileup = reads_per_pileup
        self.samples_per_pos = samples_per_pos
        self.vals_per_class = vals_per_class

    def iter_once(self, batch_size):
        logger.info(f"Encoding tensors from {self.bam} and {self.vcf}")
        for src, tgt, vaftgt, varsinfo in encode_chunks(self.bam,
                                          self.reference,
                                          self.bed,
                                          self.vcf,
                                          batch_size,
                                          self.reads_per_pileup,
                                          self.samples_per_pos,
                                          self.vals_per_class,
                                          max_to_load=1e9):
                yield src, tgt, vaftgt, varsinfo


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

    def __init__(self, device, datadir, threads, max_decomped_batches=10, src_prefix="src", tgt_prefix="tgt", vaftgt_prefix="vaftgt", pathpairs=None):
        """
        Create a new loader that reads tensors from a 'pre-gen' directory
        :param device: torch.device
        :param datadir: Directory to read data from
        :param threads: Max threads to use for decompressing data
        :param max_decomped_batches: Maximum number of batches to decompress at once. Increase this on machines with tons of RAM
        :param pathpairs: List of (src path, tgt path, vaftgt path) tuples to use for data
        """
        self.device = device
        self.datadir = Path(datadir) if datadir else None
        self.src_prefix = src_prefix
        self.tgt_prefix = tgt_prefix
        self.vaftgt_prefix = vaftgt_prefix
        if pathpairs and datadir:
            raise ValueError(f"Both datadir and pathpairs specified for PregenLoader - please choose just one")
        if pathpairs:
            self.pathpairs = pathpairs
        else:
            self.pathpairs = util.find_files(self.datadir, self.src_prefix, self.tgt_prefix, self.vaftgt_prefix)
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
        src, tgt, vaftgt = [], [], []
        for i in range(0, len(self.pathpairs), self.max_decomped):
            decomp_start = datetime.now()
            paths = self.pathpairs[i:i+self.max_decomped]
            decomped = decompress_multi(chain.from_iterable(paths), self.threads)
            decomp_end = datetime.now()
            decomp_time = (decomp_end - decomp_start).total_seconds()

            for j in range(0, len(decomped), 3):
                src.append(decomped[j])
                tgt.append(decomped[j+1])
                vaftgt.append(decomped[j+2])

            total_size = sum([s.shape[0] for s in src])
            if total_size < batch_size:
                # We need to decompress more data to make a batch
                continue

            # Make a big tensor.
            src_t = torch.cat(src, dim=0)
            tgt_t = torch.cat(tgt, dim=0)
            vaftgt_t = torch.cat(vaftgt, dim=0)

            nbatch = total_size // batch_size
            remain = total_size % batch_size

            # Slice the big tensors for batches
            for n in range(0, nbatch):
                start = n * batch_size
                end = (n + 1) * batch_size
                yield (
                    src_t[start:end].to(self.device).float(),
                    tgt_t[start:end].to(self.device).long(),
                    None, #vaftgt_t[start:end].to(self.device), 
                    None,
                    {"decomp_time": decomp_time},
                )
                decomp_time = 0.0


            if remain:
                # The remaining data points will be in next batch. 
                src = [src_t[nbatch * batch_size:]]
                tgt = [tgt_t[nbatch * batch_size:]]
                vaftgt = [vaftgt_t[nbatch * batch_size:]]
            else:
                src, tgt, vaftgt = [], [], []


        if len(src) > 0:
            # We need to yield the last batch.
            yield (
                torch.cat(src, dim=0).to(self.device).float(),
                torch.cat(tgt, dim=0).to(self.device).long(),
                None, #atorch.cat(vaftgt, dim=0).to(self.device),
                None,
                {"decomp_time": 0.0},
            )


class ShorteningLoader:
    """
    This loader shortens the sequence dimension (dimension 1 of src, 2 of tgt) to seq_len
    It should be used to wrap another loader that actually does the loading
    For instance:

        loader = ShorteningLoader(PregenLoader(...), seq_len=100)

    """
    def __init__(self, wrapped_loader, seq_len, fraction_to_augment):
        self.wrapped_loader = wrapped_loader
        self.seq_len = seq_len
        self.fraction_to_augment = fraction_to_augment

    def iter_once(self, batch_size):
        iter = 0
        for src, tgt, vaftgt, _, log_info in self.wrapped_loader.iter_once(batch_size):
            if np.random.rand() < self.fraction_to_augment:
                start = src.shape[1] // 2 - self.seq_len // 2
                end = src.shape[1] // 2 + self.seq_len // 2
                src = src[:, start:end, :, :]
                tgt = tgt[:, :, start:end]
                iter += 1
            yield src, tgt, vaftgt, None, log_info
        logger.info(f"Total number of sample batches that had their sequence dimension shortened are: {iter}")


class ShufflingLoader:
    """
    This loader shuffles the reads (dimension 2 of src), excluding the 0th element i.e. ref read.
    It should be used to wrap another loader that actually does the loading
    For instance:

        loader = ShufflingLoader(PregenLoader(...))

    """
    def __init__(self, wrapped_loader, fraction_to_augment):
        self.wrapped_loader = wrapped_loader
        self.fraction_to_augment = fraction_to_augment

    def iter_once(self, batch_size):
        iter = 0
        for src, tgt, vaftgt, _, log_info in self.wrapped_loader.iter_once(batch_size):
            if np.random.rand() < self.fraction_to_augment:
                src_non_ref_reads = src[:, :, 1:, :]
                src_non_ref_reads_shuf = src_non_ref_reads[:, :, torch.randperm(src_non_ref_reads.shape[2]), :]
                src = torch.cat((src[:, :, :1, :], src_non_ref_reads_shuf), dim=2)
                iter += 1
            yield src, tgt, vaftgt, None, log_info
        logger.info(f"Total number of sample batches that had their reads shuffled are: {iter}")


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
    def __init__(self, wrapped_loader, prob_of_read_being_dropped, fraction_to_augment):
        self.wrapped_loader = wrapped_loader
        self.prob_of_read_being_dropped = prob_of_read_being_dropped
        self.fraction_to_augment = fraction_to_augment

    def iter_once(self, batch_size):
        for src, tgt, vaftgt, _, log_info in self.wrapped_loader.iter_once(batch_size):
            iter = 0
            if np.random.rand() < self.fraction_to_augment:
                num_reads_to_drop = stats.binom(n=src.shape[2], p=self.prob_of_read_being_dropped).rvs(src.shape[0])
                for idx in range(src.shape[0]):
                    logger.debug(f"{num_reads_to_drop[idx]} reads to be dropped out of total {src.shape[2]} reads in batch: {idx}")
                    read_index_to_drop = random.sample(list(range(src.shape[2])[1:]), num_reads_to_drop[idx])
                    logger.debug(f"Reads at batch id {idx} and read index {read_index_to_drop} are being dropped, excluding ref read at 0th position.")
                    src[idx, :, read_index_to_drop, :] = 0
                iter += 1
            yield src, tgt, vaftgt, None, log_info
        logger.info(f"Total number of sample batches that had their reads downsampled are: {iter}")


class MultiLoader:
    """
    Combines multiple loaders into a single loader
    """

    def __init__(self, loaders):
        self.loaders = loaders

    def iter_once(self, batch_size):
        for loader in self.loaders:
            for src, tgt in loader.iter_once(batch_size):
                yield src, tgt, None, None, None


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
            yield src.to(self.device), tgt.to(self.device), vaftgt.to(self.device), altmask.to(self.device), None


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
            yield src.to(self.device), tgt.to(self.device), vaftgt.to(self.device), altmask.to(self.device), None


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
        z = torch.zeros(width - src.shape[0], src.shape[1], src.shape[2], dtype=src.dtype)
        src = torch.cat((src, z))
        t = torch.zeros(tgt.shape[0], width - tgt.shape[1]).long()
        tgt = torch.cat((tgt, t), dim=1)
    else:
        start = src.shape[0] // 2 - width // 2
        src = src[start:start+width, :, :]
        tgt = tgt[:, start:start+width]

    return src, tgt


# def assign_class_indexes(rows):
#     """
#     Iterate over every row in rows and create a list of class indexes for each element
#     , where class index is currently row.vtype-row.status. So a class will be something like
#     snv-TP or small_del-FP, and each class gets a unique number
#     :returns : 1. List of class indexes across rows.
#                2. A dictionary keyed by class index and valued by class names.
#     """
#     classcount = 0
#     classes = defaultdict(int)
#     idxs = []
#     for row in rows:
#         clz = f"{row.vtype}-{row.status}" # Could imagine putting VAF here too - maybe low / medium / high vaf?
#         if clz in classes:
#             idx = classes[clz]
#         else:
#             idx = classcount
#             classcount += 1
#             classes[clz] = idx
#         idxs.append(idx)
#     class_names = {v: k for k, v in classes.items()}
#     return idxs, class_names


def parse_rows_classes(bed):
    """
    Iterate over every row in rows and create a list of class indexes for each element
    , where class index is currently row.vtype-row.status. So a class will be something like
    snv-TP or small_del-FP, and each class gets a unique number
    :returns : 0. List of chrom / start / end tuples from the input BED
               1. List of class indexes across rows.
               2. A dictionary keyed by class index and valued by class names.
    """
    classcount = 0
    classes = defaultdict(int)
    idxs = []
    rows = []
    with open(bed) as fh:
        for line in fh:
            row = line.strip().split("\t")
            chrom = row[0]
            start = int(row[1])
            end = int(row[2])
            clz = row[3]
            rows.append((chrom, start, end))
            if clz in classes:
                idx = classes[clz]
            else:
                idx = classcount
                classcount += 1
                classes[clz] = idx
            idxs.append(idx)
    class_names = {v: k for k, v in classes.items()}
    return rows, idxs, class_names

def resample_classes(classes, class_names, rows, vals_per_class):
    """
    Randomly sample each class so that there are 'vals_per_class' members of each class
    then return a list of class assignments and the selected rows
    """
    byclass = defaultdict(list)
    for clz, row in zip(classes, rows):
        byclass[clz].append(row)

    for name in class_names.values():
        if name not in vals_per_class:
            logger.warning(f"Class name '{name}' not found in vals per class, will select default of {vals_per_class['asldkjas']} items from class {name}")

    result_rows = []
    result_classes = []
    for clz, classrows in byclass.items():

        # If there are MORE instances on this class than we want, grab a random sample of them
        logger.info(f"Variant class {class_names[clz]} contains {len(classrows)} variants")
        if len(classrows) > vals_per_class[class_names[clz]]:
            logger.info(f"Randomly sampling {vals_per_class[class_names[clz]]} variants for variant class {class_names[clz]}")
            rows = random.sample(classrows, vals_per_class[class_names[clz]])
            result_rows.extend(rows)
            result_classes.extend([clz] * vals_per_class[class_names[clz]])
        else:
            # If there are FEWER instances of the class than we want, then loop over them
            # repeatedly - no random sampling since that might ignore a few of the precious few samples we do have
            logger.info(f"Loop over {len(classrows)} variants repeatedly to generate {vals_per_class[class_names[clz]]} {class_names[clz]} variants")
            for i in range(vals_per_class[class_names[clz]]):
                result_classes.append(clz)
                idx = i % len(classrows)
                result_rows.append(classrows[idx])

    return result_classes, result_rows


def upsample_labels(bed, vals_per_class):
    """
    Generate class assignments for each element in rows (see assign_class_indexes),
     then create a new list of rows that is "upsampled" to normalize class frequencies
     The new list will be multiple times larger then the old list and will contain repeated
     elements of the low frequency classes
    :param rows: List of elements with vtype and status attributes (probably read in from a labels CSV file)
    :param vals_per_class: Number of instances to retain for each class, OR a Mapping with class names and values
    :returns: List of rows, with less frequent rows included multiple times to help normalize frequencies
    """
    rows, label_idxs, label_names = parse_rows_classes(bed)
    label_idxs, rows = resample_classes(
        np.array(label_idxs), label_names, rows, vals_per_class=vals_per_class
    )
    # classes, counts = np.unique(label_idxs, return_counts=True

    random.shuffle(rows)
    return rows


def load_from_csv(bampath, refpath, bed, vcfpath, max_reads_per_aln, samples_per_pos, vals_per_class):
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
    vcf = pysam.VariantFile(vcfpath)

    upsampled_labels = upsample_labels(bed, vals_per_class=vals_per_class)
    logger.info(f"Will save {len(upsampled_labels)} with up to {samples_per_pos} samples per site from {bed}")
    for region in upsampled_labels:
        chrom, start, end = region[0:3]
        variants = list(vcf.fetch(chrom, start, end))
        logger.info(f"Number of variants in {chrom}:{start}-{end} : {len(variants)}")
        try:
            for encoded, (minref, maxref) in encode_and_downsample(chrom,
                                                 start,
                                                 end,
                                                 bam,
                                                 refgenome,
                                                 max_reads_per_aln,
                                                 samples_per_pos):

                hap0, hap1 = phaser.gen_haplotypes(bam, refgenome, chrom, minref, maxref, variants)
                regionsize = end - start
                midstart = max(0, start - minref)
                midend = midstart + regionsize

                tgt0 = target_string_to_tensor(hap0[midstart:midend])
                tgt1 = target_string_to_tensor(hap1[midstart:midend])
                tgt_haps = torch.stack((tgt0, tgt1))

                if encoded.shape[0] > regionsize:
                    encoded = encoded[midstart:midend, :, :]
                if encoded.shape[0] != tgt_haps.shape[-1]:
                    # This happens sometimes at the edges of regions where the reads only overlap a few bases at
                    # the start of the region.. raising an exception just causes this region to be skipped
                    raise ValueError(f"src shape {encoded.shape} doesn't match haplotype shape {tgt_haps.shape}, skipping")

                yield encoded, tgt_haps, region

        except Exception as ex:
            logger.warning(f"Error encoding position {chrom}:{start}-{end} for bam: {bampath}, skipping it: {ex}")
            tb.print_exc()



def encode_chunks(bampath, refpath, bed, vcf, chunk_size, max_reads_per_aln, samples_per_pos, vals_per_class, max_to_load=1e9):
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
    varsinfo = []
    count = 0
    seq_len = 300
    logger.info(f"Creating new data loader from {bampath}, vals_per_class: {vals_per_class}")
    for enc, tgt, region in load_from_csv(bampath, refpath, bed, vcf, max_reads_per_aln=max_reads_per_aln, samples_per_pos=samples_per_pos, vals_per_class=vals_per_class):
        src, tgt = trim_pileuptensor(enc, tgt, seq_len)
        src = ensure_dim(src, seq_len, max_reads_per_aln)
        assert src.shape[0] == seq_len, f"Src tensor #{count} had incorrect shape after trimming, found {src.shape[0]} but should be {seq_len}"
        assert tgt.shape[-1] == seq_len, f"Tgt tensor #{count} had incorrect shape after trimming, found {tgt.shape[-1]} but should be {seq_len}"
        allsrc.append(src)
        alltgt.append(tgt)

        count += 1
        if count % 1 == 0:
            logger.info(f"Loaded {count} tensors from {bampath}")
        if count == max_to_load:
            logger.info(f"Stopping tensor load after {max_to_load}")
            yield torch.stack(allsrc).char(), torch.stack(alltgt).long(), torch.tensor(alltgtvaf), varsinfo
            break

        if len(allsrc) >= chunk_size:
            yield torch.stack(allsrc).char(), torch.stack(alltgt).long(), torch.tensor(alltgtvaf), varsinfo
            allsrc = []
            alltgt = []
            alltgtvaf = []
            varsinfo = []

    if len(allsrc):
        yield torch.stack(allsrc).char(), torch.stack(alltgt).long(), torch.tensor(alltgtvaf), varsinfo

    logger.info(f"Done loading {count} tensors from {bampath}")



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


