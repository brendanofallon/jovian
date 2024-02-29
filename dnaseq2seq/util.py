import itertools
import datetime
import os
import time

import torch
import torch.nn as nn
import numpy as np
import gzip
import lz4.frame
import pysam
import io
from pathlib import Path
import logging
import shutil
from itertools import chain

from torch.nn.parallel import DistributedDataParallel

logger = logging.getLogger(__name__)

INDEX_TO_BASE = [
    'A', 'C', 'G', 'T'
]

def log_timer(func):
    """
    A decorator which when applied will make the total time taken to complete the function
    appear as a log message when the function completes
    :param func:
    :return: Decorator
    """

    def wrapper(*args, **kwargs):
        starttime = datetime.now()
        result = func(*args, **kwargs)
        elapsed = datetime.now() - starttime
        try:
            fname = func.__name__
        except:
            fname = "?"
        logger.debug(
            f"Function {fname} completed in {elapsed.total_seconds()} seconds"
        )
        return result

    return wrapper

def make_kmer_lookups(size):
    """
    Generate forward and reverse lookups for kmers
    str2index returns the index int for a given kmer,
    index2str returns the kmer string for a given index
    """
    bases = "ACGT"
    baselist = [bases] * size
    str2index = {}
    index2str = [None] * (len(bases) ** size)
    for i, combo in enumerate(itertools.product(*baselist)):
        s = ''.join(combo)
        str2index[s] = i
        index2str[i] = s
    return str2index, index2str

TGT_KMER_SIZE = 4
s2i, i2s = make_kmer_lookups(TGT_KMER_SIZE)
KMER_COUNT = 4 ** TGT_KMER_SIZE
FEATURE_DIM = KMER_COUNT + 4 # Add 4 to make it 260, which is evenly divisible by lots of numbers - needed for MHA
START_TOKEN = torch.zeros((1, FEATURE_DIM),  dtype=float)
START_TOKEN[:, FEATURE_DIM-1] = 1


def var_type(variant):
    if len(variant.ref) == 1 and len(variant.alt) == 1:
        return 'snv'
    elif len(variant.ref) == 0 and len(variant.alt) > 0:
        return 'ins'
    elif len(variant.ref) > 0 and len(variant.alt) == 0:
        return 'del'
    elif len(variant.ref) > 0 and len(variant.alt) > 0:
        return 'mnv'
    print(f"Whoa, unknown variant type: {variant}")
    return 'unknown'


def concat_metafile(sample_metafile, dest_metafh):
    """
    Concate the given sample metadata file to destination metadata file. 
    The sample metadata file is also removed as a side effect!

    :param smaple_metafile: the path of sample metadata
    :param dest_metafh: the file handle of destination metadata file.
    """
    with open(sample_metafile, 'rb') as sample_fh:
        shutil.copyfileobj(sample_fh, dest_metafh)
    os.unlink(sample_metafile)


def find_files(datadir, src_prefix='src', tgt_prefix='tgt'):
    """
    Examine files in datadir and match up all src / tgt / vaftgt files and store them as tuples in a list
    :returns : List of (src, tgt, vaftgt) tuples of matched files
    """
    datadir = Path(datadir)
    allsrc = list(datadir.glob(src_prefix + "*"))
    pairs = []
    for src in allsrc:
        suffix = src.name.split("_")[-1]
        pairs.append((src,
                     f"{datadir}/{tgt_prefix}_{suffix}",
                      ))
    return pairs


def tensor_from_lz4(path, device):
    with io.BytesIO(lz4.frame.decompress(path)) as bfh:
        return torch.load(bfh, map_location=device)


def tensor_from_gzip(path, device):
    return torch.load(io.BytesIO(gzip.decompress(path)), map_location=device)


def tensor_from_file(path, device):
    with open(path, "rb") as fh:
        data = fh.read()
    if str(path).endswith('.lz4'):
        return tensor_from_lz4(data, device)
    else:
        return torch.load(path, map_location=device)


def sortreads(reads):
    return sorted(reads, key=lambda r: r.reference_start)


def unzip_load(path, device='cpu'):
    """
    Read the given file and load then tensor it contains, 
    If path has a .gz suffix, ungzip it first then load
    :returns: torch.Tensor read from file
    """
    if str(path).endswith('.gz'):
        with gzip.open(path, 'rb') as fh:
            return torch.load(fh, map_location=device)
    else:
        return torch.load(path, map_location=device)


def readstr(t):
    t = t.detach().cpu().numpy()
    assert len(t.shape) == 2, "Need two dimensional input"
    output = []
    for pos in range(t.shape[0]):
        if t[pos, 0:4].sum() == 0:
            output.append(".")
        else:
            output.append(INDEX_TO_BASE[np.argmax(t[pos, 0:4])])
            if t.shape[-1] > 4 and t[pos, -1] == 1: # Clip flag is set, which means emit lowercase
                output[-1] = output[-1].lower()

    return "".join(output)


def to_pileup(data, altmask=None):
    pileup = []
    for i in range(data.shape[1]):
        alt = "" if altmask is None else altmask[i].item()
        pileup.append(readstr(data[:, i, :]) + f"\t{alt}")
    return "\n".join(pileup)


def predprobs(t):
    t = t.detach().cpu().numpy()
    output = []
    for pos in range(t.shape[0]):
        if t[pos, :].sum() == 0:
            output.append(".")
        else:
            output.append(f"{t[pos, np.argmax(t[pos, :])]:.1}f")
    return "".join(output)


def tgt_str(seq):
    return "".join(INDEX_TO_BASE[b] for b in seq)


def correctstr(seq, predseq):
    seq = "".join(INDEX_TO_BASE[b] for b in seq)
    output = "".join('*' if a==b else 'x' for a,b in zip(seq, predseq))
    return output


def writeseqtensor(t):
    assert len(t.shape) == 2, f"Expected 2-dimensional input"
    for pos in range(t.shape[0]):
        clip = 1 if t[pos, 7] else 0
        print(clip, end="")
    print()
    for pos in range(t.shape[0]):
        refmatch = 1 if t[pos, 6] else 0
        print(refmatch, end="")
    print()
    for pos in range(t.shape[0]):
        if torch.sum(t[pos, 0:4]) == 0:
            base = '.'
        elif torch.sum(t[pos, 0:4]) < 0:
            base = '-'
        else:
            base = INDEX_TO_BASE[torch.argmax(t[pos, 0:4])]
        print(f"{base}", end="")
    print()


def count_bases(bedpath):
    """
    Return total number of bases in a BED file
    """
    tot = 0
    with open(bedpath) as fh:
        for line in fh:
            if len(line.strip())==0 or line.startswith("#"):
                continue
            toks = line.split("\t")
            tot += int(toks[2]) - int(toks[1])
    return tot


def sort_chrom_vcf(input_vcf, dest):
    """
    Sort variants found in the given vcf file and write them to the given destination file
    Raises an exception if not all variants are on the same chromosome
    """
    vcf = pysam.VariantFile(input_vcf)
    chrom = None
    with open(dest, "w") as ofh:
        ofh.write(str(vcf.header))
        for var in sorted(vcf.fetch(), key=lambda x: x.pos):
            if chrom is None:
                chrom = var.chrom
            else:
                assert var.chrom == chrom, f"Variants must be on same chromsome, but found {var} which is not on chrom {chrom}"
            ofh.write(str(var))

        
def _varkey(variant):
    return variant.chrom, variant.pos, variant.ref, ",".join(str(s) for s in variant.alts)


def dedup_vcf(input_vcf, dest):
    """
    Iterate a VCF file and remove any variants that duplicates in terms of chrom/pos/ref/alt
    Requires input VCF to be sorted
    """
    vcf = pysam.VariantFile(input_vcf)
    clump = []
    with open(dest, "w") as ofh:
        ofh.write(str(vcf.header))
        for var in vcf.fetch():
            if len(clump) == 0:
                clump.append(var)
            else:
                if _varkey(var) == _varkey(clump[0]):
                    clump.append(var)
                else:
                    # Big question here: Which variant to write if there are duplicates found?
                    # Currently we just write the first one, but maybe we could be smarter
                    ofh.write(str(clump[0]))
                    clump = [var]
        if clump:
            ofh.write(str(clump[0]))

def check_overlap(interval1, interval2):
    start1, end1 = interval1
    start2, end2 = interval2
    return not (end1 < start2 or end2 < start1)

def records_overlap(rec1, rec2):
    """ True if the two records share any reference bases """
    return check_overlap(
        (rec1.pos, rec1.pos + (len(rec1.ref) - len(rec1.alts[0]))),
        (rec2.pos, rec2.pos + (len(rec2.ref) - len(rec2.alts[0]))),
    )

def merge_overlapping_regions(regions):
    """
    Merge any overlapping regions in the list into a single region
    Assumes regions is list / iterable of (chrom, chunk_index, start, end) tuples
    """
    if not len(list(regions)):
        return []
    regions = sorted(regions, key=lambda x: x[1])
    merged = [regions[0]]
    for region in regions[1:]:
        if region[0] == merged[-1][0] and region[2] <= merged[-1][3]:
            merged[-1] = (merged[-1][0], merged[-1][1], merged[-1][2], max(region[3], merged[-1][3]))
        else:
            merged.append(region)
    return merged


def kmer_preds_to_seq(preds, i2s):
    """
    Return a sequence of bases from the given predictions
    """
    m = torch.argmax(preds, dim=-1).cpu().detach().numpy()
    return kmer_idx_to_str(m, i2s)


def kmer_idx_to_str(kmer_idx, i2s):
    return ''.join(i2s[i] for i in kmer_idx)


def bases_to_kvec(bases, s2i, kmersize=4):
    """ Return a list of indices for nonoverlapping kmers read from the base """
    indices = []
    for i in range(0, len(bases), kmersize):
        kmer = bases[i:i+kmersize]
        indices.append(s2i[kmer])
    return indices


def seq_to_onehot_kmers(seq):
    h = []
    for i in bases_to_kvec(seq, s2i, TGT_KMER_SIZE):
        t = torch.zeros(FEATURE_DIM, dtype=torch.float)
        t[i] = 1
        h.append(t)
    return torch.stack(h)


def tgt_to_kmers(tgt):
    result = []
    for i in range(tgt.shape[0]):
        tgtseq0 = tgt_str(tgt[i, 0, :])
        tgtseq1 = tgt_str(tgt[i, 1, :])
        h0 = torch.concat((START_TOKEN, seq_to_onehot_kmers(tgtseq0)))
        h1 = torch.concat((START_TOKEN, seq_to_onehot_kmers(tgtseq1)))
        t = torch.stack((h0, h1))
        result.append(t)
    return torch.stack(result, dim=0)


def expand_to_bases(probs, expansion_factor=TGT_KMER_SIZE):
    return list(chain(*([k]*expansion_factor for k in probs)))


def predict_sequence(src, model, n_output_toks, device):
    """
    Generate a predicted sequence with next-word prediction be repeatedly calling the model
    """
    if isinstance(model, nn.DataParallel) or isinstance(model, DistributedDataParallel):
        model = model.module
    start = time.perf_counter()
    predictions = torch.stack((START_TOKEN, START_TOKEN), dim=0).expand(src.shape[0], -1, -1, -1).float().to(device)
    probs = torch.zeros(src.shape[0], 2, 1).float().to(device)
    mem = model.encode(src)
    encode = time.perf_counter()
    for i in range(n_output_toks + 1):
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(predictions.shape[-2]).to(device)
        new_preds = model.decode(mem, predictions, tgt_mask=tgt_mask)[:, :, -1:, :]
        new_probs, tophit = torch.max(new_preds, dim=-1)
        p = torch.nn.functional.one_hot(tophit, num_classes=260)
        predictions = torch.concat((predictions, p), dim=2)
        probs = torch.concat((probs, new_probs), dim=-1)
    encode_elapsed = encode - start
    decode_elapsed = time.perf_counter() - encode
    logger.debug(f"Encoding time: {encode_elapsed :.3f} n_toks: {n_output_toks}, decoding time: {decode_elapsed :.3f}")
    return predictions[:, :, 1:, :], probs[:, :, 1:]


class WarmupCosineLRScheduler:

    def __init__(self, max_lr, min_lr, warmup_iters, lr_decay_iters):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_iters = warmup_iters
        self.lr_decay_iters = lr_decay_iters
        self.iters = 0
        self.last_lr = float("NaN")

    def add_iters(self, iters):
        self.iters += iters

    def set_iters(self, iters):
        self.iters = iters

    def get_last_lr(self):
        return self.last_lr

    def get_lr(self):
        # 1) linear warmup for warmup_iters steps
        if self.iters < self.warmup_iters:
            lr = self.max_lr * (self.iters+1) / (self.warmup_iters)
        # 2) if it > lr_decay_iters, return min learning rate
        elif self.iters > self.lr_decay_iters:
            lr = self.min_lr
        else:
           # 3) in between, use cosine decay down to min learning rate
            decay_ratio = (self.iters - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))  # coeff ranges 0..1
            lr = self.min_lr + coeff * (self.max_lr - self.min_lr)
        self.last_lr = lr
        return lr
