import itertools
import os
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

logger = logging.getLogger(__name__)

import logging
logger = logging.getLogger(__name__)


INDEX_TO_BASE = [
    'A', 'C', 'G', 'T'
]


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
START_TOKEN_INDEX = FEATURE_DIM - 1
START_TOKEN[:, START_TOKEN_INDEX] = 1


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


def find_files(datadir, src_prefix='src', tgt_prefix='tgt', vaftgt_prefix='vaftgt'):
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
                     f"{datadir}/{vaftgt_prefix}_{suffix}"
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



def predict_sequence_greedy(src, model, n_output_toks, device):
    """
    Generate a predicted sequence with next-word prediction be repeatedly calling the model
    We just grab the kmer with the highest predicted probability and use it no matter what
    """
    predictions = torch.stack((START_TOKEN, START_TOKEN), dim=0).expand(src.shape[0], -1, -1, -1).float().to(device)
    mem = model.encode(src)
    for i in range(n_output_toks + 1):
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(predictions.shape[-2]).to(device)
        new_preds = model.decode(mem, predictions, tgt_mask=tgt_mask)[:, :, -1:, :]
        tophit = torch.argmax(new_preds, dim=-1)
        p = torch.nn.functional.one_hot(tophit, num_classes=260)
        predictions = torch.concat((predictions, p), dim=2)
    return predictions[:, :, 1:, :]


def search_one_step(paths, probs, model, mem, device, split_threshold):
    path0, path1 = paths
    prob0, prob1 = probs
    pred0_input = nn.functional.one_hot(path0, num_classes=START_TOKEN.shape[1])
    pred1_input = nn.functional.one_hot(path1, num_classes=START_TOKEN.shape[1])
    input = torch.stack((pred0_input, pred1_input), dim=0).unsqueeze(0)
    tgt_mask = nn.Transformer.generate_square_subsequent_mask(input.shape[-2]).to(device)
    new_preds = model.decode(mem, input, tgt_mask=tgt_mask)[:, :, -1:, :]
    tophits = new_preds.topk(k=2)
    diffs = tophits.values[0, :, :, 0] - tophits.values[0, :, :, 1]

    path_result = [
        (torch.cat((path0, tophits.indices[0, 0:1, -1, 0]), dim=0),
         torch.cat((path1, tophits.indices[0, 1:2, -1, 0]), dim=0))
    ]
    prob_result = [
        (prob0 + [tophits.values[0, 0, -1, 0].item()], prob1 + [tophits.values[0, 1, -1, 0].item()])
    ]

    if diffs[0] >= split_threshold and diffs[1] < split_threshold:
        path_result.append(
            (torch.cat((path0, tophits.indices[0, 0:1, -1, 0]), dim=0),
             torch.cat((path1, tophits.indices[0, 1:2, -1, 1]), dim=0))
        )
         # Use highest val from path 0 and second highest from path 1
        prob_result.append(
            (prob0 + [tophits.values[0, 0, -1, 0].item()], prob1 + [tophits.values[0, 1, -1, 1].item()])
        )

    elif diffs[0] < split_threshold and diffs[1] >= split_threshold:
        path_result.append(
            (torch.cat((path0, tophits.indices[0, 0:1, -1, 1]), dim=0),  # Second highest val from path 0 and highest from path 1
             torch.cat((path1, tophits.indices[0, 1:2, -1, 0]), dim=0))
        )
        prob_result.append(
            (prob0 + [tophits.values[0, 0, -1, 1].item()], prob1 + [tophits.values[0, 1, -1, 0].item()])
        )

    elif diffs[0] < split_threshold and diffs[1] < split_threshold:
        # This should be rare and I'm not sure how to handle it, for now we add *one* new path
        # with second-best values from both haplotypes, instead of two new paths (one with each second-best values)
        # or both each one individually and both together
        path_result.append(
            (torch.cat((path0, tophits.indices[0, 0:1, -1, 1]), dim=0),  # Second highest val from path 0 and second highest from path 1
             torch.cat((path1, tophits.indices[0, 1:2, -1, 1]), dim=0))
        )
        prob_result.append(
            (prob0 + [tophits.values[0, 0, -1, 1].item()], prob1 + [tophits.values[0, 1, -1, 1].item()])
        )

    return path_result, prob_result

def search_sequence_paths(src, model, n_output_toks, device, split_thresh):
    """
    Generate a predicted sequence with next-word prediction be repeatedly calling the model
    Modified beam search, only exploring other paths if they have a probability sufficiently close
    to the highest probability
    """
    assert src.shape[0] == 1, f"Can only handle batch size of 1 for now"
    mem = model.encode(src)
    path0 = torch.tensor(START_TOKEN_INDEX).unsqueeze(0)
    path1 = torch.tensor(START_TOKEN_INDEX).unsqueeze(0)
    paths = [(path0, path1)]
    probs = [([0.0], [0.0])]
    for i in range(n_output_toks):
        newpaths = []
        newprobs = []
        for path, prob in zip(paths, probs):
            pth, prb = search_one_step(path, prob, model, mem, device, split_thresh)
            newpaths.extend(pth)
            newprobs.extend(prb)
        paths = newpaths
        probs = newprobs

    trimmed_paths = []
    trimmed_probs = []
    for (p0, p1), (b0, b1) in zip(paths, probs):
        trimmed_paths.append((p0[1:], p1[1:]))
        trimmed_probs.append((b0[1:], b1[1:]))

    return trimmed_paths, trimmed_probs


def expand_to_bases(probs, expansion_factor=TGT_KMER_SIZE):
    result = []
    for a,b in probs:
        newa = list(chain(*([k]*expansion_factor for k in a)))
        newb = list(chain(*([k]*expansion_factor for k in b)))
        result.append((newa, newb))
    return result


def predict_sequence_search(src, model, n_output_toks, device, split_thresh=1.0):
    allpaths = []
    allprobs = []
    for i in range(src.shape[0]):
        paths, probs = search_sequence_paths(src[i:i+1, :, :, :], model, n_output_toks, device, split_thresh=split_thresh)
        allpaths.append(paths)
        baseprobs = expand_to_bases(probs)
        allprobs.append(baseprobs)

        # if len(paths) > 1:
        #     x0 = kmer_idx_to_str(paths[0][0].cpu().detach().numpy()[1:], i2s)
        #     y0 = kmer_idx_to_str(paths[0][1].cpu().detach().numpy()[1:], i2s)
        #
        #     x1 = kmer_idx_to_str(paths[1][0].cpu().detach().numpy()[1:], i2s)
        #     y1 = kmer_idx_to_str(paths[1][1].cpu().detach().numpy()[1:], i2s)
        #     z0 = x0 == x1
        #     z1 = y0 == y1
        # for j, (p0, p1) in enumerate(baseprobs):
        #     psums.append(sum(p0) + sum(p1))
        #     print(f"P0 sum: {sum(p0) :.5f}: P1 sum: {sum(p1) :.5f}")
        #     p0str = ' '.join(f"{np.exp(x):.3f}" for x in p0[4:])
        #     p1str = ' '.join(f"{np.exp(x):.3f}" for x in p1[4:])
        #     b0 = '     '.join(b for b in kmer_idx_to_str(paths[j][0].cpu().detach().numpy()[1:], i2s))
        #     b1 = '     '.join(b for b in kmer_idx_to_str(paths[j][1].cpu().detach().numpy()[1:], i2s))
        #     print(p0str)
        #     print(b0)
        #     print(p1str)
        #     print(b1)
            # print(kmer_idx_to_str(paths[j][1].cpu().detach().numpy()[1:-1], i2s))

        # best_path = np.argmax(psums)
        # if best_path != 0:
        #     logger.info("Its not 0!!")
        # winners.append(torch.stack(paths[best_path], dim=0))

    return allpaths, allprobs


