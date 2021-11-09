import os
import torch
import numpy as np
import gzip
import lz4.frame
import io
from pathlib import Path
import logging
import shutil

logger = logging.getLogger(__name__)

import logging
logger = logging.getLogger(__name__)


INDEX_TO_BASE = [
    'A', 'C', 'G', 'T'
]


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


def find_tgt(suffix, files):
    found = None
    for tgt in files:
        tsuf = tgt.name.split("_")[-1].split(".")[0]
        if tsuf == suffix:
            if found:
                raise ValueError(f"Uh oh, found multiple matches for suffix {suffix}!")
            found = tgt
            break
    if found is None:
        raise ValueError(f"Could not find matching tgt file for {suffix}")
    return found


def find_files(datadir, src_prefix='src', tgt_prefix='tgt', vaftgt_prefix='vaftgt', posflag_prefix=None):
    """
    Examine files in datadir and match up all src / tgt / vaftgt files and store them as tuples in a list
    :returns : List of (src, tgt, vaftgt) tuples of matched files
    """
    datadir = Path(datadir)
    allsrc = list(datadir.glob(src_prefix + "*"))
    alltgt = list(datadir.glob(tgt_prefix + "*"))
    allvaftgt = list(datadir.glob(vaftgt_prefix + "*"))
    if posflag_prefix:
        allposflag = list(datadir.glob(posflag_prefix + "*"))
    else:
        allposflag = []
    pairs = []
    for src in allsrc:
        suffix = src.name.split("_")[-1].split(".")[0]
        tgt = find_tgt(suffix, alltgt)
        alltgt.remove(tgt)
        vaftgt = find_tgt(suffix, allvaftgt)
        allvaftgt.remove(vaftgt)
        if allposflag:
            posflags = find_tgt(suffix, allposflag)
        else:
            posflags = []
        pairs.append((src, tgt, vaftgt, posflags))
    return pairs

def tensor_from_lz4(path, device):
    return torch.load(io.BytesIO(lz4.frame.decompress(path)), map_location=device)


def tensor_from_gzip(path, device):
    return torch.load(io.BytesIO(gzip.decompress(path)), map_location=device)


def tensor_from_file(path, device):
    with open(path, "rb") as fh:
        data = fh.read()
    if str(path).endswith('.gz'):
        return tensor_from_gz(data, device)
    elif str(path).endswith('.lz4'):
        return tensor_from_lz4(data, device)
    else:
        return torch.load(path, map_location=device)


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
