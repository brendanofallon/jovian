
import random
import traceback

import numpy as np
import torch
import logging
import pysam

import util

logger = logging.getLogger(__name__)



EMPTY_TENSOR = torch.zeros(9)

class MockRead:

    def __init__(self, bases, quals, start, end, cigartups):
        assert len(bases) == len(quals)
        self.query_sequence = bases
        self.query_qualities = quals
        self.reference_start = start
        self.query_alignment_start = self.reference_start
        self.reference_end = end
        self.query_length = len(self.query_sequence)
        self.cigartuples = cigartups
        n_cig_bases = sum(a[1] for a in self.cigartuples)
        assert n_cig_bases == len(bases), "Cigar bases count doesn't match actual number of bases"

    def get_aligned_pairs(self):
        refpos = self.reference_start
        readpos = 0
        result = []
        for cigop, nbases in self.cigartuples:
            ref_base_consumed = cigop in {0, 2, 4, 5, 7}
            seq_base_consumed = cigop in {0, 1, 3, 4,5,7}
            for i in range(nbases):
                read_emit = readpos if seq_base_consumed else None
                ref_emit = refpos if ref_base_consumed else None
                result.append((read_emit, ref_emit))
                if ref_base_consumed:
                    refpos += 1
                if seq_base_consumed:
                    readpos += 1
        return result



def base_index(base):
    base = base.upper()
    if base == 'A':
        return 0
    elif base == 'C':
        return 1
    elif base == 'G':
        return 2
    elif base == 'T':
        return 3
    raise ValueError(f"Expected [ACTG], got {base}")


def update_from_base(base, tensor):
    if base == 'A':
        tensor[0] = 1
    elif base == 'C':
        tensor[1] = 1
    elif base == 'G':
        tensor[2] = 1
    elif base == 'T':
        tensor[3] = 1
    elif base == 'N':
        tensor[0:4] = 1
    elif base == '-':
        tensor[0:4] = 0
    return tensor


def encode_basecall(base, qual, consumes_ref_base, consumes_read_base, strand, clipped):
    ebc = torch.zeros(9).char() # Char is a signed 8-bit integer, so ints from -128 - 127 only
    ebc = update_from_base(base, ebc)
    ebc[4] = int(round(qual / 10))
    ebc[5] = consumes_ref_base # Consumes a base on reference seq - which means not insertion
    ebc[6] = consumes_read_base # Consumes a base on read - so not a deletion
    ebc[7] = 1 if strand else 0
    ebc[8] = 1 if clipped else 0
    return ebc


def decode(t):
    t = t.squeeze()
    if torch.sum(t[0:4]) == 0.0:
        return '-'
    else:
        return util.INDEX_TO_BASE[t[0:4].argmax()]


# def encode_cigop(readpos, refpos):
#     if readpos == refpos:
#         return 0
#     elif readpos is None:
#         return -1
#     elif refpos is None:
#         return 1
#     return 0



def string_to_tensor(bases):
    return torch.vstack([encode_basecall(b, 50, 0, 0, 0, 0) for b in bases])


def target_string_to_tensor(bases):
    """
    Encode the string into a tensor with base index values, like class labels, for each position
     The tensor looks like [0,1,2,1,0,2,3,0...]
    """
    result = torch.tensor([base_index(b) for b in bases]).long()
    return result


def pad_zeros(pre, data, post):
    if pre:
        prepad = torch.zeros(pre, data.shape[-1], dtype=data.dtype)
        data = torch.cat((prepad, data))
    if post:
        postpad = torch.zeros(post, data.shape[-1], dtype=data.dtype)
        data = torch.cat((data, postpad))
    return data


def iterate_cigar(rec):
    cigtups = rec.cigartuples
    bases = rec.query_sequence
    quals = rec.query_qualities
    cig_index = 0
    n_bases_cigop = cigtups[cig_index][1]
    cigop = cigtups[cig_index][0]
    is_ref_consumed = cigop in {0, 2, 4, 5, 7}  # 2 is deletion
    is_seq_consumed = cigop in {0, 1, 3, 4, 7}  # 1 is insertion, 3 is 'ref skip'
    is_clipped = cigop in {4, 5}
    base_index = 0
    refstart = alnstart(rec)
    refpos = refstart
    while True:
        if is_seq_consumed:
            base = bases[base_index]
            qual = quals[base_index]
            base_index += 1
        else:
            base = "-"
            qual = 0
        if is_ref_consumed:
            refpos += 1

        yield encode_basecall(base, qual, is_ref_consumed, is_seq_consumed, rec.is_reverse, is_clipped), is_ref_consumed
        n_bases_cigop -= 1
        if n_bases_cigop <= 0:
            cig_index += 1
            if cig_index >= len(cigtups):
                break
            n_bases_cigop = cigtups[cig_index][1]
            cigop = cigtups[cig_index][0]
            is_ref_consumed = cigop in {0, 2, 4, 5, 7}
            is_seq_consumed = cigop in {0, 1, 3, 4, 7}
            is_clipped = cigop in {4, 5}


def iterate_bases(rec):
    """
    Generate encoded base calls for the given variant record, this version does NOT
    insert gaps into the bases if there's a deletion in the cigar - it just reads right on thru
    :param rec: pysam VariantRecord
    :return: Generator for encoded base calls
    """
    cigtups = rec.cigartuples
    bases = rec.query_sequence
    quals = rec.query_qualities
    cig_index = 0
    n_bases_cigop = cigtups[cig_index][1]
    cigop = cigtups[cig_index][0]
    is_ref_consumed = cigop in {0, 2, 4, 5, 7}  # 2 is deletion
    is_seq_consumed = cigop in {0, 1, 3, 4, 7}  # 1 is insertion, 3 is 'ref skip'
    is_clipped = cigop in {4, 5}
    for i, (base, qual) in enumerate(zip(bases, quals)):
        readpos = i/150 if not rec.is_reverse else 1.0 - i/150
        yield encode_basecall(base, qual, is_ref_consumed, is_seq_consumed, rec.is_reverse, is_clipped), is_ref_consumed
        n_bases_cigop -= 1
        if n_bases_cigop <= 0:
            cig_index += 1
            if cig_index >= len(cigtups):
                break
            n_bases_cigop = cigtups[cig_index][1]
            cigop = cigtups[cig_index][0]
            is_ref_consumed = cigop in {0, 2, 4, 5, 7}
            is_seq_consumed = cigop in {0, 1, 3, 4, 7}
            is_clipped = cigop in {4, 5}


def rec_tensor_it(read, minref):
    for i in range(alnstart(read) - minref):
        yield EMPTY_TENSOR, True

    try:
        for t in iterate_bases(read):
            yield t
    except StopIteration:
        pass

    while True:
        yield EMPTY_TENSOR, True


def emit_tensor_aln(t):
    """
    Expecting t [read, position, bases]
    """
    for read_idx in range(t.shape[1]):
        for pos_idx in range(t.shape[0]):
            b = decode(t[pos_idx, read_idx, :])
            print(b, end='')
        print()


def alnstart(read):
    """
    If the first cigar element is hard or soft clip, return read.reference_start - size of first cigar element,
    otherwise return read.reference_start
    """
    if read.cigartuples[0][0] in {4, 5}:
        return read.reference_start - read.cigartuples[0][1]
    else:
        return read.reference_start


def _consume_n(it, n):
    """ Yield the first n elements of the given iterator """
    for i in range(n):
        yield next(it)


def encode_pileup3(reads, start, end):
    """
    Convert a list of reads (pysam VariantRecords) into a single tensor

    :param reads: List of pysam reads
    :param start: Genomic start coordinate
    :param end: Genomic end coordinate
    :return: Tensor with shape [position, read, features]
    """
    # minref = min(alnstart(r) for r in reads)
    # maxref = max(alnstart(r) + r.query_length for r in reads)
    isalt = ["alt" in r.query_name for r in reads]
    everything = []
    for readnum, read in enumerate(reads):
        try:
            readencoded = [enc.char() for enc, refconsumed in _consume_n(rec_tensor_it(read, start), end-start)]
            everything.append(torch.stack(readencoded))
        except Exception as ex:
            logger.warn(f"Error processing read {read.query_name}: {ex}, skipping it")
            traceback.print_exception(type(ex), ex, ex.__traceback__)
            raise ex

    return torch.stack(everything).transpose(0,1), torch.tensor(isalt)


def ensure_dim(readtensor, seqdim, readdim):
    """
    Trim or zero-pad the readtensor to make sure it has exactly 'seqdim' size for the sequence length
    and 'readdim' size for the read dimension
    Assumes readtensor has dimension [seq, read, features]
    :return:
    """
    if readtensor.shape[0] >= seqdim:
        readtensor = readtensor[0:seqdim, :, :]
    else:
        pad = torch.zeros(seqdim - readtensor.shape[0], readtensor.shape[1], readtensor.shape[2], dtype=readtensor.dtype)
        readtensor = torch.cat((readtensor, pad), dim=0)

    if readtensor.shape[1] >= readdim:
        readtensor = readtensor[:, 0:readdim, :]
    else:
        pad = torch.zeros(readtensor.shape[0], readdim - readtensor.shape[1], readtensor.shape[2], dtype=readtensor.dtype)
        readtensor = torch.cat((readtensor, pad), dim=1)
    return readtensor


def format_cigar(cig):
    return cig.replace("M", "M ").replace("S", "S ").replace("I", "I ").replace("D", "D ")


def reads_spanning(bam, chrom, pos, max_reads):
    """
    Return a list of reads spanning the given position, generally attempting to take
    reads in which 'pos' is approximately in the middle of the read
    :return : list of reads spanning the given position
    """
    start = pos - 10
    bamit = bam.fetch(chrom, start)
    reads = []
    try:
        read = next(bamit)
        while read.reference_start < pos:
            if read.reference_end is not None and read.reference_start < pos < read.reference_end:
                reads.append(read)
            read = next(bamit)
    except StopIteration:
        pass
    mid = len(reads) // 2
    return reads[max(0, mid-max_reads//2):min(len(reads), mid+max_reads//2)]

def reads_spanning_range(bam, chrom, start, end):
    """
    Return a list of reads spanning the given position, generally attempting to take
    reads in which 'pos' is approximately in the middle of the read
    :return : list of reads spanning the given position
    """
    bamit = bam.fetch(chrom, start - 10)
    reads = []
    try:
        read = next(bamit)
        while read.reference_start < end:
            if read.reference_end is not None and read.reference_end > start:
                reads.append(read)
            read = next(bamit)
    except StopIteration:
        pass
    return reads


def encode_with_ref(chrom, pos, ref, alt, bam, fasta, maxreads):
    """
    Fetch reads from the given BAM file, encode them into a single tensor, and also
    fetch & create the corresponding ref sequence and alternate sequence based on the given chrom/pos/ref/alt coords
    :returns: Tuple of encoded reads, reference sequence, alt sequence
    """
    reads = reads_spanning(bam, chrom, pos, max_reads=maxreads)
    if len(reads) < 5:
        raise ValueError(f"Not enough reads spanning {chrom} {pos}, aborting")

    minref = min(alnstart(r) for r in reads)
    maxref = max(alnstart(r) + r.query_length for r in reads)
    reads_encoded, _ = encode_pileup3(reads, minref, maxref)
    pos = pos - 1 # Believe fetch() is zero-based, but input typically in 1-based VCF coords?
    refseq = fasta.fetch(chrom, minref, maxref) 
    assert refseq[pos - minref: pos-minref+len(ref)] == ref, f"Ref sequence / allele mismatch (found {refseq[pos - minref: pos-minref+len(ref)]})"
    altseq = refseq[0:pos - minref] + alt + refseq[pos-minref+len(ref):]
    assert len(refseq) == reads_encoded.shape[0], f"Length of reference sequence doesn't match width of encoded read tensor ({len(refseq)} vs {reads_encoded.shape[0]})"

    ref_encoded = string_to_tensor(refseq)
    encoded_with_ref = torch.cat((ref_encoded.unsqueeze(1), reads_encoded), dim=1)[:, 0:maxreads, :]

    return encoded_with_ref, refseq, altseq


def encode_and_downsample(chrom, pos, ref, alt, bam, fasta, maxreads, num_samples):
    """
    Returns 'num_samples' tuples of read tensors and corresponding reference sequence and alt sequence for the given
    chrom/pos/ref/alt. Each sample is for the same position, but contains a random sample of 'maxreads' from all of the
    reads overlapping the position.
    :param maxreads: Number of reads to downsample to
    :returns: Tuple of encoded reads, reference sequence, alt sequence
    """

    start = np.random.randint(pos - 100, pos - 10)
    end = start + 100
    allreads = reads_spanning_range(bam, chrom, start, end)
    if len(allreads) < 5:
        raise ValueError(f"Not enough reads spanning {chrom} {pos}, aborting")

    if (len(allreads) // maxreads) < num_samples:
        num_samples = max(1, len(allreads) // maxreads)
        # logger.info(f"Only {len(allreads)} reads here, will only return {num_samples} samples")

    pos = pos - 1 # Believe fetch() is zero-based, but input typically in 1-based VCF coords?
    for i in range(num_samples):
        reads = random.sample(allreads, min(len(allreads), maxreads))
        reads = util.sortreads(reads)
        minref = min(alnstart(r) for r in reads)
        maxref = max(alnstart(r) + r.query_length for r in reads)
        reads_encoded, _ = encode_pileup3(reads, minref, maxref)

        refseq = fasta.fetch(chrom, minref, maxref)
        assert refseq[pos - minref: pos-minref+len(ref)] == ref, f"Ref sequence / allele mismatch (found {refseq[pos - minref: pos-minref+len(ref)]})"
        altseq = refseq[0:pos - minref] + alt + refseq[pos-minref+len(ref):]
        if ref == alt:
            assert refseq == altseq, "Ref == alt allele, but ref sequence didn't match alt sequence!"
        assert len(refseq) == reads_encoded.shape[0], f"Length of reference sequence doesn't match width of encoded read tensor ({len(refseq)} vs {reads_encoded.shape[0]})"

        ref_encoded = string_to_tensor(refseq)
        encoded_with_ref = torch.cat((ref_encoded.unsqueeze(1), reads_encoded), dim=1)[:, 0:maxreads, :]

        yield encoded_with_ref, refseq, altseq
