import torch
import logging
import pysam

import util

logger = logging.getLogger(__name__)



EMPTY_TENSOR = torch.zeros(7)

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
        tensor[0:3] = 0.25
    elif base == '-':
        tensor[0:3] = 0.0
    return tensor


def encode_basecall(base, qual, cigop, clipped):
    ebc = torch.zeros(7)
    ebc = update_from_base(base, ebc)
    ebc[4] = qual / 100 - 0.5
    ebc[5] = cigop
    ebc[6] = 1 if clipped else 0
    return ebc


def decode(t):
    t = t.squeeze()
    if torch.sum(t[0:4]) == 0.0:
        return '-'
    else:
        return util.INDEX_TO_BASE[t[0:4].argmax()]


def encode_cigop(readpos, refpos):
    if readpos == refpos:
        return 0
    elif readpos is None:
        return -1
    elif refpos is None:
        return 1
    return 0


# def variantrec_to_tensor(rec):
#     seq = []
#     for readpos, refpos in rec.get_aligned_pairs():
#         if readpos is not None and refpos is not None:
#             seq.append(encode_basecall(rec.query_sequence[readpos], rec.query_qualities[readpos],
#                                        encode_cigop(readpos, refpos)))
#         elif readpos is None and refpos is not None:
#             seq.append(encode_basecall('-', 50, encode_cigop(readpos, refpos)))  # Deletion
#         elif readpos is not None and refpos is None:
#             seq.append(encode_basecall(rec.query_sequence[readpos], rec.query_qualities[readpos],
#                                        encode_cigop(readpos, refpos)))  # Insertion
#
#     return torch.vstack(seq)



def string_to_tensor(bases):
    return torch.vstack([encode_basecall(b, 50, 0, 0) for b in bases])


def target_string_to_tensor(bases):
    """
    The target version doesn't include the qual or cigop features
    """
    result = torch.tensor([base_index(b) for b in bases]).long()
    return result


def pad_zeros(pre, data, post):
    if pre:
        prepad = torch.zeros(pre, data.shape[-1])
        data = torch.cat((prepad, data))
    if post:
        postpad = torch.zeros(post, data.shape[-1])
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

        # print(f"{base}\t{reftok}\t cig op: {cigop} num bases left in cig op: {n_bases_cigop}")
        encoded_cig = 0
        if is_ref_consumed and is_seq_consumed:
            encoded_cig = 0
        elif is_ref_consumed and not is_seq_consumed:
            encoded_cig = -1
        else:
            encoded_cig = 1
        yield encode_basecall(base, qual, encoded_cig, is_clipped), is_ref_consumed
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
        if is_ref_consumed and is_seq_consumed:
            encoded_cig = 0
        elif is_ref_consumed and not is_seq_consumed:
            encoded_cig = -1
        else:
            encoded_cig = 1
        yield encode_basecall(base, qual, encoded_cig, is_clipped), is_ref_consumed
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
    other return read.reference_start
    """
    if read.cigartuples[0][0] in {4, 5}:
        return read.reference_start - read.cigartuples[0][1]
    else:
        return read.reference_start


def encode_pileup(reads):
    """
    Convert a list of reads (pysam VariantRecords) into a single tensor

    :param reads: List of pysam VariantRecords
    :return: Tensor with shape [position, read, features]
    """
    minref = min(alnstart(r) for r in reads)
    maxref = max(alnstart(r) + r.query_length for r in reads)
    its = [rec_tensor_it(r, minref) for r in reads]
    refpos = minref
    pos_tensors = [next(it) for it in its]
    everything = []
    total_positions = 0
    alldone = False
    while not alldone:
        total_positions += 1
        any_insertion = any(not r[1] for r in pos_tensors)  # r[1] is True if ref is consumed (which implies no insertion)
        thispos = []
        # if any_insertion:
        #     print(f"Found insertion(s) at refpos {refpos} in reads " + ",".join(str(i) for i,r in enumerate(pos_tensors) if not r[1]))
        while any_insertion:
            for i, (it, pos_tensor) in enumerate(zip(its, pos_tensors)):
                if not pos_tensor[1]:
                    thispos.append(pos_tensor[0])
                    pos_tensors[i] = next(it)
                else:
                    thispos.append(EMPTY_TENSOR)

            any_insertion = any(not r[1] for r in pos_tensors)
            assert len(thispos) == len(pos_tensors), "Yikes, this pos somehow has more entries than pos_tensors"
            all_stacked = torch.stack(thispos)
            thispos = []
            everything.append(all_stacked)

        thispos = []
        refpos += 1
        for i, (it, pos_tensor) in enumerate(zip(its, pos_tensors)):
            thispos.append(pos_tensor[0])
            pos_tensors[i] = next(it)
        all_stacked = torch.stack(thispos)
        everything.append(all_stacked)
        alldone = refpos > maxref and all(t.sum() == 0 for t in thispos)
    return torch.stack(everything)


def encode_pileup2(reads):
    """
    Convert a list of reads (pysam VariantRecords) into a single tensor

    :param reads: List of pysam reads
    :return: Tensor with shape [position, read, features]
    """
    minref = min(alnstart(r) for r in reads)
    maxref = max(alnstart(r) + r.query_length for r in reads)
    isalt = ["alt" in r.query_name for r in reads]
    its = [rec_tensor_it(r, minref) for r in reads]
    refpos = minref
    pos_tensors = [next(it) for it in its]
    everything = []
    total_positions = 0
    alldone = False
    while not alldone:
        total_positions += 1
        thispos = []
        refpos += 1
        for i, (it, pos_tensor) in enumerate(zip(its, pos_tensors)):
            thispos.append(pos_tensor[0])
            pos_tensors[i] = next(it)
        all_stacked = torch.stack(thispos)
        everything.append(all_stacked)
        alldone = refpos > maxref and all(t.sum() == 0 for t in thispos)
    return torch.stack(everything), torch.tensor(isalt)

def _consume_n(it, n):
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
    for read in reads:
        try:
            readencoded = [enc for enc, refconsumed in _consume_n(rec_tensor_it(read, start), end-start)]
            everything.append(torch.stack(readencoded))
        except Exception as ex:
            logger.warn(f"Error processing read {read.query_name}: {ex}, skipping it")
            continue

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
        pad = torch.zeros(seqdim - readtensor.shape[0], readtensor.shape[1], readtensor.shape[2])
        readtensor = torch.cat((readtensor, pad), dim=0)

    if readtensor.shape[1] >= readdim:
        readtensor = readtensor[:, 0:readdim, :]
    else:
        pad = torch.zeros(readtensor.shape[0], readdim - readtensor.shape[1], readtensor.shape[2])
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
            if read.reference_start < pos < read.reference_end:
                reads.append(read)
            read = next(bamit)
    except StopIteration:
        pass
    mid = len(reads) // 2
    return reads[max(0, mid-max_reads//2):min(len(reads), mid+max_reads//2)]


def encode_with_ref(chrom, pos, ref, alt, bam, fasta, maxreads):
    """
    Fetch reads from the given BAM file, encode them into a single tensor, and also
    fetch & create the corresponding ref sequence and alternate sequence based on the given chrom/pos/ref/alt coords
    :returns: Tuple of encoded reads, reference sequence, alt sequence
    """
    reads = reads_spanning(bam, chrom, pos, max_reads=maxreads)
    if len(reads) < 5:
        raise ValueError(f"Not enough reads spanning {chrom} {pos}, aborting")

    reads_encoded = encode_pileup2(reads)
    minref = min(alnstart(r) for r in reads)
    pos = pos - 1 # Believe fetch() is zero-based, but input typically in 1-based VCF coords?
    maxref = minref + reads_encoded.shape[0]
    refseq = fasta.fetch(chrom, minref, maxref) 
    assert refseq[pos - minref: pos-minref+len(ref)] == ref, f"Ref sequence / allele mismatch (found {refseq[pos - minref: pos-minref+len(ref)]})"
    altseq = refseq[0:pos - minref] + alt + refseq[pos-minref+len(ref):]
    assert len(refseq) == reads_encoded.shape[0], f"Length of reference sequence doesn't match width of encoded read tensor ({len(refseq)} vs {reads_encoded.shape[0]})"
    return reads_encoded, refseq, altseq


# from util import readstr
# aln = pysam.AlignmentFile("batch78.bam")
# for read in aln.fetch():
#     if read.query_name == "read_hetalt12":
#         encoded_old = list(b for b, _ in iterate_cigar(read))
#         encoded_new = list(b for b, _ in iterate_bases(read))
#         told = torch.stack(encoded_old)
#         tnew = torch.stack(encoded_new)
#         print(readstr(told))
#         print("".join(str(int(x)) for x in told[:, 5]))
#         print(readstr(tnew))
#         print("".join(str(int(x)) for x in tnew[:, 5]))
#         break
