
import ssw_aligner
from dataclasses import dataclass

@dataclass
class Variant:
    ref: str
    alt: str
    pos: int
    qual: float

    def __eq__(self, other):
        return self.ref == other.ref and self.alt == other.alt and self.pos == other.pos


@dataclass
class Cigar:
    op: str
    len: int


def _cigtups(cigstr):
    """
    Generator for Cigar objects from a cigar string
    :param cig: cigar string
    :return: Generator of Cigar objects
    """
    digits = []
    for c in cigstr:
        if c.isdigit():
            digits.append(c)
        else:
            cig = Cigar(op=c, len=int("".join(digits)))
            digits = []
            yield cig


def align_sequences(query, target):
    """
    Return Smith-Watterman alignment of both sequences
    """
    aln = ssw_aligner.local_pairwise_align_ssw(query,
                                               target,
                                               gap_open_penalty=3,
                                               gap_extend_penalty=1,
                                               match_score=2,
                                               mismatch_score=-1)
    return aln

def _mismatches_to_vars(query, target, offset):
    """
    Zip both sequences and look for mismatches, if any are found convert them to Variant objects
    and return them
    This is for finding variants that are inside an "Match" region according to the cigar from an alignment result
    :returns: Generator over Variants from the paired sequences
    """
    mismatches = []
    mismatchstart = None
    for i, (a, b) in enumerate(zip(query, target)):
        if a == b:
            if mismatches:
                yield Variant(ref="".join(mismatches[0]).replace("-", ""),
                              alt="".join(mismatches[1]).replace("-", ""),
                              pos=mismatchstart,
                              qual=-1)
            mismatches = []
        else:
            if mismatches:
                mismatches[0] += a
                mismatches[1] += b
            else:
                mismatches = [a, b]
                mismatchstart = i + offset

    # Could be mismatches at the end
    if mismatches:
        yield Variant(ref="".join(mismatches[0]).replace("-", ""),
                      alt="".join(mismatches[1]).replace("-", ""),
                      pos=mismatchstart,
                      qual=-1)

def aln_to_vars(refseq, altseq, offset=0):
    """
    Smith-Watterman align the given sequences and return a generator over Variant objects
    that describe differences between the sequences
    :param refseq: String of bases representing reference sequence
    :param altseq: String of bases representing alt sequence
    :param offset: This amount will be added to each variant position
    :return: Generator over variants
    """
    aln = align_sequences(altseq, refseq)
    # print(aln.aligned_query_sequence)
    # print(aln.aligned_target_sequence)
    # cigs = list(c for c in _cigtups(aln.cigar))
    q_offset = 0
    t_offset = 0

    if aln.query_begin > 0:
        # yield Variant(ref='', alt=altseq[0:aln.query_begin], pos=offset, qual=-1)
        q_offset += aln.query_begin
    if aln.target_begin > 0:
        # yield Variant(ref=refseq[0:aln.target_begin], alt='', pos=offset, qual=-1)
        t_offset += aln.target_begin

    for cig in _cigtups(aln.cigar):
        if cig.op == "M":
            for v in _mismatches_to_vars(refseq[t_offset:t_offset+cig.len], altseq[q_offset:q_offset+cig.len], offset + t_offset):
                yield v
            q_offset += cig.len
            t_offset += cig.len

        elif cig.op == "I":
            yield Variant(ref='', alt=altseq[q_offset:q_offset+cig.len], pos=offset + q_offset, qual=-1)
            q_offset += cig.len

        elif cig.op == "D":
            yield Variant(ref=refseq[t_offset:t_offset + cig.len], alt='', pos=offset + t_offset, qual=-1)
            t_offset += cig.len

    # if aln.query_end+1 < len(altseq):
    #     yield Variant(ref='', alt=altseq[aln.query_end+1:], pos=offset + q_offset+1, qual=-1)
    # if aln.target_end_optimal+1 < len(refseq):
    #     yield Variant(ref=refseq[aln.target_end_optimal+1:], alt='', pos=offset + 1, qual=-1)


# ref = "ACCCTTTGGGAAAGCGCGCGC"
# alt = "AAACCCTTTGCGAAA"
# for v in aln_to_vars(ref, alt):
#     print(v)

