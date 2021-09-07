
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


def aln_to_vars(query, target, offset=0):
    """
    Smith-Watterman align the given sequences and return a generator over Variant objects
    that describe differences between the sequences
    :param query: String of bases
    :param target:String of bases
    :param offset: This amount will be added to each variant position
    :return:
    """
    aln = align_sequences(query, target)
    # print(aln.aligned_query_sequence)
    # print(aln.aligned_target_sequence)
    cigs = list(c for c in _cigtups(aln.cigar))
    if cigs[0].op == "M" and cigs[-1].op == "M":
        # Easy case, both start and end are matches, which means we dont have to deal with edge cases
        mismatches = []
        mismatchstart = None
        for i, (a,b) in enumerate(zip(aln.aligned_query_sequence, aln.aligned_target_sequence)):
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
                          pos=mismatchstart)


