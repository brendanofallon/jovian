
import ssw_aligner
from dataclasses import dataclass

@dataclass
class Variant:
    ref: str
    alt: str
    pos: int


@dataclass
class Cigar:
    op: str
    len: int


def _cigtups(cigstr):
    """
    Convert a cigar string into objects with len, tup fields
    :param cig: cigar string
    :return: List of Cigar objects
    """
    digits = []
    for c in cigstr:
        if c.isdigit():
            digits.append(c)
        else:
            cig = Cigar(op=c, len=int("".join(digits)))
            digits = []
            yield cig

def align_seqs(query, target, offset=0):
    aln = ssw_aligner.local_pairwise_align_ssw(query,
                                     target,
                                     gap_open_penalty=3,
                                     gap_extend_penalty=1,
                                     match_score=2,
                                     mismatch_score=-1)
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
                                  pos=mismatchstart)
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


# for v in align_seqs("AAAAACCCCCTTTTT", "AAAAACGCTTTTT"):
#     print(v)
