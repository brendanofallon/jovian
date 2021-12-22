
import numpy as np
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

    def __hash__(self):
        return hash(f"{self.ref}&{self.alt}&{self.pos}&{self.qual}")

    def __gt__(self, other):
        return self.pos > other.pos

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


def _geomean(probs):
    return np.exp(np.log(probs).mean())

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

def _mismatches_to_vars(query, target, offset, probs):
    """
    Zip both sequences and look for mismatches, if any are found convert them to Variant objects
    and return them
    This is for finding variants that are inside an "Match" region according to the cigar from an alignment result
    :returns: Generator over Variants from the paired sequences
    """
    mismatches = []
    mismatch_quals = []
    mismatchstart = None
    for i, (a, b) in enumerate(zip(query, target)):
        if a == b:
            if mismatches:
                yield Variant(ref="".join(mismatches[0]).replace("-", ""),
                              alt="".join(mismatches[1]).replace("-", ""),
                              pos=mismatchstart,
                              qual=_geomean(mismatch_quals))  # Geometric mean?
            mismatches = []
            mismatch_quals = []
        else:
            if mismatches:
                mismatches[0] += a
                mismatches[1] += b
                mismatch_quals.append(probs[i])
            else:
                mismatches = [a, b]
                mismatch_quals = [probs[i]]
                mismatchstart = i + offset

    # Could be mismatches at the end
    if mismatches:
        yield Variant(ref="".join(mismatches[0]).replace("-", ""),
                      alt="".join(mismatches[1]).replace("-", ""),
                      pos=mismatchstart,
                      qual=_geomean(mismatch_quals))

def aln_to_vars(refseq, altseq, offset=0, probs=None):
    """
    Smith-Watterman align the given sequences and return a generator over Variant objects
    that describe differences between the sequences
    :param refseq: String of bases representing reference sequence
    :param altseq: String of bases representing alt sequence
    :param offset: This amount will be added to each variant position
    :return: Generator over variants
    """
    if probs is not None:
        assert len(probs) == len(altseq), f"Probabilities must contain same number of elements as alt sequence"
    else:
        probs = np.ones(len(altseq))
    aln = align_sequences(altseq, refseq)
    q_offset = 0
    t_offset = 0

    variant_pos_offset = 0
    if aln.query_begin > 0:
        # yield Variant(ref='', alt=altseq[0:aln.query_begin], pos=offset, qual=-1)
        q_offset += aln.query_begin # Maybe we don't want this?
    if aln.target_begin > 0:
        # yield Variant(ref=refseq[0:aln.target_begin], alt='', pos=offset, qual=-1)
        t_offset += aln.target_begin

    for cig in _cigtups(aln.cigar):
        if cig.op == "M":
            for v in _mismatches_to_vars(
                    refseq[t_offset:t_offset+cig.len],
                    altseq[q_offset:q_offset+cig.len],
                    offset + t_offset,
                    probs[q_offset:q_offset+cig.len]):
                yield v
            q_offset += cig.len
            variant_pos_offset += cig.len
            t_offset += cig.len

        elif cig.op == "I":
            yield Variant(ref='',
                          alt=altseq[q_offset:q_offset+cig.len],
                          pos=offset + variant_pos_offset,
                          qual=_geomean(probs[q_offset:q_offset+cig.len]))
            q_offset += cig.len
            variant_pos_offset += cig.len

        elif cig.op == "D":
            yield Variant(ref=refseq[t_offset:t_offset + cig.len],
                          alt='',
                          pos=offset + t_offset,
                          qual=_geomean(probs[q_offset-1:q_offset+cig.len]))  # Is this right??
            t_offset += cig.len

    # if aln.query_end+1 < len(altseq):
    #     yield Variant(ref='', alt=altseq[aln.query_end+1:], pos=offset + q_offset+1, qual=-1)
    # if aln.target_end_optimal+1 < len(refseq):
    #     yield Variant(ref=refseq[aln.target_end_optimal+1:], alt='', pos=offset + 1, qual=-1)


# import numpy as np
# ref = "ACTGACTG"
# alt = "ACTGCTG"
# probs = np.arange(7) * 0.1
# for v in aln_to_vars(ref, alt, probs=probs):
#     print(v)

