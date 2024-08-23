import time
from intervaltree import Interval, IntervalTree
from collections import defaultdict
import logging
from functools import partial

import pysam

from dnaseq2seq import util

logger = logging.getLogger(__name__)


# Cache for get_sus_positions, initialized in get_sus_positions_cached on first call
readcache = None

def get_sus_positions_cached(read: pysam.AlignedSegment, ref: pysam.FastaFile, min_qual: int=10):
    global readcache

    def cachekey(read):
        if read.cigarstring is not None:
            return read.query_name + read.cigarstring
        else:
            return read.query_name + "None"

    if readcache is None:
        logger.info(f"Initializing read cache for sus_pos_finder")
        pfunc = partial(get_sus_positions, ref=ref, min_qual=min_qual)
        readcache = util.LRUCache(func=pfunc, capacity=1000, key_function=cachekey)

    return readcache[read]


def get_sus_positions(read: pysam.AlignedSegment, ref: pysam.FastaFile, min_qual: int=10):
    """
    Return a list of intervals where the read either has a mismatch with the reference or where there is a gap
    """
    sus_positions = []

    if read.is_secondary or read.is_supplementary or read.is_unmapped:
        return sus_positions, 0, 0

    bt0 = time.perf_counter()
    blocks = read.get_blocks()
    sus_positions.extend(
        [(b0[1], b1[0] + 1) for b0, b1 in zip(blocks[:-1], blocks[1:])]
    )
    bt1 = time.perf_counter()
    # No sequence, so no mismatches
    if read.query_sequence is None:
        return sus_positions, bt1 - bt0, 0

    cigtups = read.cigartuples
    refdist = 0
    readist = 0

    fullrefseq = ref.fetch(read.reference_name, read.reference_start, read.reference_end)

    tq_sum = 0
    for i, (op, length) in enumerate(cigtups):
        if op == 0 and length > 0:
            t0 = time.perf_counter()
            refseq = fullrefseq[refdist:refdist + length]
            readseq = read.query_sequence[readist:readist + length]
            readquals = read.query_qualities[readist:readist + length]


            mismatches = mismatch_pos(refseq, readseq)

            mm_intervals = [(read.reference_start + refdist + i, read.reference_start + refdist + i + 1) for i in
                            mismatches]


            mm_quals = [readquals[i] for i in mismatches]

            # Filter mismatches by min_qual
            flt_mm_intervals = [mm_intervals[i] for i in range(len(mm_intervals)) if mm_quals[i] >= min_qual]

            sus_positions.extend(
                flt_mm_intervals
            )
            refdist += length
            readist += length

            t1 = time.perf_counter()
            tq_sum += t1 - t0

        elif op in {1, 3, 4, 5}:  # Insertion, 'ref skip' or soft-clip
            readist += length
        elif op == 2:  # Deletion
            refdist += length
        else:
            logger.warning(f"Unknown CIGAR operation {op} for read ({read.query_name})")

    # print(f"Block: {bt1 - bt0 :.6f} mismatch_pos: {tq_sum:.6f}")

    return sus_positions, bt1 - bt0, tq_sum


class CountingIntervalTree:
    """
    A class that holds intervals and counts how many times each interval has been added
    """

    def __init__(self):
        self.tree = IntervalTree()
        self.counts = defaultdict(int)

    def add(self, begin, end):
        interval = Interval(begin, end)
        self.tree.add(interval)
        self.counts[interval] += 1  # Increment the count for this interval

    def remove(self, begin, end):
        interval = Interval(begin, end)
        if interval in self.tree:
            self.tree.remove(interval)
            self.counts[interval] -= 1  # Decrement the count
            if self.counts[interval] == 0:
                del self.counts[interval]  # Remove from counts if zero

    def get_count(self, begin, end):
        interval = Interval(begin, end)
        return self.counts[interval]  # Return the count for this interval

    def overlap(self, begin, end):
        interval = Interval(begin, end)
        return self.tree.overlap(interval)

    def __len__(self):
        return sum(v for v in self.counts.values())

    def __iter__(self):
        for interval in self.tree:
            yield interval, self.counts[interval]


def mismatch_pos(seq0, seq1):
    """
    Return all positions that are not equal in seq0, seq1
    """
    return [i for i, (a, b) in enumerate(zip(seq0, seq1)) if a != b]


def build_postree(bam, ref, chrom, start, end):
    reads = bam.fetch(chrom, start, end)
    itree = CountingIntervalTree()
    bsum = 0
    msum = 0
    for i, r in enumerate(reads):
        poslist, bt, mt = get_sus_positions_cached(r, ref)
        bsum += bt
        msum += mt
        for pos in poslist:
            if start < pos[0] <= pos[1] < end:
                itree.add(pos[0], pos[1])

    print(f"Block time sum {bsum:.6f} mm time sum {msum:.6f}")
    return itree