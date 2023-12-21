

from functools import partial
import multiprocessing as mp
import itertools

import pysam

import call
import util

def find_sus_regions(bam, bed, reference, threads):
    aln = pysam.AlignmentFile(bam, reference_filename=reference)
    cluster_positions_func = partial(
        call.cluster_positions_for_window,
        bamfile=aln,
        reference_fasta=reference,
        maxdist=100,
    )
    sus_regions = mp.Pool(threads).map(cluster_positions_func, bed)
    sus_regions = util.merge_overlapping_regions(list(itertools.chain(*sus_regions)))
    return sus_regions

