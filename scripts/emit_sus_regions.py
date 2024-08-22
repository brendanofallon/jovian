#!/usr/bin/env python

import sys

sys.path.insert(0, "/uufs/chpc.utah.edu/common/home/u0379426/src/jovian/dnaseq2seq")

from dnaseq2seq import call
from dnaseq2seq import util
import pysam
import random


def main(reference_path, bampath, input_bed):
    for line in open(input_bed).readlines():
        toks = line.strip().split("\t")
        chrom = toks[0]
        window_start = int(toks[1])
        window_end = int(toks[2])
        region_type = toks[3]

        if region_type == "tn" and random.random() < 0.5:
            pos = list(util.cluster_positions(
                call.gen_suspicious_spots(bampath, chrom, window_start, window_end, reference_path, min_indel_count=3, min_mismatch_count=3 ), maxdist=50,
            ))
            if pos:
                print(line.strip() + "-sus")
            else:
                print(line.strip())
        else:
            print(line.strip())


if __name__=="__main__":
    main(*sys.argv[1:])





