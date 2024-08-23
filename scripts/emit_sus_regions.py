#!/usr/bin/env python

import sys

sys.path.insert(0, "/uufs/chpc.utah.edu/common/home/u0379426/src/jovian/dnaseq2seq")

from dnaseq2seq import call, util
import pysam
import random
import time


def main(reference_path, bampath, input_bed):
    t0 = time.perf_counter()
    lines_counted = 0
    for line in open(input_bed).readlines():
        toks = line.strip().split("\t")
        chrom = toks[0]
        window_start = int(toks[1])
        window_end = int(toks[2])
        region_type = toks[3]

        if region_type == "tn" and random.random() < 0.25:
            pos = list(util.cluster_positions(
                call.gen_suspicious_spots(bampath, chrom, window_start, window_end, reference_path, min_indel_count=3, min_mismatch_count=3 ), maxdist=50,
            ))
            if pos:
                print(line.strip() + "-sus")
            else:
                print(line.strip())
        else:
            print(line.strip())
        lines_counted += 1
        if lines_counted == 100:
            elapsed = time.perf_counter() - t0
            lines_per_second = lines_counted / elapsed
            sys.stderr.write(f"Lines per second: {lines_per_second :.3f}\n")
            sys.stderr.flush()
            t0 = time.perf_counter()
            lines_counted = 0


if __name__=="__main__":
    main(*sys.argv[1:])





