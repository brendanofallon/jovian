#!/usr/bin/env python

import sys

sys.path.insert(0, "/home/brendan/src/dnaseq2seq/dnaseq2seq/")

from dnaseq2seq import call
import pysam

reference_fasta="/home/brendan/Public/genomics/reference/human_g1k_v37_decoy_phiXAdaptr.fasta"
bamfile="/home/brendan/Public/genomics/WGS/99702211631_1ug.bam"
chrom="2"
window_start=25260000
window_end=25370000


def emit(bampath, chrom, window_start,window_end, reference_fasta):
    for start, end in call.cluster_positions(
        call.gen_suspicious_spots(bam, chrom, window_start, window_end, reference_fasta), maxdist=100,
        ):
    print(f"{chrom}\t{start}\t{end}")


def main(reference_path, bampath, input_bed):
    for line in open(input_bed).readlines():
        toks = line.split("\t")
        chrom = toks[0]
        window_start = int(toks[1])
        window_end = int(toks[2])
        
        pos = list(call.cluster_positions(
            call.gen_suspicious_spots(bampath, chrom, window_start, window_end, reference_path), maxdist=100,
        ))
        if pos:
            print(line.strip())


if __name__=="__main__":
    main(*sys.argv[1:])





