#!/usr/bin/env python


import sys
import pysam


"""
Given a 'split' BED file (where every region has the same length, usually 300bp), generate a fourth column
of the BED file that contains a label that can be used for upsampling / downsampling

Right now we do this based on the presence of variants, so regions are labeled according to whether or not
they contain snvs, indels, etc etc. Anything goes, really

"""

vcf = pysam.VariantFile(sys.argv[1])

for line in open(sys.argv[2]):
    toks = line.split("\t")
    start = int(toks[1])
    end = int(toks[2])
    chrom = toks[0]
    variants = list(vcf.fetch(chrom, start, end))
    snv_count = len([v for v in variants if len(v.ref) == 1 and len(v.alts[0]) == 1])
    del_count = len([v for v in variants if len(v.ref) > 1 and len(v.alts[0]) == 1])
    ins_count = len([v for v in variants if len(v.ref) == 1 and len(v.alts[0]) > 1])
    multi_count = len([v for v in variants if len(v.alts) > 1])
     
    if (del_count and ins_count) or (del_count > 1) or (ins_count > 1):
        label = "multi_indel"
    elif (ins_count > 0 or del_count > 0) and snv_count > 0:
        label = "indel-snv"
    elif multi_count:
        label = "multi"
    elif del_count:
        label = "del"
    elif ins_count:
        label = "ins"
    elif snv_count:
        label = "snv"
    else:
        label = "tn"
    print(f"{line.strip()}\t{label}")

