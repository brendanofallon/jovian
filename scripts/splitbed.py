#!/usr/bin/env python

import sys

"""
Splits a BED file into chunks of size *targetsize* 
You probably want to do this to all BED files prior to generating 'labels' BED files for pregen
"""

targetsize = 300

for line in open(sys.argv[1]):
    toks = line.split("\t")
    start = int(toks[1])
    end = int(toks[2])
    mid = (start + end) // 2

    # If the region is smaller than 2x targetsize, just put the region in the middle?
    if end-start < 2*targetsize:
        print(f"{toks[0]}\t{mid - targetsize // 2}\t{mid + targetsize // 2}")

    # If the region is bigger than the target size, create regions that tile across it
    # Maybe these should be overlapping a little? 
    else:
        p = start + targetsize
        while start < end:
            print(f"{toks[0]}\t{start}\t{start+targetsize}")
            start += targetsize
            