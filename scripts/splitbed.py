#!/usr/bin/env python

import sys


targetsize = 300

for line in open(sys.argv[1]):
    toks = line.split("\t")
    start = int(toks[1])
    end = int(toks[2])
    mid = (start + end) // 2
    if end-start < 2*targetsize:
        print(f"{toks[0]}\t{mid - targetsize // 2}\t{mid + targetsize // 2}")
    else:
        p = start + targetsize
        while start < end:
            print(f"{toks[0]}\t{start}\t{start+targetsize}")
            start += targetsize
            
