#!/usr/bin/env python

import sys
import pysam

MIN_QUAL=0.10
#MIN_QUAL=25.0 # FB - lots of variability here, for some samples its list 0.07, others may be closer to 50
#MIN_QUAL=50 # HC
#MIN_QUAL=3.0 # DV

vcf = pysam.VariantFile(sys.argv[1])
print(str(vcf.header).strip())
for v in vcf:
    if v.qual > MIN_QUAL:
        v = str(v).strip().split("\t")
        print("\t".join(v[0:6]) + "\tPASS\t" + "\t".join(v[7:]))
