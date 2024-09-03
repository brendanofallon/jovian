#!/usr/bin/env python

import sys
import pysam

# This forces the VCF Filter field to be PASS for all variants with quality greater than 0.01 and LOWQUAL for all others

MIN_QUAL=0.10

vcf = pysam.VariantFile(sys.argv[1])
print(str(vcf.header).strip())
for v in vcf:
    v = str(v).strip().split("\t")
    if v.qual > MIN_QUAL:
        print("\t".join(v[0:6]) + "\tPASS\t" + "\t".join(v[7:]))
    else:
        print("\t".join(v[0:6]) + "\tLOWQUAL\t" + "\t".join(v[7:]))
