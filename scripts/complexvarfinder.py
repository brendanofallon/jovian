#!/usr/bin/env python

import sys
import pysam

"""
Find indels greater than 10bp where the aligned reads do not agree on the position 
"""

REFGENOME="/uufs/chpc.utah.edu/common/home/u0379426/arup-storage3/Reference/Data/B37/GATKBundle/2.8/human_g1k_v37_decoy_phiXAdaptr.fasta"

def find_indel_starts(aln, chrom, start, stop):
    for col in aln.pileup(chrom, start=start, stop=stop, stepper='nofilter'):
        # The pileup returned by pysam actually starts long before the first start position, but we only want to
        # report positions in the actual requested window
        if start <= col.reference_pos < stop:
            indel_count = 0

            for i, read in enumerate(col.pileups):
                if read.indel != 0:
                    indel_count += 1

                if indel_count > 1:
                    yield col.reference_pos
                    break


def main(vcf, bam):
    min_size = 4
    ref = pysam.Fastafile(REFGENOME)
    aln = pysam.AlignmentFile(bam, reference_filename=REFGENOME)
    vcf = pysam.VariantFile(vcf)
    print(vcf.header, end='')
    for var in vcf:
        if len(var.ref) > min_size or any(len(a) > min_size for a in var.alts):
            indel_starts = [s for s in find_indel_starts(aln, var.chrom, var.pos - 20, var.pos + 20)]
            if len(indel_starts) > 3:
                print(var, end='')


if __name__=="__main__":
    main(*sys.argv[1:])
