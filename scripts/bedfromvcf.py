import pysam
import sys
import random

def emit_regions(chrom: str, pos: int, window_size: int = 150):
    start = pos - random.randint(60, 120)
    yield (chrom, start, start + window_size)
    start = pos - random.randint(10, 70)
    yield (chrom, start, start + window_size)

def is_FP(rec):
    return rec.samples['QUERY']['BD'] == 'FP'

def is_FN(rec):
    return rec.samples['TRUTH']['BD'] == 'FN'

def main(vcfpath: str):
    for rec in pysam.VariantFile(vcfpath):
        if is_FP(rec):
            for region in emit_regions(rec.chrom, rec.pos):
                print("\t".join((str(r) for r in region)) + "\ttn-fp")
        elif is_FN(rec):
            for region in emit_regions(rec.chrom, rec.pos):
                print("\t".join((str(r) for r in region)) + "\tfn")



if __name__=="__main__":
    main(sys.argv[1])
