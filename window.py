

import pysam
from collections import defaultdict


def gen_suspicious_spots(aln, chrom, start, stop, refseq):
    assert len(refseq) == stop - start, f"Ref sequence length doesn't match start - stop coords"
    for col in aln.pileup(chrom, start=start, stop=stop, stepper='nofilter'):
        # The pileup returned by pysam actually starts long before the first start position, but we only want to
        # report positions in the actual requested window
        if start <= col.reference_pos < stop:
            refbase = refseq[col.reference_pos - start]
            indel_count = 0
            base_mismatches = 0

            for i, read in enumerate(col.pileups):
                if read.indel != 0:
                    indel_count += 1

                if read.query_position is not None:
                    base = read.alignment.query_sequence[read.query_position]
                    if base != refbase:
                        base_mismatches += 1

                if indel_count > 1 or base_mismatches > 2:
                    yield col.reference_pos
                    break



def main(bampath, reference):
    aln = pysam.AlignmentFile(bampath, reference_filename=reference)
    ref = pysam.Fastafile(reference)
    chrom = "21"
    start = 35053900
    stop = 35053950
    refseq = ref.fetch(chrom, start, stop)
    for pos in gen_suspicious_spots(aln, chrom, start, stop, refseq):
        print(f"Found a suspicious spot: {pos}")


def split_large_regions(regions, max_region_size):
    """
    Split any regions greater than max_region_size into regions smaller than max_region_size
    """
    for chrom, start, end in regions:
        while start < end:
            yield chrom, start, min(end, start + max_region_size)
            start += max_region_size


def cluster_positions(poslist, maxdist=500):
    cluster = []
    for pos in poslist:
        if len(cluster) == 0 or pos - min(cluster) < maxdist:
            cluster.append(pos)
        else:
            yield min(cluster), max(cluster)
            cluster = [pos]

    if len(cluster) == 1:
        yield cluster[0] - 10, cluster[0] + 10
    elif len(cluster) > 1:
        yield min(cluster), max(cluster)

if __name__=="__main__":
    pos = [10, 11, 17, 100, 150, 175, 500, 501, 555]
    for start, end in cluster_positions(pos, maxdist=500):
        print(f"{start}-{end}")
    # main("/Volumes/Share/genomics/NIST-002/final.cram",
    #      "/Volumes/Share/genomics/reference/human_g1k_v37_decoy_phiXAdaptr.fasta")
