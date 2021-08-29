
import numpy as np
import scipy.stats as stats
import random
import pysam
import subprocess


def run_bwa(fastq1, fastq2, refgenome, dest):
    cmd = f"bwa mem -t 2 {refgenome} {fastq1} {fastq2} | samtools --threads 4 sort - | samtools view -b - -o {dest}"
    subprocess.run(cmd, shell=True)


def revcomp(seq):
    d = {
        'A': 'T',
        'T': 'A',
        'C': 'G',
        'G': 'C',
    }
    return "".join(d[b] for b in reversed(seq))


def generate_reads(seq, numreads, readlength, fragsize):
    assert fragsize < len(seq), "Fragment size cant be greater than sequence length"
    for i in range(numreads):
        template_len = int(np.random.normal(fragsize, fragsize/10))
        start = np.random.randint(0, max(1, len(seq)-template_len))
        read1 = seq[start:min(len(seq), start+readlength)]
        read2 = seq[max(0, start+template_len-readlength):min(len(seq), start+template_len)]
        yield read1, read2


def to_fastq(read, idx):
    return f"@read{idx}\n{read}\n+\n" +'F' * len(read) + "\n"


def fq_from_seq(seq, numreads, readlength, fragsize):
    for i, (r1, r2) in enumerate(generate_reads(seq, numreads, readlength, fragsize)):
        r2 = revcomp(r2)
        yield to_fastq(r1, i), to_fastq(r2, i)


def fq_to_file(seq, numreads, readlength, fragsize, prefix):
    fq1path = f"{prefix}_R1.fq"
    fq2path = f"{prefix}_R2.fq"
    with open(fq1path, "a") as fq1, open(fq2path, "a") as fq2:
        for r1, r2 in fq_from_seq(seq, numreads, readlength, fragsize):
            fq1.write(r1)
            fq2.write(r2)
    return fq1path, fq2path


def bgzip(path):
    subprocess.run(f"bgzip {path}", shell=True)
    return str(path) + ".gz"


def var2fastq(seq, altseq, readlength, totreads, prefix, vaf=0.5, fragsize=250, error_rate=0, clip_prob=0):
    num_altreads = stats.binom(totreads - 1, vaf).rvs(1)[0] + 1
    fq_to_file(seq, totreads - num_altreads, readlength, fragsize, prefix)
    fq1, fq2 = fq_to_file(altseq, num_altreads, readlength, fragsize, prefix)
    return fq1, fq2



def make_het_snv(seq, readlength, totreads, vaf, prefix, error_rate=0, clip_prob=0):
    snvpos = random.choice(range(max(0, len(seq) // 2 - 25), min(len(seq), len(seq) // 2 + 25)))
    altseq = list(seq)
    altseq[snvpos] = random.choice('ACTG')
    while altseq[snvpos] == seq[snvpos]:
        altseq[snvpos] = random.choice('ACTG')
    altseq = "".join(altseq)
    fq1, fq2 = var2fastq(seq, altseq, readlength, totreads, prefix=prefix, vaf=vaf)
    return fq1, fq2, altseq, vaf


def main():
    #ref = pysam.FastaFile("/Volumes/Share/genomics/reference/human_g1k_v37_decoy_phiXAdaptr.fasta")
    ref = pysam.FastaFile("/home/brendan/Public/genomics/reference/human_g1k_v37_decoy_phiXAdaptr.fasta")
    seq = ref.fetch("2", 73612900, 73613200)
    fq1, fq2, altseq, vaf = make_het_snv(seq, 150, 100, 0.5, prefix="myhetsnv")
    fq1 = bgzip(fq1)
    fq2 = bgzip(fq2)
    print(f"Saved results to {fq1}, {fq2}")


main()