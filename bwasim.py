
import numpy as np
import logging
import scipy.stats as stats
import string
import torch
import torch.nn.functional as F

from bam import reads_spanning, encode_pileup2, target_string_to_tensor, ensure_dim
import random
import pysam
import subprocess


logger = logging.getLogger(__name__)

def run_bwa(fastq1, fastq2, refgenome, dest):
    cmd = f"bwa mem -t 2 {refgenome} {fastq1} {fastq2} | samtools sort - | samtools view -b - -o {dest}"
    logger.info(f"Executing {cmd}")
    subprocess.run(cmd, shell=True)
    subprocess.run(f"samtools index {dest}", shell=True)


def revcomp(seq):
    d = {
        'A': 'T',
        'T': 'A',
        'C': 'G',
        'G': 'C',
    }
    return "".join(d[b] for b in reversed(seq))


def generate_reads(seq, numreads, readlength, fragsize, error_rate, clip_prob):
    assert fragsize < len(seq), "Fragment size cant be greater than sequence length"
    for i in range(numreads):
        template_len = int(np.random.normal(fragsize, fragsize/10))
        start = np.random.randint(0, max(1, len(seq)-template_len))
        read1 = seq[start:min(len(seq), start+readlength)]
        read2 = seq[max(0, start+template_len-readlength):min(len(seq), start+template_len)]
        if error_rate:
            read1 = mutate_seq(read1, error_rate)
            read2 = mutate_seq(read2, error_rate)
        yield read1, read2


def to_fastq(read, label, idx):
    return f"@read_{label}{idx}\n{read}\n+\n" +'F' * len(read) + "\n"


def fq_from_seq(seq, numreads, readlength, fragsize, read_label, error_rate, clip_prob):
    for i, (r1, r2) in enumerate(generate_reads(seq, numreads, readlength, fragsize, error_rate, clip_prob)):
        r2 = revcomp(r2)
        yield to_fastq(r1, read_label, i), to_fastq(r2, read_label, i)


def fq_to_file(seq, numreads, readlength, fragsize, prefix, read_label, error_rate, clip_prob):
    fq1path = f"{prefix}_R1.fq"
    fq2path = f"{prefix}_R2.fq"
    with open(fq1path, "a") as fq1, open(fq2path, "a") as fq2:
        for r1, r2 in fq_from_seq(seq, numreads, readlength, fragsize, read_label, error_rate, clip_prob):
            fq1.write(r1)
            fq2.write(r2)
    return fq1path, fq2path


def bgzip(path):
    subprocess.run(f"bgzip {path}", shell=True)
    return str(path) + ".gz"


def var2fastq(seq, altseq, readlength, totreads, prefix, vaf=0.5, fragsize=200, error_rate=0, clip_prob=0):
    num_altreads = stats.binom(totreads - 1, vaf).rvs(1)[0] + 1
    fq_to_file(seq, totreads - num_altreads, readlength, fragsize, prefix, "ref", error_rate, clip_prob)
    fq1, fq2 = fq_to_file(altseq, num_altreads, readlength, fragsize, prefix, "hetalt", error_rate, clip_prob)
    return fq1, fq2


def mutate_seq(seq, error_rate):
    """
    Randomly (uniformly) alter the given sequence at error_rate fraction of positions
    :param seq: Input sequence of bases
    :param error_rate: Fraction of bases to alter
    :return: New, altered sequence
    """
    if error_rate == 0:
        return seq
    n_muts = np.random.poisson(error_rate * len(seq))
    if n_muts == 0:
        return seq
    output = list(seq)
    for i in range(n_muts):
        which = np.random.randint(0, len(seq))
        c = random.choice('ACTG')
        while c == output[which]:
            c = random.choice('ACTG')
        output[which] = c

    return "".join(output)


def make_het_snv(seq, readlength, totreads, vaf, prefix, fragment_size, error_rate=0, clip_prob=0):
    snvpos = random.choice(range(max(0, len(seq) // 2 - 25), min(len(seq), len(seq) // 2 + 25)))
    altseq = list(seq)
    altseq[snvpos] = random.choice('ACTG')
    while altseq[snvpos] == seq[snvpos]:
        altseq[snvpos] = random.choice('ACTG')
    altseq = "".join(altseq)
    fq1, fq2 = var2fastq(seq, altseq, readlength, totreads, prefix=prefix, vaf=vaf, fragsize=fragment_size, error_rate=error_rate, clip_prob=clip_prob)
    return fq1, fq2, altseq, vaf


def make_batch(batch_size, regions, refpath, numreads, readlength, error_rate, clip_prob):
    suf = "".join(random.choices(string.ascii_lowercase + string.ascii_uppercase, k=6))
    prefix = "simfqs" # + suf
    refgenome = pysam.FastaFile(refpath)
    region_size = 200
    fragment_size = 150
    var_info = [] # Stores region, altseq, and vaf for each variant created
    for region in random.sample(regions, k=batch_size):
        pos = np.random.randint(region[1], region[2])
        seq = refgenome.fetch(region[0], pos-region_size//2, pos+region_size//2)
        fq1, fq2, altseq, vaf = make_het_snv(seq, readlength, numreads, vaf=0.5, prefix=prefix, fragment_size=fragment_size, error_rate=error_rate, clip_prob=clip_prob)
        var_info.append((region[0], pos, altseq, vaf))

    fq1 = bgzip(fq1)
    fq2 = bgzip(fq2)
    bamdest = "batch78.bam"
    run_bwa(fq1, fq2, refpath, bamdest)
    bam = pysam.AlignmentFile(bamdest)
    max_reads = 2*numreads

    src = []
    tgt = []
    vafs = []
    altmasks = []
    for chrom, pos, altseq, vaf in var_info:
        #print(f"Encoding tensors around {chrom}:{pos}")
        reads = reads_spanning(bam, chrom, pos, max_reads=max_reads)
        if len(reads) < numreads // 10:
            raise ValueError(f"Not enough reads spanning {chrom} {pos}, aborting")
        reads_encoded, altmask = encode_pileup2(reads)
        reads = ensure_dim(reads_encoded, region_size, numreads)
        src.append(reads)
        tgt.append(target_string_to_tensor(altseq))
        vafs.append(vaf)
        altmasks.append(F.pad(altmask, (0,numreads-altmask.shape[0])))
        #print(f"reads: {src[-1].shape}, alt: {tgt[-1].shape} vafs: {vafs[-1]} altmask: {altmasks[-1].shape}")

    return torch.stack(src), torch.stack(tgt), torch.tensor(vafs), torch.stack(altmasks)


def load_regions(regionsbed):
    regions = []
    with open(regionsbed) as fh:
        for line in fh:
            toks = line.split('\t')
            regions.append((toks[0], int(toks[1]), int(toks[2])))
    return regions




def main():
    refpath = "/home/brendan/Public/genomics/reference/human_g1k_v37_decoy_phiXAdaptr.fasta"
    #refpath = "/Volumes/Share/genomics/reference/human_g1k_v37_decoy_phiXAdaptr.fasta"
    # ref = pysam.FastaFile(refpath)
    # seq = ref.fetch("2", 73612900, 73613200)
    # fq1, fq2, altseq, vaf = make_het_snv(seq, 150, 100, 0.5, prefix="myhetsnv")
    # fq1 = bgzip(fq1)
    # fq2 = bgzip(fq2)
    # print(f"Saved results to {fq1}, {fq2}")
    regions = load_regions("cds100.bed")

    src, tgt, vafs = make_batch(10, regions, refpath, numreads=100, readlength=150, error_rate=0.005, clip_prob=0)
    print(f"src: {src.shape}")
    print(f"tgt: {tgt.shape}")


