
import numpy as np
import logging
import scipy.stats as stats
import string
import torch
import torch.nn.functional as F

from bam import reads_spanning, target_string_to_tensor, ensure_dim, string_to_tensor, encode_pileup3
import random
import pysam
import subprocess


logger = logging.getLogger(__name__)

def run_bwa(fastq1, fastq2, refgenome, dest):
    cmd = f"bwa mem -t 2 {refgenome} {fastq1} {fastq2} 2> /dev/null | samtools sort - | samtools view -b - -o {dest}"
    #logger.info(f"Executing {cmd}")
    subprocess.run(cmd, shell=True, stderr=None)
    subprocess.run(f"samtools index {dest}", shell=True)


def revcomp(seq):
    d = {
        'A': 'T',
        'T': 'A',
        'C': 'G',
        'G': 'C',
        'N': 'N',
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
            read1 = mutate_chunk(mutate_seq(read1, error_rate), clip_prob)
            read2 = mutate_chunk(mutate_seq(read2, error_rate), clip_prob)
        yield read1, read2


def to_fastq(read, label, idx):
    quals = "".join(random.choices('F:,!'), k=len(read), weights=[84, 10, 5, 1])
    return f"@read_{label}{idx}\n{read}\n+\n" + quals + "\n"


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
    subprocess.run(f"bgzip -f {path}", shell=True)
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


def mutate_chunk(seq, clip_prob):
    if clip_prob == 0 or np.random.rand() > clip_prob:
        return seq

    num_bases = np.random.randint(1, 30)
    if np.random.rand() < 0.5:

        seq = "".join(random.choice('ACTG') for _ in range(num_bases)) + seq[num_bases:]
    else:
        seq = seq[0:len(seq)-num_bases] + "".join(random.choice('ACTG') for _ in range(num_bases))
    return seq

def make_het_snv(seq, readlength, totreads, vaf, prefix, fragment_size, error_rate=0, clip_prob=0):
    snvpos = random.choice(range(max(0, len(seq) // 2 - 25), min(len(seq), len(seq) // 2 + 25)))
    altseq = list(seq)
    altseq[snvpos] = random.choice('ACTG')
    while altseq[snvpos] == seq[snvpos]:
        altseq[snvpos] = random.choice('ACTG')
    altseq = "".join(altseq)
    fq1, fq2 = var2fastq(seq, altseq, readlength, totreads, prefix=prefix, vaf=vaf, fragsize=fragment_size, error_rate=error_rate, clip_prob=clip_prob)
    return fq1, fq2, altseq, vaf, snvpos


def make_het_del(seq, readlength, totreads, vaf, prefix, fragment_size, error_rate=0, clip_prob=0):
    del_len = random.choice(range(min(len(seq)-1, 10)))
    delpos = random.choice(range(max(0, len(seq) // 2 - 8), min(len(seq) - del_len, len(seq) // 2 + 8)))
    ls = list(seq)
    for i in range(del_len):
        del ls[delpos]
    altseq = "".join(ls + ["A"] * del_len)
    fq1, fq2 = var2fastq(seq, altseq, readlength, totreads, prefix=prefix, vaf=vaf, fragsize=fragment_size, error_rate=error_rate, clip_prob=clip_prob)
    return fq1, fq2, altseq, vaf, delpos


def make_het_ins(seq, readlength, totreads, vaf, prefix, fragment_size, error_rate=0, clip_prob=0):
    ins_len = random.choice(range(min(len(seq)-1, 10))) + 1
    inspos = random.choice(range(max(0, len(seq) // 2 - 8), min(len(seq) - ins_len, len(seq) // 2 + 8)))
    altseq = "".join(seq[0:inspos]) + "".join(random.choices("ACTG", k=ins_len)) + "".join(seq[inspos:-ins_len])
    altseq = altseq[0:len(seq)]
    fq1, fq2 = var2fastq(seq, altseq, readlength, totreads, prefix=prefix, vaf=vaf, fragsize=fragment_size,
                         error_rate=error_rate, clip_prob=clip_prob)
    return fq1, fq2, altseq, vaf, inspos


def make_mnv(seq, readlength, totreads, vaf, prefix, fragment_size, error_rate=0, clip_prob=0):
    del_len = random.choice(range(min(len(seq)-1, 15))) + 1
    delpos = random.choice(range(max(0, len(seq) // 2 - 8), min(len(seq) - del_len, len(seq) // 2 + 8)))
    ls = list(seq)
    for i in range(del_len):
        del ls[delpos]

    ins_len = random.choice(range(15)) + 1
    altseq = "".join(ls[0:delpos]) + "".join(random.choices("ACTG", k=ins_len)) + "".join(ls[delpos:-ins_len])
    altseq = altseq[0:len(seq)]
    altseq = altseq + "A" * (len(seq) - len(altseq))
    fq1, fq2 = var2fastq(seq, altseq, readlength, totreads, prefix=prefix, vaf=vaf, fragsize=fragment_size,
                         error_rate=error_rate, clip_prob=clip_prob)
    return fq1, fq2, altseq, vaf, delpos


def make_novar(seq, readlength, totreads, vaf, prefix, fragment_size, error_rate=0, clip_prob=0):
    fq1, fq2 = var2fastq(seq, seq, readlength, totreads, prefix=prefix, vaf=0.000001, fragsize=fragment_size,
                         error_rate=error_rate, clip_prob=clip_prob)
    return fq1, fq2, seq, vaf, len(seq)//2


def vaf_uniform(lower=0.1, upper=1.0):
    """ Generates uniform random vars betwen upper and lower """
    assert upper <= 1.0
    assert lower > 0.0
    return np.random.uniform(low=lower, high=upper)


def betavaf():
    if np.random.rand() < 0.1:
        return 1.0
    else:
        return stats.beta(a=1.0, b=5.0).rvs(1)[0]


def make_batch(batch_size, regions, refpath, numreads, readlength, var_funcs=None, vaf_func=betavaf, weights=None, error_rate=0.01, clip_prob=0):
    prefix = "simfqs"
    refgenome = pysam.FastaFile(refpath)
    region_size = 2 * readlength
    fragment_size = int(1.5 * readlength)
    if var_funcs is None:
        var_funcs = [
            make_novar,
            make_het_del,
            make_het_snv,
            make_het_ins,
            make_mnv
        ]
    if weights is None:
        weights = np.ones(len(var_funcs))

    var_info = [] # Stores region, altseq, and vaf for each variant created
    for i, region in enumerate(random.sample(regions, k=batch_size)):
        var_func = random.choices(var_funcs, weights=weights)[0]
        pos = np.random.randint(region[1], region[2])
        seq = refgenome.fetch(region[0], pos-region_size//2, pos+region_size//2)
        vaf = vaf_func()
        fq1, fq2, altseq, vaf, varpos = var_func(seq, readlength, numreads, vaf=vaf, prefix=prefix, fragment_size=fragment_size, error_rate=error_rate, clip_prob=clip_prob)
        ns = sum(1 if b=='N' else 0 for b in seq)
        if 0 < ns < 10:
            logger.warning(f"Replacing {ns} Ns with As near position {region[0]}:{pos}")
            seq = seq.replace('N', 'A')
        elif ns >= 10:
            logger.warning(f"Skipping {regions[0]}:{pos}, too many Ns ({ns})")
            continue
        var_info.append((region[0], pos-region_size//2, pos+region_size//2, seq, altseq, vaf, varpos + pos-region_size//2))
        #logger.info(f"Item #{i}: {region[0]}:{pos-region_size//2}-{pos+region_size//2} alt: {altseq}")

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
    for chrom, region_start, region_end, refseq, altseq, vaf, varpos in var_info:
        #print(f"Encoding tensors around {chrom}:{pos}")
        reftensor = string_to_tensor(refseq)
        reads = reads_spanning(bam, chrom, varpos, max_reads=max_reads)
        if len(reads) < 3:
            logger.warning(f"Not enough reads spanning {chrom}:{(region_start+region_end)//2}, skipping")
            continue
        reads_encoded, altmask = encode_pileup3(reads, region_start, region_end)
        reads_w_ref = torch.cat((reftensor.unsqueeze(1), reads_encoded), dim=1)[:, 0:numreads, :]
        altmask = torch.cat((torch.Tensor([True]), altmask)) # Extra False entry for Ref seq, for alt mask purposes we treat it as true, since we don't want it to be masked?
        padded_reads = ensure_dim(reads_w_ref, region_size, numreads)

        src.append(padded_reads)
        tgt.append(target_string_to_tensor(altseq))
        vafs.append(vaf)
        altmasks.append(F.pad(altmask, (0, numreads-altmask.shape[0])))
        #print(f"reads: {src[-1].shape}, alt: {tgt[-1].shape} vafs: {vafs[-1]} altmask: {altmasks[-1].shape}")

    return torch.stack(src), torch.stack(tgt), torch.tensor(vafs), torch.stack(altmasks)


def load_regions(regionsbed):
    regions = []
    with open(regionsbed) as fh:
        for line in fh:
            toks = line.split('\t')
            regions.append((toks[0], int(toks[1]), int(toks[2])))
    return regions


def make_mixed_batch(size, seqlen, readsperbatch, readlength, error_rate, clip_prob):
    """
    Make a batch of training data that is a mixture of different variant types
    :param size: Total size of batch
    :param seqlen: Reference sequence length per data point
    :param readsperbatch: Number of reads in each pileup
    :param readlength: Length of each read in pileup
    :param error_rate: Per-base error fraction
    :param clip_prob: Per-read probability of having some soft-clipping
    :return: source data tensor, target data tensor
    """
    snv_w = 9 # Bigger values here equal less variance among sizes
    del_w = 10
    ins_w = 7
    mnv_w = 5
    novar_w = 10
    mix = np.ceil(np.random.dirichlet((snv_w, del_w, ins_w, mnv_w, novar_w)) * size) # Ceil because zero sizes break things
    snv_src, snv_tgt, snv_vaf_tgt, snv_altmask = make_batch(int(mix[0]), seqlen, readsperbatch, readlength, make_het_snv, error_rate, clip_prob)
    del_src, del_tgt, del_vaf_tgt, del_altmask = make_batch(int(mix[1]), seqlen, readsperbatch, readlength, make_het_del, error_rate, clip_prob)
    ins_src, ins_tgt, ins_vaf_tgt, ins_altmask = make_batch(int(mix[2]), seqlen, readsperbatch, readlength, make_het_ins, error_rate, clip_prob)
    mnv_src, mnv_tgt, mnv_vaf_tgt, mnv_altmask = make_batch(int(mix[3]), seqlen, readsperbatch, readlength, make_mnv, error_rate, clip_prob)
    novar_src, novar_tgt, novar_vaf_tgt, novar_altmask = make_batch(int(mix[4]), seqlen, readsperbatch, readlength, make_novar, error_rate, clip_prob)
    return (torch.cat((snv_src, del_src, ins_src, mnv_src, novar_src)),
           torch.cat((snv_tgt, del_tgt, ins_tgt, mnv_tgt, novar_tgt)),
           torch.cat((snv_vaf_tgt, del_vaf_tgt, ins_vaf_tgt, mnv_vaf_tgt, novar_vaf_tgt)),
           torch.cat((snv_altmask, del_altmask, ins_altmask, mnv_altmask, novar_altmask)))




# def main():
#     # refpath = "/home/brendan/Public/genomics/reference/human_g1k_v37_decoy_phiXAdaptr.fasta"
#     refpath = "/Volumes/Share/genomics/reference/human_g1k_v37_decoy_phiXAdaptr.fasta"
#     # ref = pysam.FastaFile(refpath)
#     # seq = ref.fetch("2", 73612900, 73613200)
#     # fq1, fq2, altseq, vaf = make_het_snv(seq, 150, 100, 0.5, prefix="myhetsnv")
#     # fq1 = bgzip(fq1)
#     # fq2 = bgzip(fq2)
#     # print(f"Saved results to {fq1}, {fq2}")
#     regions = load_regions("cds100.bed")
#
#     regions = regions[0:5]
#     for r in regions:
#         print(r)
#     src, tgt, vafs = make_batch(5, regions, refpath, numreads=100, readlength=150, error_rate=0.02, clip_prob=0.02)
#     print(f"src: {src.shape}")
#     print(f"tgt: {tgt.shape}")
#
# if __name__=="__main__":
#     main()
