import random
import numpy as np
import torch
import logging
import torch.multiprocessing as mp
import scipy.stats as stats

logger = logging.getLogger(__name__)

INDEX_TO_BASE = [
    'A', 'C', 'G', 'T'
]



def base_index(base):
    if base == 'A':
        return 0
    elif base == 'C':
        return 1
    elif base == 'G':
        return 2
    elif base == 'T':
        return 3
    raise ValueError("Expected [ACTG]")


def update_from_base(base, tensor):
    if base == 'A':
        tensor[0] = 1
    elif base == 'C':
        tensor[1] = 1
    elif base == 'G':
        tensor[2] = 1
    elif base == 'T':
        tensor[3] = 1
    elif base == 'N':
        tensor[0:4] = 0.25
    elif base == '-':
        tensor[0:4] = 0.0
    return tensor


def encode_basecall(base, qual, cigop, clipped, refmatch):
    ebc = torch.zeros(8)
    ebc = update_from_base(base, ebc)
    ebc[4] = qual / 100 - 0.5
    ebc[5] = cigop
    ebc[6] = 1 if refmatch else 0
    ebc[7] = 1 if clipped else 0
    return ebc


def random_bases(n):
    return "".join(random.choices("ACTG", k=n))


def pad_zeros(pre, data, post):
    if pre:
        prepad = torch.zeros(pre, data.shape[-1])
        data = torch.cat((prepad, data))
    if post:
        postpad = torch.zeros(post, data.shape[-1])
        data = torch.cat((data, postpad))
    return data


def string_to_tensor(bases, refseq, clip_frac=0):
    if np.random.rand() < clip_frac:
        num_bases_clipped = np.random.randint(3, 50)
        if np.random.rand() < 0.5:
            # Clip left
            clipped_bases = [encode_basecall(b, 50, 0, 1, refmatch=False) for b in random_bases(num_bases_clipped)]
            normal_bases = [encode_basecall(b, 50, 0, 0, refmatch=(b==r)) for b,r  in zip(bases[num_bases_clipped:], refseq[num_bases_clipped:])]
            return torch.vstack(clipped_bases + normal_bases)
        else:
            # Clip right
            normal_bases = [encode_basecall(b, 50, 0, 0, refmatch=(b==r)) for b,r in zip(bases[0:len(bases) - num_bases_clipped], refseq[0:len(bases) - num_bases_clipped])]
            clipped_bases = [encode_basecall(b, 50, 0, 1, refmatch=False) for b in random_bases(num_bases_clipped)]
            return torch.vstack(normal_bases + clipped_bases)
    else:
        # No Clipping
        return torch.vstack([encode_basecall(b, 50, 0, 0, refmatch=(b==r)) for b,r in zip(bases, refseq)])


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


def tensors_from_seq(seq, numreads, readlen, refseq, error_rate=0.0, clip_prob=0):
    """
    Given a sequence of bases, seq, generate "reads" (encoded read tensors) that align to the sequence
    :param seq: Base sequence, reads will match subsequences of this
    :param numreads: Number of read tensors to generate
    :param readlen: Length of each read
    :param clipprob: Probability that this read will contain some clipped bases
    :param error_rate: Per-base error rate
    :return:
    """
    seqs = []
    for i in range(numreads):
        startpos = random.randint(0, len(seq) - readlen)
        seqs.append(
            pad_zeros(startpos,
                      string_to_tensor(mutate_seq(seq[startpos:startpos + readlen], error_rate), refseq[startpos:startpos + readlen], clip_prob),
                      len(seq) - startpos - readlen)
        )

    return torch.stack(seqs)


def stack_refalt_tensrs(refseq, altseq, readlength, totreads, vaf=0.5, error_rate=0, clip_prob=0):
    assert len(refseq) == len(altseq), f"Sequences must be the same length (got {len(refseq)} and {len(altseq)})"
    num_altreads = stats.binom(totreads - 2, vaf).rvs(1)[0] + 1
    fullref = string_to_tensor(refseq, refseq, 0)
    #logger.info(f"Vaf: {vaf:.3f} tot reads: {totreads} num alt reads: {num_altreads}")
    reftensors = tensors_from_seq(refseq, totreads-num_altreads, readlength, refseq, error_rate, clip_prob)
    alttensors = tensors_from_seq(altseq, num_altreads, readlength, refseq, error_rate, clip_prob)
    combined = torch.cat([reftensors, alttensors])
    idx = np.random.permutation(totreads)
    combined[range(totreads)] = combined[idx]
    return torch.cat((fullref.unsqueeze(0), combined)), altseq, vaf


def make_het_snv(seq, readlength, totreads, vaf, error_rate, clip_prob):
    snvpos = random.choice(range(max(0, len(seq) // 2 - 8), min(len(seq), len(seq) // 2 + 8)))
    altseq = list(seq)
    altseq[snvpos] = random.choice('ACTG')
    while altseq[snvpos] == seq[snvpos]:
        altseq[snvpos] = random.choice('ACTG')
    altseq = "".join(altseq)
    result = stack_refalt_tensrs(seq, altseq, readlength, totreads, vaf, error_rate, clip_prob)
    return result




def make_het_del(seq, readlength, totreads, vaf, error_rate, clip_prob):
    del_len = random.choice(range(10))
    delpos = random.choice(range(max(0, len(seq) // 2 - 8), min(len(seq) - del_len, len(seq) // 2 + 8)))
    ls = list(seq)
    for i in range(del_len):
        del ls[delpos]
    altseq = "".join(ls + ["A"] * del_len)
    return stack_refalt_tensrs(seq, altseq, readlength, totreads, vaf, error_rate, clip_prob)


def make_het_ins(seq, readlength, totreads, vaf=0.5, error_rate=0, clip_prob=0):
    ins_len = random.choice(range(10)) + 1
    inspos = random.choice(range(max(0, len(seq) // 2 - 8), min(len(seq) - ins_len, len(seq) // 2 + 8)))
    altseq = "".join(seq[0:inspos]) + "".join(random.choices("ACTG", k=ins_len)) + "".join(seq[inspos:-ins_len])
    altseq = altseq[0:len(seq)]
    return stack_refalt_tensrs(seq, altseq, readlength, totreads, vaf, error_rate=error_rate, clip_prob=clip_prob)


def make_mnv(seq, readlength, totreads, vaf=0.5, error_rate=0, clip_prob=0):
    del_len = random.choice(range(15)) + 1
    delpos = random.choice(range(max(0, len(seq) // 2 - 8), min(len(seq) - del_len, len(seq) // 2 + 8)))
    ls = list(seq)
    for i in range(del_len):
        del ls[delpos]

    ins_len = random.choice(range(15)) + 1
    altseq = "".join(ls[0:delpos]) + "".join(random.choices("ACTG", k=ins_len)) + "".join(ls[delpos:-ins_len])
    altseq = altseq[0:len(seq)]
    altseq = altseq + "A" * (len(seq) - len(altseq))
    return stack_refalt_tensrs(seq, altseq, readlength, totreads, vaf, error_rate=error_rate, clip_prob=clip_prob)


def make_novar(seq, readlength, totreads, vaf=0.5, error_rate=0, clip_prob=0):
    return stack_refalt_tensrs(seq, seq, readlength, totreads, 0, error_rate=error_rate, clip_prob=clip_prob)


def make_batch(batchsize, seqlen, readsperbatch, readlength, factory_func, error_rate, clip_prob, vafs=None):
    src = []
    tgt = []
    tgt_vafs = []
    if vafs is not None:
        assert len(vafs) == batchsize, f"When vafs are provided, there must be exactly one VAF per batch item"
    vafdist = stats.beta(a=1.0, b=5.0) # Only used if vafs is not supplied
    for i in range(batchsize):
        if vafs is not None:
            vaf = vafs[i]
        else:
            if np.random.rand() < 0.10:
                vaf = 1.0
            else:
                vaf = vafdist.rvs(1)[0]
        seq = [b for b in random_bases(seqlen)]
        reads, altseq, tvaf = factory_func(seq, readlength, readsperbatch, vaf=vaf, error_rate=error_rate, clip_prob=clip_prob)
        src.append(reads)
        tgt_vafs.append(tvaf)
        alt_t = target_string_to_tensor(altseq)
        # seq_t = target_string_to_tensor(seq)
        # x = torch.stack((seq_t, alt_t))
        tgt.append(alt_t)
    return torch.stack(src).transpose(1, 2), torch.stack(tgt), torch.tensor(tgt_vafs)


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
    del_w = 8
    ins_w = 8
    mnv_w = 5
    novar_w = 10
    mix = [int(m) for m in np.ceil(np.random.dirichlet((snv_w, del_w, ins_w, mnv_w, novar_w)) * size)] # Ceil because zero sizes break things
    snv_src, snv_tgt, snv_vaf_tgt = make_batch(int(mix[0]), seqlen, readsperbatch, readlength, make_het_snv, error_rate, clip_prob)
    del_src, del_tgt, del_vaf_tgt = make_batch(int(mix[1]), seqlen, readsperbatch, readlength, make_het_del, error_rate, clip_prob)
    ins_src, ins_tgt, ins_vaf_tgt = make_batch(int(mix[2]), seqlen, readsperbatch, readlength, make_het_ins, error_rate, clip_prob)
    mnv_src, mnv_tgt, mnv_vaf_tgt = make_batch(int(mix[3]), seqlen, readsperbatch, readlength, make_mnv, error_rate, clip_prob)
    novar_src, novar_tgt, novar_vaf_tgt = make_batch(int(mix[4]), seqlen, readsperbatch, readlength, make_novar, error_rate, clip_prob)
    return (torch.cat((snv_src, del_src, ins_src, mnv_src, novar_src)),
           torch.cat((snv_tgt, del_tgt, ins_tgt, mnv_tgt, novar_tgt)),
           torch.cat((snv_vaf_tgt, del_vaf_tgt, ins_vaf_tgt, mnv_vaf_tgt, novar_vaf_tgt)))


def target_string_to_tensor(bases):
    """
    The target version doesn't include the qual or cigop features
    """
    result = torch.tensor([base_index(b) for b in bases]).long()
    return result


def make_batch_multi(size, seqlen, readsperbatch, readlength, error_rate, clip_prob, threads):
    """
    Create multiple ReadLoaders in parallel for each element in Inputs
    :param inputs: List of (BAM path, labels csv) tuples
    :param threads: Number of threads to use
    :param max_reads_per_aln: Max number of reads for each pileup
    :return: List of loaders
    """
    results = []
    subsize = size // threads + 1
    with mp.Pool(processes=threads) as pool:
        for i in range(threads):
            result = pool.apply_async(make_mixed_batch, (subsize, seqlen, readsperbatch, readlength, error_rate, clip_prob))
            results.append(result)
        pool.close()

        data = [d.get(timeout=60 * 60) for d in results]
        allsrc = torch.cat([d[0] for d in data])
        alltgt = torch.cat([d[1] for d in data])
        return allsrc, alltgt


# make_batch(12, 30, 10, 18, make_het_snv, 0, clip_prob=0)
