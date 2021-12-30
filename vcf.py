
import numpy as np
import ssw_aligner
from dataclasses import dataclass
import pysam

@dataclass
class Variant:
    ref: str
    alt: str
    pos: int
    qual: float

    def __eq__(self, other):
        return self.ref == other.ref and self.alt == other.alt and self.pos == other.pos

    def __hash__(self):
        return hash(f"{self.ref}&{self.alt}&{self.pos}&{self.qual}")

    def __gt__(self, other):
        return self.pos > other.pos

@dataclass
class Cigar:
    op: str
    len: int


@dataclass
class VcfVar:
    """
    holds params for vcf variant records
    """
    chrom: str
    pos: int
    ref: str
    alt: str
    qual: float
    filter: str
    depth: int
    phased: bool
    phase_set: int
    haplotype: int
    window_idx: int
    window_var_count: int
    window_cis_vars: int
    window_trans_vars: int
    genotype: tuple
    het: bool
    duplicate: bool


def _cigtups(cigstr):
    """
    Generator for Cigar objects from a cigar string
    :param cig: cigar string
    :return: Generator of Cigar objects
    """
    digits = []
    for c in cigstr:
        if c.isdigit():
            digits.append(c)
        else:
            cig = Cigar(op=c, len=int("".join(digits)))
            digits = []
            yield cig


def _geomean(probs):
    return np.exp(np.log(probs).mean())

def align_sequences(query, target):
    """
    Return Smith-Watterman alignment of both sequences
    """
    aln = ssw_aligner.local_pairwise_align_ssw(query,
                                               target,
                                               gap_open_penalty=3,
                                               gap_extend_penalty=1,
                                               match_score=2,
                                               mismatch_score=-1)
    return aln

def _mismatches_to_vars(query, target, offset, probs):
    """
    Zip both sequences and look for mismatches, if any are found convert them to Variant objects
    and return them
    This is for finding variants that are inside an "Match" region according to the cigar from an alignment result
    :returns: Generator over Variants from the paired sequences
    """
    mismatches = []
    mismatch_quals = []
    mismatchstart = None
    for i, (a, b) in enumerate(zip(query, target)):
        if a == b:
            if mismatches:
                yield Variant(ref="".join(mismatches[0]).replace("-", ""),
                              alt="".join(mismatches[1]).replace("-", ""),
                              pos=mismatchstart,
                              qual=_geomean(mismatch_quals))  # Geometric mean?
            mismatches = []
            mismatch_quals = []
        else:
            if mismatches:
                mismatches[0] += a
                mismatches[1] += b
                mismatch_quals.append(probs[i])
            else:
                mismatches = [a, b]
                mismatch_quals = [probs[i]]
                mismatchstart = i + offset

    # Could be mismatches at the end
    if mismatches:
        yield Variant(ref="".join(mismatches[0]).replace("-", ""),
                      alt="".join(mismatches[1]).replace("-", ""),
                      pos=mismatchstart,
                      qual=_geomean(mismatch_quals))

def aln_to_vars(refseq, altseq, offset=0, probs=None):
    """
    Smith-Watterman align the given sequences and return a generator over Variant objects
    that describe differences between the sequences
    :param refseq: String of bases representing reference sequence
    :param altseq: String of bases representing alt sequence
    :param offset: This amount will be added to each variant position
    :return: Generator over variants
    """
    if probs is not None:
        assert len(probs) == len(altseq), f"Probabilities must contain same number of elements as alt sequence"
    else:
        probs = np.ones(len(altseq))
    aln = align_sequences(altseq, refseq)
    q_offset = 0
    t_offset = 0

    variant_pos_offset = 0
    if aln.query_begin > 0:
        # yield Variant(ref='', alt=altseq[0:aln.query_begin], pos=offset, qual=-1)
        q_offset += aln.query_begin # Maybe we don't want this?
    if aln.target_begin > 0:
        # yield Variant(ref=refseq[0:aln.target_begin], alt='', pos=offset, qual=-1)
        t_offset += aln.target_begin

    for cig in _cigtups(aln.cigar):
        if cig.op == "M":
            for v in _mismatches_to_vars(
                    refseq[t_offset:t_offset+cig.len],
                    altseq[q_offset:q_offset+cig.len],
                    offset + t_offset,
                    probs[q_offset:q_offset+cig.len]):
                yield v
            q_offset += cig.len
            variant_pos_offset += cig.len
            t_offset += cig.len

        elif cig.op == "I":
            yield Variant(ref='',
                          alt=altseq[q_offset:q_offset+cig.len],
                          pos=offset + variant_pos_offset,
                          qual=_geomean(probs[q_offset:q_offset+cig.len]))
            q_offset += cig.len
            variant_pos_offset += cig.len

        elif cig.op == "D":
            yield Variant(ref=refseq[t_offset:t_offset + cig.len],
                          alt='',
                          pos=offset + t_offset,
                          qual=_geomean(probs[q_offset-1:q_offset+cig.len]))  # Is this right??
            t_offset += cig.len

    # if aln.query_end+1 < len(altseq):
    #     yield Variant(ref='', alt=altseq[aln.query_end+1:], pos=offset + q_offset+1, qual=-1)
    # if aln.target_end_optimal+1 < len(refseq):
    #     yield Variant(ref=refseq[aln.target_end_optimal+1:], alt='', pos=offset + 1, qual=-1)


# import numpy as np
# ref = "ACTGACTG"
# alt = "ACTGCTG"
# probs = np.arange(7) * 0.1
# for v in aln_to_vars(ref, alt, probs=probs):
#     print(v)


def var_depth(var, chrom, aln):
    """
    Get read depth at variant start position from bam  pysam AlignmentFile to get depth at
    :param var: local variant object with just pos ref alt
    :param chrom:
    :param aln: bam pysam AlignmentFile
    :return:
    """
    # get bam depths for now, not same as tensor depth
    counts_acgt = aln.count_coverage(
         contig=chrom,
         start=var.pos,
         stop=var.pos + 1,
         quality_threshold=15,
         read_callback="all",
    )
    return sum(x[0] for x in counts_acgt)


def vcf_vars(vars_hap0, vars_hap1, chrom, window_idx, aln, reference, mindepth=30):
    """
    From hap0 and hap1 lists of vars (pos ref alt qual) create vcf variant record information for entire call window
    :param vars_hap0:
    :param vars_hap1:
    :param chrom:
    :param window_idx: simple index of sequential call windows
    :param aln: bam AlignmentFile
    :param mindepth: read depth cut off in bam to be labled as LowCov in filter field
    :return: single dict of vars for call window
    """
    # get bam depths for now, not same as tensor depth
    depths_hap0 = [var_depth(var, chrom, aln) for var in vars_hap0]
    depths_hap1 = [var_depth(var, chrom, aln) for var in vars_hap1]

    # index vars by (pos, ref, alt) to make dealing with homozygous easier
    vcfvars_hap0 = {}
    for var, depth in zip(vars_hap0, depths_hap0):
        vcfvars_hap0[(var.pos, var.ref, var.alt)] = VcfVar(
            chrom=chrom,
            pos=var.pos + 1,
            ref=var.ref,
            alt=var.alt,
            qual=var.qual,
            filter="PASS",
            depth=depth,
            phased=False,  # set below
            phase_set=min(v.pos for v in (vars_hap0 + vars_hap1)),  # default to first var in window for now
            haplotype=0,  # defalt vars_hap0 to haplotype 0 for now
            window_idx=window_idx,
            window_var_count=len(vars_hap0) + len(vars_hap1),
            window_cis_vars=len(vars_hap1),
            window_trans_vars=len(vars_hap0),
            genotype=(0, 1),  # default to haplotype 0
            het=True,
            duplicate = False,  # initialize but check later
        )
    vcfvars_hap1 = {}
    for var, depth in zip(vars_hap1, depths_hap1):
        vcfvars_hap1[(var.pos, var.ref, var.alt)] = VcfVar(
            chrom=chrom,
            pos=var.pos + 1,
            ref=var.ref,
            alt=var.alt,
            qual=var.qual,
            filter="PASS",
            depth=depth,
            phased=False,  # set below
            phase_set=min(v.pos for v in (vars_hap0 + vars_hap1)),  # default to first var in window for now
            haplotype=1,  # defalt vars_hap1 to haplotype 1 for now
            window_idx=window_idx,
            window_var_count=len(vars_hap0) + len(vars_hap1),
            window_cis_vars=len(vars_hap1),
            window_trans_vars=len(vars_hap0),
            genotype=(1, 0),  # default to haplotype 1 for now
            het=True,
            duplicate=False,
        )
    # check for homozygous vars
    homs = list(set(vcfvars_hap0) & set(vcfvars_hap1))
    for var in homs:
        # modify hap0 var info
        vcfvars_hap0[var].qual = 0.0
        vcfvars_hap0[var].window_cis_vars += vcfvars_hap0[var].window_trans_vars  # cis and trans are now cis
        vcfvars_hap0[var].window_trans_vars -= 1  # one less trans var for hap0
        vcfvars_hap0[var].window_var_count -= 1  # one less overall var for hap0
        vcfvars_hap0[var].genotype = (1, 1)
        vcfvars_hap0[var].het = False
        # then remove from hap1 vars
        vcfvars_hap1.pop(var)

    # combine haplotypes
    assert len(set(vcfvars_hap0) & set(vcfvars_hap1)) == 0, (
        "There should be no vars shared between vcfvars_hap0 and vcfvars_hap1 (hom vars should all be in hap0"
    )
    vcfvars = {**vcfvars_hap0, **vcfvars_hap1}

    for key, var in vcfvars.items():
        # adjust filters
        if var.depth < mindepth:
            var.filter = "LowCov"

        # adjust insertions and deletions so no blank ref or alt
        if var.ref == "" or var.alt == "":
            leading_ref_base = reference.fetch(reference=var.chrom, start=var.pos - 2, end=var.pos - 1)
            var.pos = var.pos - 1
            var.ref = leading_ref_base + var.ref
            var.alt = leading_ref_base + var.alt

    # go back and set phased to true if more than one het variant remains
    het_vcfvars = [k for k, v in vcfvars.items() if v.het]
    if len(het_vcfvars) > 1:
        for var in het_vcfvars:
            vcfvars[var].phased = True

    return vcfvars


def init_vcf(path, sample_name="sample", lowcov=30):
    """
    Initialize pysam VariantFile vcf object and create it's header
    :param path: vcf file path
    :param sample_name: sample name for FORMAT field
    :param lowcov: info for FILTER field "LowCov" header entry
    :return: VariantFile object
    """

    # Create a VCF header
    vcfh = pysam.VariantHeader()
    # Add a sample named "sample"
    vcfh.add_sample("sample")
    # Add contigs
    vcfh.add_meta('contig', items=[('ID', 1)])
    vcfh.add_meta('contig', items=[('ID', 2)])
    vcfh.add_meta('contig', items=[('ID', 3)])
    vcfh.add_meta('contig', items=[('ID', 4)])
    vcfh.add_meta('contig', items=[('ID', 5)])
    vcfh.add_meta('contig', items=[('ID', 6)])
    vcfh.add_meta('contig', items=[('ID', 7)])
    vcfh.add_meta('contig', items=[('ID', 8)])
    vcfh.add_meta('contig', items=[('ID', 9)])
    vcfh.add_meta('contig', items=[('ID', 10)])
    vcfh.add_meta('contig', items=[('ID', 11)])
    vcfh.add_meta('contig', items=[('ID', 12)])
    vcfh.add_meta('contig', items=[('ID', 13)])
    vcfh.add_meta('contig', items=[('ID', 14)])
    vcfh.add_meta('contig', items=[('ID', 15)])
    vcfh.add_meta('contig', items=[('ID', 16)])
    vcfh.add_meta('contig', items=[('ID', 17)])
    vcfh.add_meta('contig', items=[('ID', 18)])
    vcfh.add_meta('contig', items=[('ID', 19)])
    vcfh.add_meta('contig', items=[('ID', 20)])
    vcfh.add_meta('contig', items=[('ID', 21)])
    vcfh.add_meta('contig', items=[('ID', 22)])
    # FILTER values other than "PASS"
    vcfh.add_meta('FILTER', items=[('ID', "LowCov"),
                                   ('Description', f'cov depth at var start pos < {lowcov}')])
    # FORMAT items
    vcfh.add_meta('FORMAT', items=[('ID', "GT"), ('Number', 1), ('Type', 'String'),
                                   ('Description', 'Genotype')])
    vcfh.add_meta('FORMAT', items=[('ID', "DP"), ('Number', 1), ('Type', 'Integer'),
                                   ('Description', 'Bam depth at variant start position')])
    vcfh.add_meta('FORMAT', items=[('ID', "PS"), ('Number', 1), ('Type', 'Integer'),
                                   ('Description', 'Phase set equal to POS of first record in phased set')])
    # INFO items
    vcfh.add_meta('INFO', items=[('ID', "WIN_IDX"), ('Number', "."), ('Type', 'Integer'),
                                 ('Description', 'In which window(s) was the variant called')])
    vcfh.add_meta('INFO', items=[('ID', "WIN_VAR_COUNT"), ('Number', "."), ('Type', 'Integer'),
                                 ('Description', 'Total variants called in same window(s)')])
    vcfh.add_meta('INFO', items=[('ID', "WIN_CIS_COUNT"), ('Number', "."), ('Type', 'Integer'),
                                 ('Description', 'Total cis variants called in same window(s)')])
    vcfh.add_meta('INFO', items=[('ID', "WIN_TRANS_COUNT"), ('Number', "."), ('Type', 'Integer'),
                                 ('Description', 'Total cis variants called in same window(s)')])
    vcfh.add_meta('INFO', items=[('ID', "QUAL"), ('Number', "."), ('Type', 'float'),
                                 ('Description', 'QUAL value(s) for calls in window(s)')])
    vcfh.add_meta('INFO', items=[('ID', "DUPLICATE"), ('Number', 0), ('Type', 'String'),
                                 ('Description', 'Duplicate of call made in previous window')])
    # write to new vcf file object
    return pysam.VariantFile("example.vcf", "w", header=vcfh)


def create_vcf_rec(var, vcf_file):
    """
    create single variant record from pandas row
    :param var:
    :param vcf_file:
    :return:
    """
    # Create record
    r = vcf_file.new_record(contig=var.chrom, start=var.pos -1, stop=var.pos,
                       alleles=(var.ref, var.alt), filter=var["filter"], qual=var.qual)
    # Set FORMAT values
    r.samples['sample']['GT'] = var.genotype
    r.samples['sample'].phased = var.phased  # note: need to set phased after setting genotype
    r.samples['sample']['DP'] = var.depth
    r.samples['sample']['PS'] = var.phase_set
    # Set INFO values
    r.info['WIN_IDX'] = var.window_idx
    r.info['WIN_VAR_COUNT'] = var.window_var_count
    r.info['WIN_CIS_COUNT'] = var.window_cis_vars
    r.info['WIN_TRANS_COUNT'] = var.window_trans_vars
    return r


def vars_to_vcf(vcf_file, pr_vars):
    """
    create variant records from pyranges variant table and write all to pysam VariantFile
    :param vcf_file:
    :param pr_vars:
    :return: None
    """
    for i, var in pr_vars.df.iterrows():
        r = create_vcf_rec(var, vcf_file)
        vcf_file.write(r)
