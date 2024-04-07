
import pytest
from dataclasses import dataclass
from dnaseq2seq import call, vcf, util
from typing import List



def test_merge_genotypes():
    # gconf = [
    #     ([vcf.Variant(pos=10, ref='A', alt='T', qual=1), vcf.Variant(pos=20, ref='G', alt='C', qual=1)],
    #      []),
    #     ([vcf.Variant(pos=10, ref='A', alt='T', qual=1), vcf.Variant(pos=20, ref='G', alt='C', qual=1)],
    #      []),
    #     ([],
    #      [vcf.Variant(pos=10, ref='A', alt='T', qual=1), vcf.Variant(pos=20, ref='G', alt='C', qual=1)]),
    # ]

    # gconf = [
    #     ([vcf.Variant(pos=10, ref='A', alt='T', qual=1)],
    #      []),
    #     ([],
    #      [vcf.Variant(pos=10, ref='A', alt='T', qual=1)]),
    #     ([],
    #      [vcf.Variant(pos=10, ref='A', alt='T', qual=1)]),
    # ]

    # gconf = [
    #     ([vcf.Variant(pos=10, ref='A', alt='T', qual=1), vcf.Variant(pos=20, ref='G', alt='C', qual=1)],
    #      []),
    #
    #     ([vcf.Variant(pos=20, ref='G', alt='C', qual=1)],
    #      [vcf.Variant(pos=10, ref='A', alt='T', qual=1)]),
    #
    #     ([vcf.Variant(pos=20, ref='G', alt='C', qual=1)],
    #      [vcf.Variant(pos=10, ref='A', alt='T', qual=1)]),
    # ]

    # gconf = [
    #     ([],
    #      [vcf.Variant(pos=10, ref='A', alt='T', qual=1), vcf.Variant(pos=20, ref='G', alt='C', qual=1)]),
    #
    #     ([vcf.Variant(pos=10, ref='A', alt='T', qual=1), vcf.Variant(pos=20, ref='G', alt='C', qual=1)],
    #      [vcf.Variant(pos=10, ref='A', alt='T', qual=1)]),
    #
    #     ([],
    #      [vcf.Variant(pos=10, ref='A', alt='T', qual=1)]),
    # ]

def test_resolve_haplotypes():
    gconf = [
        ([vcf.Variant(chrom='X', pos=5, ref='X', alt='Y', qual=1)],
         [vcf.Variant(chrom='X', pos=10, ref='A', alt='T', qual=1), vcf.Variant(chrom='X', pos=20, ref='G', alt='C', qual=1)]),

        ([vcf.Variant(chrom='X', pos=10, ref='A', alt='T', qual=1), vcf.Variant(chrom='X', pos=20, ref='G', alt='C', qual=1)],
         [vcf.Variant(chrom='X', pos=10, ref='A', alt='T', qual=1)]),

        ([vcf.Variant(chrom='X', pos=10, ref='A', alt='T', qual=1)],
         [vcf.Variant(chrom='X', pos=10, ref='A', alt='T', qual=1)]),
    ]


def test_resolve_haplotypes_samepos():
    gconf = [
        ([vcf.Variant(chrom='X', pos=5, ref='X', alt='Y', qual=1)],
         []),

        ([],
         [vcf.Variant(chrom='X', pos=5, ref='A', alt='T', qual=1)]),

        ([vcf.Variant(chrom='X', pos=5, ref='A', alt='T', qual=1)],
         []),
    ]

    hap0, hap1 = call.resolve_haplotypes(gconf)
    assert len(hap0) == 1
    assert len(hap1) == 1
    assert len(hap0[('X', 5, 'X', 'Y')]) == 1
    assert len(hap1[('X', 5, 'A', 'T')]) == 2




def test_split_overlaps():
    v0 = [
        vcf.Variant(chrom='X', pos=5, ref='AGC', alt='T', qual=1)
    ]
    v1 = [
        vcf.Variant(chrom='X', pos=7, ref='A', alt='T', qual=0.9)
    ]
    result = call.split_overlaps(v0, v1)
    assert result


def test_altmatch():
    assert call.any_alt_match(
        vcf.Variant(chrom='X', pos=5, ref='AG', alt='TC', qual=1),
        vcf.Variant(chrom='X', pos=6, ref='G', alt='C', qual=1),
    )
    assert not call.any_alt_match(
        vcf.Variant(chrom='X', pos=6, ref='G', alt='T', qual=1),
        vcf.Variant(chrom='X', pos=6, ref='G', alt='C', qual=1),
    )
    assert call.any_alt_match(
        vcf.Variant(chrom='X', pos=6, ref='G', alt='T', qual=1),
        vcf.Variant(chrom='X', pos=6, ref='C', alt='T', qual=1),
    )
    assert not call.any_alt_match(
        vcf.Variant(chrom='X', pos=6, ref='G', alt='T', qual=1),
        vcf.Variant(chrom='X', pos=5, ref='GG', alt='T', qual=1),
    )
    assert call.any_alt_match(
        vcf.Variant(chrom='X', pos=6, ref='G', alt='T', qual=1),
        vcf.Variant(chrom='X', pos=5, ref='GG', alt='CT', qual=1),
    )
