
import pytest
from dataclasses import dataclass
from dnaseq2seq import call, vcf, util
from typing import List

@dataclass
class MockVariantRecord:
    chrom: str
    start: int
    ref: str
    alts: List[str]

    @property
    def alt(self):
        return self.alts[0]

    @property
    def end(self):
        return self.start + len(self.ref)

    @property
    def pos(self):
        return self.start + 1

    @alt.setter
    def alt(self, a):
        self.alts = [a]



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


    # gconf = [
    #     ([vcf.Variant(pos=5, ref='X', alt='Y', qual=1)],
    #      [vcf.Variant(pos=10, ref='A', alt='T', qual=1), vcf.Variant(pos=20, ref='G', alt='C', qual=1)]),
    #
    #     ([vcf.Variant(pos=10, ref='A', alt='T', qual=1), vcf.Variant(pos=20, ref='G', alt='C', qual=1)],
    #      [vcf.Variant(pos=10, ref='A', alt='T', qual=1)]),
    #
    #     ([vcf.Variant(pos=10, ref='A', alt='T', qual=1)],
    #      [vcf.Variant(pos=10, ref='A', alt='T', qual=1)]),
    # ]

    gconf = [
        ([vcf.Variant(chrom='X', pos=5, ref='X', alt='Y', qual=1)],
         []),

        ([],
         [vcf.Variant(chrom='X', pos=5, ref='A', alt='T', qual=1)]),

        ([vcf.Variant(chrom='X', pos=5, ref='A', alt='T', qual=1)],
         []),
    ]

    haps = call.merge_genotypes(gconf)


def test_overlaps():
    assert util.records_overlap(
        MockVariantRecord(chrom='X', start=5, ref='A', alts=['T']),
        MockVariantRecord(chrom='X', start=5, ref='G', alts=['C']),
    )
    assert not util.records_overlap(
        MockVariantRecord(chrom='X', start=5, ref='A', alts=['T']),
        MockVariantRecord(chrom='X', start=6, ref='G', alts=['C']),
    )
    assert util.records_overlap(
        MockVariantRecord(chrom='X', start=5, ref='AC', alts=['T']),
        MockVariantRecord(chrom='X', start=6, ref='G', alts=['C']),
    )
    assert not util.records_overlap(
        MockVariantRecord(chrom='X', start=5, ref='A', alts=['TGTG']),
        MockVariantRecord(chrom='X', start=6, ref='G', alts=['C']),
    )
    assert util.records_overlap(
        MockVariantRecord(chrom='X', start=5, ref='ATGGTA', alts=['']),
        MockVariantRecord(chrom='X', start=8, ref='GG', alts=['C']),
    )

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
