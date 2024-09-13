
import sys
import pytest
import dnaseq2seq.util as util
from dnaseq2seq.util import merge_overlapping_regions, VariantSortedBuffer, records_overlap, expand_to_bases, bases_to_kvec
from pathlib import Path
from io import StringIO
from typing import List
from dataclasses import dataclass
import random
from tempfile import NamedTemporaryFile


DATADIR = Path(__file__).parent / "test_data"

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

    def __str__(self):
        return f"{self.chrom} {self.pos} {self.ref} -> {self.alt}"


def test_overlaps():
    assert records_overlap(
        MockVariantRecord(chrom='X', start=5, ref='A', alts=['T']),
        MockVariantRecord(chrom='X', start=5, ref='G', alts=['C']),
    )
    assert not records_overlap(
        MockVariantRecord(chrom='X', start=5, ref='A', alts=['T']),
        MockVariantRecord(chrom='X', start=6, ref='G', alts=['C']),
    )
    assert records_overlap(
        MockVariantRecord(chrom='X', start=5, ref='AC', alts=['T']),
        MockVariantRecord(chrom='X', start=6, ref='G', alts=['C']),
    )
    assert not records_overlap(
        MockVariantRecord(chrom='X', start=5, ref='A', alts=['TGTG']),
        MockVariantRecord(chrom='X', start=6, ref='G', alts=['C']),
    )
    assert records_overlap(
        MockVariantRecord(chrom='X', start=5, ref='ATGGTA', alts=['']),
        MockVariantRecord(chrom='X', start=8, ref='GG', alts=['C']),
    )


def test_empty_input():
    assert merge_overlapping_regions([]) == []

def test_no_overlapping_regions():
    regions = [('chr1', 1, 100, 200), ('chr1', 2, 300, 400)]
    assert merge_overlapping_regions(regions) == regions

def test_adjacent_regions():
    regions = [('chr1', 1, 100, 200), ('chr1', 2, 200, 300)]
    assert merge_overlapping_regions(regions) == [('chr1', 1, 100, 300)]

def test_overlapping_regions():
    regions = [('chr1', 1, 100, 250), ('chr1', 2, 200, 300)]
    expected = [('chr1', 1, 100, 300)]
    assert merge_overlapping_regions(regions) == expected

def test_multiple_overlapping_regions():
    regions = [('chr1', 1, 100, 200), ('chr1', 2, 150, 250), ('chr1', 3, 240, 340)]
    expected = [('chr1', 1, 100, 340)]
    assert merge_overlapping_regions(regions) == expected

def test_non_consecutive_overlapping_regions():
    regions = [('chr1', 1, 100, 200), ('chr1', 3, 250, 350), ('chr1', 2, 150, 240)]
    expected = [('chr1', 1, 100, 240), ('chr1', 3, 250, 350)]
    assert merge_overlapping_regions(sorted(regions, key=lambda x: x[1])) == expected

def test_mixed_chromosomes():
    regions = [('chr1', 1, 100, 200), ('chr2', 2, 150, 250)]
    assert merge_overlapping_regions(regions) == regions

# Optional: Test for the sorting behavior if it's critical for the merging logic
def test_sorting_behavior():
    regions = [('chr1', 2, 150, 250), ('chr1', 1, 100, 200)]
    expected = [('chr1', 1, 100, 250)]
    assert merge_overlapping_regions(regions) == expected


def test_variantbuffer():
    vars_a = [MockVariantRecord(chrom='chr1', start=i, ref='A', alts=['G']) for i in range(10)]
    vars_b = [MockVariantRecord(chrom='chr1', start=i, ref='A', alts=['G']) for i in range(10, 20)]
    vars_c = [MockVariantRecord(chrom='chr1', start=i, ref='A', alts=['G']) for i in range(20, 30)]
    random.shuffle(vars_a)
    random.shuffle(vars_b)
    random.shuffle(vars_c)
    variants = vars_a + vars_b + vars_c

    out = StringIO()
    vb = VariantSortedBuffer(outputfh=out, buff_size=10, capacity_factor=2)
    for i in range(20):
        vb.put(variants[i])

    outlines = list(filter(lambda x: len(x.strip()) > 0, out.getvalue().split("chr")))
    assert len(vb) == 10
    assert len(outlines) == 10
    pos_list = list(int(line.split()[1]) for line in outlines)
    assert pos_list == sorted(pos_list)

    for i in range(20, 30):
        vb.put(variants[i])

    outlines = list(filter(lambda x: len(x.strip()) > 0, out.getvalue().split("chr")))
    assert len(vb) == 10
    assert len(outlines) == 20
    pos_list = list(int(line.split()[1]) for line in outlines)
    assert pos_list == sorted(pos_list)

    vb.flush()
    outlines = list(filter(lambda x: len(x.strip()) > 0, out.getvalue().split("chr")))
    assert len(outlines) == 30
    pos_list = list(int(line.split()[1]) for line in outlines)
    assert pos_list == sorted(pos_list)


def test_expand_vals():
    vals = ["A", "B", "C", "D"]
    result = expand_to_bases(vals, expansion_factor=3)
    assert result == ["A", "A", "A", "B", "B", "B", "C", "C", "C", "D", "D", "D"]


def test_convert_kmers():
    seq0 = "ACTAGACAATTCGATGCGATAGATCGGCTTGGAAAACCCCTTTTGAGA"
    kvec0 = bases_to_kvec(seq0, util.s2i, kmersize=4)
    s0 = util.kmer_idx_to_str(kvec0, util.i2s)
    assert s0 == seq0


def test_kmer_onehot():
    seq0 = "ACTAGACAATTCGATGCGATAGATCGGCTTGGAAAACCCCTTTTGAGA"
    km = util.seq_to_onehot_kmers(seq0)
    s0 = util.kmer_preds_to_seq(km, util.i2s)
    assert seq0 == s0


def test_unique_chroms():
    bed = """#this is a header
chr1\t10\t20
chr1\t20\t30
chr5\t3\t5
chr5\t10\t25
chrX\t100\t250
"""
    testfile = NamedTemporaryFile(delete=False)
    with open(testfile.name, "w") as fh:
        fh.write(bed)
        fh.flush()
    chroms = util.unique_chroms(Path(testfile.name))
    assert chroms == ["chr1", "chr5", "chrX"]

def test_format_bp():
    assert util.format_bp(123) == "123"
    assert util.format_bp(123000) == "123.0 Kb"
    assert util.format_bp(123500456) == "123.5 MB"
    assert util.format_bp(123500456000) == "123.5 GB"


def test_empty_regions():
    regions = []
    max_region_size = 100
    result = list(util.split_large_regions(regions, max_region_size))
    assert result == []


def test_single_region_below_max_size():
    regions = [("chr1", 100, 200)]
    max_region_size = 150
    result = list(util.split_large_regions(regions, max_region_size))
    assert result == [("chr1", 100, 200)]


def test_single_region_above_max_size():
    regions = [("chr1", 100, 300)]
    max_region_size = 150
    result = list(util.split_large_regions(regions, max_region_size))
    assert result == [("chr1", 100, 250), ("chr1", 250, 300)]


def test_multiple_regions():
    regions = [("chr1", 100, 200), ("chr1", 300, 400), ("chr2", 500, 600)]
    max_region_size = 150
    result = list(util.split_large_regions(regions, max_region_size))
    assert result == [("chr1", 100, 200), ("chr1", 300, 400), ("chr2", 500, 600)]


def test_cluster_positions_basic():
    poslist = [10, 50, 55, 160]
    maxdist = 50
    expected = [(2, 63), (152, 168)]
    result = list(util.cluster_positions(poslist, maxdist))
    assert result == expected, f"Expected {expected}, got {result}"


def test_cluster_positions_single_position():
    poslist = [20]
    expected = [(12, 28)]
    result = list(util.cluster_positions(poslist))
    assert result == expected, f"Expected {expected}, got {result}"


def test_cluster_positions_with_padding():
    poslist = [100, 110, 115, 220, 225]
    pad_bases = 10
    maxdist = 15
    expected = [(90, 125), (210, 235)]
    result = list(util.cluster_positions(poslist, maxdist, pad_bases))
    assert result == expected, f"Expected {expected}, got {result}"


def test_cluster_positions_no_overlap():
    poslist = [10, 120, 230, 340]
    maxdist = 50
    expected = [(2, 18), (112, 128), (222, 238), (332, 348)]
    result = list(util.cluster_positions(poslist, maxdist))
    assert result == expected, f"Expected {expected}, got {result}"

def test_bed_prog_counter():
    testbed = DATADIR / "test.bed"
    counter = util.RegionProgressCounter(testbed)
    assert counter.prog('1', 500) == 0
    assert counter.prog('1', 750) == 0.0625
    assert counter.prog('1', 1000) == 0.125
    assert counter.prog('1', 1500) == 0.25
    assert counter.prog('1', 15000) == 0.5
    assert counter.prog('2', 100) == 0.5
    assert counter.prog('2', 1100) == 0.75
    assert counter.prog('2', 1600) == 0.75 + 0.125
