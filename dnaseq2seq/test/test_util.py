

import pytest
from dnaseq2seq import util
import pysam
import os
from pathlib import Path

@pytest.fixture
def small_vcf():
    return Path(__file__).parent / "test_data/test.vcf"


@pytest.fixture
def small_vcf_sorted():
    return Path(__file__).parent / "test_data/small_vcf_sorted.vcf"


def test_vcf_sort(small_vcf):
    dest = Path("test_vcf_sort.vcf")
    util.sort_chrom_vcf(small_vcf, dest)
    assert dest.exists(), "Destination file does not exist"
    variants = [v for v in pysam.VariantFile(dest)]
    assert len(variants) == 12, "Number of variants in destination does not match expected value"
    prev = -1
    for v in variants:
        assert v.pos >= prev, f"Variant {v} is not sorted"
        prev = v.pos
    os.unlink(dest)


def test_vcf_dedup(small_vcf_sorted):
    dest = Path("test_vcf_dedup.vcf")
    util.dedup_vcf(small_vcf_sorted, dest)
    assert dest.exists(), "Destination file does not exist"
    variants = [v for v in pysam.VariantFile(dest)]
    assert len(variants) == 10, "Number of variants in destination does not match expected value"
    uniqvars = set()
    for v in variants:
        uniqvars.add(util._varkey(v))
    assert len(uniqvars) == len(variants), "Must be at least some duplicate vars in deduped file"

    
