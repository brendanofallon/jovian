
import pytest
import pysam

from dnaseq2seq import phaser

class MockRefAllA:

    def __init__(self):
        pass

    def fetch(self, chrom, start, end):
        return "A" * (end - start)


@pytest.fixture
def vcf1():
    return pysam.VariantFile("phasertest1.vcf")


def test_gen_haplotypes(vcf1):
    ref = MockRefAllA()
    variants = list(vcf1.fetch())
    hap0, hap1 = phaser.gen_haplotypes(None, ref, "1", 1, 10, variants)
    assert hap0 == "AAATAAAAA"
    assert hap1 == "AAAAAAAAA"
