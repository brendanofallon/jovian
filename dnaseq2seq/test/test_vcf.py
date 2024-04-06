import pysam
import pytest
from pathlib import Path
from dnaseq2seq import vcf

TEST_DATA=Path(__file__).parent / "test_data"


class MockReference:

    def fetch(self, start, end, **kwargs):
        return "A" * (end - start)

@pytest.fixture
def tinybam():
    return TEST_DATA / "tiny.bam"

def test_mismatches_to_vars():
    query =  "AAAAAAAA"
    target = "TAACGAAC"
    mm = list(vcf._mismatches_to_vars(query, target, cig_offset=17, probs=[1] * len(query), chrom='X', window_offset=0))
    assert len(mm) == 3
    assert mm[0].ref == "A"
    assert mm[0].alt == "T"
    assert mm[0].pos == 17
    assert mm[1].ref == 'AA'
    assert mm[1].alt == 'CG'
    assert mm[1].pos == 20
    assert mm[2].ref == 'A'
    assert mm[2].alt == 'C'
    assert mm[2].pos == 24


def test_aln_to_vars_ignore_delstart():
    q = "ACTGACTGACTG"
    t =   "TGACTGACTG"
    v = list(vcf.aln_to_vars(q, t, 'X'))
    assert len(v) == 0


def test_aln_to_varsinternal_del():
    q = "ACTGACTGACTG"
    t = "ACTGA--GACTG".replace("-", "")
    v = list(vcf.aln_to_vars(q, t, 'X'))
    assert len(v) == 1
    v = v[0]
    assert v.ref == 'CT'
    assert v.alt == ''
    assert v.pos == 5


def test_aln_to_varsinternal_delins():
    q = "CCCCACTGACTGACTGAAAA"
    t = "CCCCACTGAGGGGACTGAAAA".replace("-", "")
    v = list(vcf.aln_to_vars(q, t, 'X'))
    assert len(v) == 2
    assert v[0].ref == ''
    assert v[0].alt == 'G'
    assert v[0].pos == 9
    assert v[1].ref == 'CT'
    assert v[1].alt == 'GG'
    assert v[1].pos == 9


def test_aln_to_vars_offset_del():
    q = "CCCCCACTGACTGACTG"
    t = "CCCCC---GACTG-CTG".replace("-", "")
    v = list(vcf.aln_to_vars(q, t, 'X'))
    assert len(v) == 1
    assert v[0].ref == 'A'
    assert v[0].alt == ''
    assert v[0].pos == 8


def test_multi_snv_ins():
    refseq = "ACTGA--AC----TGACTGACACTG".replace("-", "")
    altseq = "ACTGATTACAGTTTTACTCACACTG"
    v = list(vcf.aln_to_vars(refseq, altseq, 'X', offset=10))
    assert len(v) == 4
    assert v[0].ref == ''
    assert v[0].alt == 'TT'
    assert v[0].pos == 15
    assert v[0].window_offset == 5

    assert v[1].ref == 'T'
    assert v[1].alt == 'G'
    assert v[1].pos == 17
    assert v[1].window_offset == 7

    assert v[2].ref == ''
    assert v[2].alt == 'TT'
    assert v[2].pos == 19
    assert v[2].window_offset == 9

    assert v[3].ref == 'G'
    assert v[3].alt == 'C'
    assert v[3].pos == 22
    assert v[3].window_offset == 12


def test_agg_variants(tinybam):
    vars0 = {
        ('1', 167085919, 'G', 'A'): [
            vcf.Variant(chrom='1', pos=167085919, ref='G', alt='A', qual=1.0, step=1, window_offset=5, var_count=2),
            vcf.Variant(chrom='1', pos=167085919, ref='G', alt='A', qual=0.5, step=1, window_offset=10, var_count=2),
        ],
        ('1', 167086038, 'G', 'T'): [
            vcf.Variant(chrom='1', pos=167086038, ref='G', alt='T', qual=0.6, step=1, window_offset=5, var_count=2),
        ],
    }
    vars1 = {
        ('1', 167086038, 'G', 'T'): [
            vcf.Variant(chrom='1', pos=167086038, ref='G', alt='T', qual=0.5, step=1, window_offset=7, var_count=1),
        ],
    }
    aln = pysam.AlignmentFile(tinybam)
    vcfvars = list(vcf.construct_vcfvars(vars0, vars1, aln, MockReference()))
    assert len(vcfvars) == 2
    first = vcfvars[0]
    assert first.chrom == '1'
    assert first.pos == 167085920
    assert first.ref == 'G'
    assert first.alt == 'A'
    assert first.quals == [1.0, 0.5]
    assert first.window_offset == [5, 10]
    assert first.het == True
    assert first.genotype == (0, 1)
    assert first.step_count == 2
    assert first.call_count == 2

    second = vcfvars[1]
    assert second.chrom == '1'
    assert second.pos == 167086039
    assert second.ref == 'G'
    assert second.alt == 'T'
    assert second.quals == [0.6, 0.5]
    assert second.step_count == 1
    assert second.call_count == 2
