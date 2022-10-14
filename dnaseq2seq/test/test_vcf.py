
import pytest

from dnaseq2seq import vcf

def test_mismatches_to_vars():
    query =  "AAAAAAAA"
    target = "TAACGAAC"
    mm = list(vcf._mismatches_to_vars(query, target, offset=17, probs=[1] * len(query)))
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
    t = "TGACTGACTG"
    v = list(vcf.aln_to_vars(q, t))
    assert len(v) == 0


def test_aln_to_varsinternal_del():
    q = "ACTGACTGACTG"
    t = "ACTGA--GACTG".replace("-", "")
    v = list(vcf.aln_to_vars(q, t))
    assert len(v) == 1
    v = v[0]
    assert v.ref == 'CT'
    assert v.alt == ''
    assert v.pos == 5


def test_aln_to_varsinternal_delins():
    q = "ACTGACTGACTG"
    t = "ACTGAGGGGACTG".replace("-", "")
    v = list(vcf.aln_to_vars(q, t))
    assert len(v) == 2
    assert v[0].ref == ''
    assert v[0].alt == 'G'
    assert v[0].pos == 5
    assert v[1].ref == 'CT'
    assert v[1].alt == 'GG'
    assert v[1].pos == 5


def test_aln_to_vars_offset_del():
    q = "ACTGACTGACTG"
    t = "---GACTG-CTG".replace("-", "")
    v = list(vcf.aln_to_vars(q, t))
    assert len(v) == 1
    assert v[0].ref == 'A'
    assert v[0].alt == ''
    assert v[0].pos == 8
