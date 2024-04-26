
from dnaseq2seq import call, vcf, util


def test_resolve_haplotypes_ambiguous():
    gconf = [
        ([vcf.Variant(chrom='X', pos=10, ref='A', alt='T', qual=1)],
         []),
        ([vcf.Variant(chrom='X', pos=20, ref='G', alt='C', qual=1)],
         []),
    ]
    hap0, hap1 = call.resolve_haplotypes(gconf)
    assert len(hap0) == 1
    assert len(hap1) == 1

def test_resolve_haplotypes_cis():
    gconf = [
        ([vcf.Variant(chrom='X', pos=10, ref='A', alt='T', qual=1)],
         [vcf.Variant(chrom='X', pos=20, ref='G', alt='C', qual=1)]),
        ([],
         [vcf.Variant(chrom='X', pos=10, ref='A', alt='T', qual=1), vcf.Variant(chrom='X', pos=20, ref='G', alt='C', qual=1)]),
        ([vcf.Variant(chrom='X', pos=20, ref='G', alt='C', qual=1), vcf.Variant(chrom='X', pos=10, ref='A', alt='T', qual=1)],
         []),
    ]
    hap0, hap1 = call.resolve_haplotypes(gconf)
    assert len(hap0) == 2
    assert len(hap1) == 0

def test_resolve_haplotypes_trans():
    gconf = [
        ([vcf.Variant(chrom='X', pos=10, ref='A', alt='T', qual=1)],
         [vcf.Variant(chrom='X', pos=20, ref='G', alt='C', qual=1)]),
        ([],
         [vcf.Variant(chrom='X', pos=10, ref='A', alt='T', qual=1), vcf.Variant(chrom='X', pos=20, ref='G', alt='C', qual=1)]),
        ([vcf.Variant(chrom='X', pos=20, ref='G', alt='C', qual=1)],
         [vcf.Variant(chrom='X', pos=10, ref='A', alt='T', qual=1)]),
    ]
    hap0, hap1 = call.resolve_haplotypes(gconf)
    assert len(hap0) == 1
    assert len(hap1) == 1

def test_resolve_haplotypes_conflicts():
    gconf = [
        ([vcf.Variant(chrom='X', pos=5, ref='X', alt='Y', qual=1)],
         [vcf.Variant(chrom='X', pos=10, ref='A', alt='T', qual=1), vcf.Variant(chrom='X', pos=20, ref='G', alt='C', qual=1)]),

        ([vcf.Variant(chrom='X', pos=10, ref='A', alt='T', qual=1), vcf.Variant(chrom='X', pos=20, ref='G', alt='C', qual=1)],
         [vcf.Variant(chrom='X', pos=10, ref='A', alt='T', qual=1)]),

        ([vcf.Variant(chrom='X', pos=10, ref='A', alt='T', qual=1)],
         [vcf.Variant(chrom='X', pos=10, ref='A', alt='T', qual=1)]),
    ]
    hap0, hap1 = call.resolve_haplotypes(gconf)
    assert len(hap0) == 2
    assert len(hap1) == 2

    assert len(hap0[('X', 10, 'A', 'T')])
    assert len(hap0[('X', 5, 'X', 'Y')])

    assert len(hap1[('X', 20, 'G', 'C')])
    assert len(hap1[('X', 10, 'A', 'T')])

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


def test_resolve_haplotypes_transitive():
    gconf = [
        ([vcf.Variant(chrom='X', pos=5, ref='X', alt='Y', qual=1), vcf.Variant(chrom='X', pos=10, ref='G', alt='C', qual=1)],
         []),

        ([],
         [vcf.Variant(chrom='X', pos=10, ref='G', alt='C', qual=1)]),

        ([vcf.Variant(chrom='X', pos=10, ref='G', alt='C', qual=1), vcf.Variant(chrom='X', pos=20, ref='T', alt='G', qual=1)],
         []),
    ]

    hap0, hap1 = call.resolve_haplotypes(gconf)
    assert len(hap0) == 3
    assert len(hap1) == 0
