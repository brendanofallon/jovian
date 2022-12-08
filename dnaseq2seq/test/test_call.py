
import pytest

from dnaseq2seq import call, vcf

@pytest.mark.parametrize(["n_items", "n_chunks"],
                         [(1,1), (5,1), (7,2)])
def test_split_chunks(n_items, n_chunks):
    results = list(call.split_even_chunks(n_items, n_chunks))
    assert results[0][0] == 0
    assert results[-1][1] == n_items
    assert len(results) == n_chunks
    prev_end = 0
    for start, end in results:
        assert start == prev_end
        prev_end = end


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
        ([vcf.Variant(pos=5, ref='X', alt='Y', qual=1)],
         []),

        ([],
         [vcf.Variant(pos=5, ref='A', alt='T', qual=1)]),

        ([vcf.Variant(pos=5, ref='A', alt='T', qual=1)],
         []),
    ]

    haps = call.merge_genotypes(gconf)
    print(f"\n\nHap0: {haps[0]}\nHap1: {haps[1]}")


def test_overlaps():
    assert call.vars_dont_overlap(
        vcf.Variant(pos=5, ref='A', alt='T', qual=1),
        vcf.Variant(pos=6, ref='G', alt='C', qual=1),
    )
    assert not call.vars_dont_overlap(
        vcf.Variant(pos=5, ref='AG', alt='TC', qual=1),
        vcf.Variant(pos=6, ref='G', alt='C', qual=1),
    )
    assert call.vars_dont_overlap(
        vcf.Variant(pos=5, ref='AG', alt='TC', qual=1),
        vcf.Variant(pos=7, ref='G', alt='C', qual=1),
    )

def test_split_overlaps():
    v0 = [
        vcf.Variant(pos=5, ref='AGC', alt='T', qual=1)
    ]
    v1 = [
        vcf.Variant(pos=7, ref='A', alt='T', qual=0.9)
    ]
    result = call.split_overlaps(v0, v1)
    assert result


def test_splitvar():
    v = vcf.Variant(pos=5, ref='AG', alt='TC', qual=1)
    a, b = call.splitvar(v, 6)
    assert a.pos == 5
    assert a.ref == 'A'
    assert a.alt == 'T'
    assert b.pos == 6
    assert b.ref == 'G'
    assert b.alt == 'C'


def test_splitvar2():
    v = vcf.Variant(pos=5, ref='AG', alt='', qual=1)
    a, b = call.splitvar(v, 6)
    assert a.pos == 5
    assert a.ref == 'A'
    assert a.alt == ''
    assert b.pos == 6
    assert b.ref == 'G'
    assert b.alt == ''


def test_splitvar3():
    v = vcf.Variant(pos=5, ref='', alt='ACTG', qual=1)
    a, b = call.splitvar(v, 7)
    assert a.pos == 5
    assert a.ref == ''
    assert a.alt == 'AC'
    assert b.pos == 7
    assert b.ref == ''
    assert b.alt == 'TG'


def test_splitallvars():
    h0 = [
        vcf.Variant(pos=1, ref='G', alt='T', qual=1),
        vcf.Variant(pos=5, ref='AC', alt='TG', qual=1),
    ]
    h1 = [
        vcf.Variant(pos=6, ref='C', alt='G', qual=1),
        vcf.Variant(pos=10, ref='T', alt='C', qual=1),
    ]

    def runasserts(a,b):
        assert len(a) == 3
        assert a[0] == vcf.Variant(pos=1, ref='G', alt='T', qual=1)
        assert a[1] == vcf.Variant(pos=5, ref='A', alt='T', qual=1)
        assert a[2] == vcf.Variant(pos=6, ref='C', alt='G', qual=1)
        assert len(b) == 2
        assert b[0] == vcf.Variant(pos=6, ref='C', alt='G', qual=1)
        assert b[1] == vcf.Variant(pos=10, ref='T', alt='C', qual=1)

    a, b = call.split_overlaps(h0, h1)
    runasserts(a,b)
    a, b = call.split_overlaps(h1, h0)
    runasserts(b,a)


def test_splitallvars2():
    h0 = [
        vcf.Variant(pos=1, ref='G', alt='T', qual=1),
        vcf.Variant(pos=5, ref='AC', alt='TG', qual=1),
    ]
    h1 = [
        vcf.Variant(pos=6, ref='CTA', alt='GGT', qual=1),
        vcf.Variant(pos=10, ref='T', alt='C', qual=1),
    ]

    def doasserts(a, b):
        assert len(a) == 3
        assert a[0] == vcf.Variant(pos=1, ref='G', alt='T', qual=1)
        assert a[1] == vcf.Variant(pos=5, ref='A', alt='T', qual=1)
        assert a[2] == vcf.Variant(pos=6, ref='C', alt='G', qual=1)
        assert len(b) == 3
        assert b[0] == vcf.Variant(pos=6, ref='C', alt='G', qual=1)
        assert b[1] == vcf.Variant(pos=7, ref='TA', alt='GT', qual=1)
        assert b[2] == vcf.Variant(pos=10, ref='T', alt='C', qual=1)

    a, b = call.split_overlaps(h0, h1)
    doasserts(a,b)
    a, b = call.split_overlaps(h1, h0)
    doasserts(b, a)


def test_splitallvars3():
    h0 = [
        vcf.Variant(pos=1, ref='G', alt='T', qual=1),
        vcf.Variant(pos=5, ref='ACTG', alt='TGAC', qual=1),
    ]
    h1 = [
        vcf.Variant(pos=6, ref='CT', alt='GA', qual=1),
        vcf.Variant(pos=11, ref='T', alt='C', qual=1),
    ]

    def doasserts(a, b):
        assert len(a) == 4
        assert a[0] == vcf.Variant(pos=1, ref='G', alt='T', qual=1)
        assert a[1] == vcf.Variant(pos=5, ref='A', alt='T', qual=1)
        assert a[2] == vcf.Variant(pos=6, ref='CT', alt='GA', qual=1)
        assert a[3] == vcf.Variant(pos=8, ref='G', alt='C', qual=1)
        assert len(b) == 2
        assert b[0] == vcf.Variant(pos=6, ref='CT', alt='GA', qual=1)
        assert b[1] == vcf.Variant(pos=11, ref='T', alt='C', qual=1)

    a, b = call.split_overlaps(h0, h1)
    doasserts(a,b)
    a, b = call.split_overlaps(h1, h0)
    doasserts(b, a)
