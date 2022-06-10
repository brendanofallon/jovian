
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


def test_gen_combos():
    for n in range(2, 5):
        allresults = list(p for p in call.gen_combos(n))
        assert len(allresults) == len(set(allresults)), "Repeated entries in permutation results"
        assert len(allresults) == 2**n, "Incorrect number of permutations"


def test_gen_idx():
    for c in call.gen_idx(3):
        print(c)

def test_score_conf():
    gconf = [
        ([vcf.Variant(pos=10, ref='A', alt='T', qual=1), vcf.Variant(pos=20, ref='G', alt='C', qual=1)],
         []),
        ([vcf.Variant(pos=10, ref='A', alt='T', qual=1), vcf.Variant(pos=20, ref='G', alt='C', qual=1)],
         []),
        ([],
         [vcf.Variant(pos=10, ref='A', alt='T', qual=1), vcf.Variant(pos=20, ref='G', alt='C', qual=1)]),
    ]

    gconf2 = [
        ([vcf.Variant(pos=10, ref='A', alt='T', qual=1), vcf.Variant(pos=20, ref='G', alt='C', qual=1)],
         []),
        ([vcf.Variant(pos=10, ref='A', alt='T', qual=1), vcf.Variant(pos=20, ref='G', alt='C', qual=1)],
         []),
        ([vcf.Variant(pos=10, ref='A', alt='T', qual=1), vcf.Variant(pos=20, ref='G', alt='C', qual=1)],
         []),
    ]
    score = call.score_conf(gconf)
    score2 = call.score_conf(gconf2)

    assert score2 < score


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



