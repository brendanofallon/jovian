
import pytest

from dnaseq2seq import call

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



