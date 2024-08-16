
import pytest

from dnaseq2seq.bam import get_mapping_coords


def test_mapping_coords():

    # Test 1: read perfectly maps within window and read_start is first position in window
    read_start, window_offset, num_bases = get_mapping_coords(
        window_start=100,
        window_end=250,
        read_length=150,
        read_idx_anchor=0,
        ref_idx_anchor=100)
    assert read_start == 0
    assert window_offset == 0
    assert num_bases == 150

    # Test 2 read maps within window and read_start is upstream of window start
    read_start, window_offset, num_bases = get_mapping_coords(
        window_start=100,
        window_end=250,
        read_length=150,
        read_idx_anchor=10,
        ref_idx_anchor=100)
    assert read_start == 10
    assert window_offset == 0
    assert num_bases == 140

    # Test 3 read maps within window and read_start is downstream of window start
    read_start, window_offset, num_bases = get_mapping_coords(
        window_start=100,
        window_end=250,
        read_length=150,
        read_idx_anchor=0,
        ref_idx_anchor=123)
    assert read_start == 0
    assert window_offset == 23
    assert num_bases == (150 - 23)

    # Test 4 read is tiny and fits entirely within window
    read_start, window_offset, num_bases = get_mapping_coords(
        window_start=100,
        window_end=250,
        read_length=50,
        read_idx_anchor=0,
        ref_idx_anchor=123)
    assert read_start == 0
    assert window_offset == 23
    assert num_bases == 50

    # Test 5 Read is big and fully overlaps window
    read_start, window_offset, num_bases = get_mapping_coords(
        window_start=100,
        window_end=250,
        read_length=500,
        read_idx_anchor=100,
        ref_idx_anchor=50)
    assert read_start == 150
    assert window_offset == 0
    assert num_bases == 150


    # Test 6 Big read,
    read_start, window_offset, num_bases = get_mapping_coords(
        window_start=100,
        window_end=250,
        read_length=500,
        read_idx_anchor=0,
        ref_idx_anchor=50)
    assert read_start == 50
    assert window_offset == 0
    assert num_bases == 150

    # Test 7 Big read, ref anchor inside window but read anchor is big enough so that read start is outside window
    read_start, window_offset, num_bases = get_mapping_coords(
        window_start=100,
        window_end=250,
        read_length=500,
        read_idx_anchor=40,
        ref_idx_anchor=120)
    assert read_start == 20
    assert window_offset == 0
    assert num_bases == 150

    # Test 8 Big read, ref anchor inside window but read anchor is smallish and read start is inside window
    read_start, window_offset, num_bases = get_mapping_coords(
        window_start=100,
        window_end=250,
        read_length=500,
        read_idx_anchor=10,
        ref_idx_anchor=120)
    assert read_start == 0
    assert window_offset == 10
    assert num_bases == 140

    # Test 9, that one annoying case
    read_start, window_offset, num_bases = get_mapping_coords(
        window_start=27000000,
        window_end=27000150,
        read_length=21144,
        read_idx_anchor=12619,
        ref_idx_anchor=27000001)
    assert read_start == 12618
    assert window_offset == 0
    assert num_bases == 150

