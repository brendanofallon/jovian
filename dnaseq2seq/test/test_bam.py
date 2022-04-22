
import pytest

import pysam
from dnaseq2seq import bam, util
import torch
from pathlib import Path

TEST_DATA = Path(__file__).parent / "test_data"

@pytest.fixture
def smallbam():
    return pysam.AlignmentFile(TEST_DATA / "smallregion.bam")

class MockRead:

    def __init__(self, seq, quals, cigartups, alignment_start, alignment_end, reverse=False):
        self.query_sequence = seq
        self.query_qualities = quals
        self.cigartuples = cigartups
        self.alignment_start = alignment_start
        self.alignment_end = alignment_end
        self.is_reverse = reverse


def test_iterate_bases():
    rec = MockRead('ACTGACTG', quals=[60] * 8, cigartups=[(4, 3), (2, 3), (0, 2)], alignment_start=10, alignment_end=18)
    enc = list(bam.iterate_bases(rec, reverse=False))

    t = torch.stack(enc, dim=0)
    p = util.readstr(t)
    assert p == 'actGACTG'



def test_iterate_bases_reverse():
    rec = MockRead('ACTGACTG', quals=[60] * 8, cigartups=[(4, 3), (2, 3), (0, 2)], alignment_start=10, alignment_end=18)
    enc = list(bam.iterate_bases(rec, reverse=True))

    t = torch.stack(enc, dim=0)
    p = util.readstr(t)
    assert p == 'CAGTCagt'
    assert len(enc) == 8


def test_alnstart_end(smallbam):
    for rec in smallbam.fetch():
        if rec.query_name == "A00576:10:HCC35DSXX:3:1215:17833:9048:CCATC+TGATA" and rec.is_read1:
            # right soft clipping
            assert bam.alnstart(rec) == 37196037
            assert bam.alnend(rec) == 37196121
        if rec.query_name == "A00576:10:HCC35DSXX:3:2266:3784:16219:AGAGA+GGATG" and rec.is_read2:
            # no clipping
            assert bam.alnstart(rec) == 37196042
            assert bam.alnend(rec) == 37196187
        if rec.query_name == "A00576:10:HCC35DSXX:3:1510:3721:11350:GGATG+CCACC" and rec.is_read2:
            # both right and left soft clipping
            assert bam.alnstart(rec) == 37196015
            assert bam.alnend(rec) == 37196068