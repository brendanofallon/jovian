
from dnaseq2seq import train, util
import torch
import torch.nn as nn

def test_hap_swaps():
    p0 = util.seq_to_onehot_kmers("AAAATTTTCCCCGGGG")
    p1 = util.seq_to_onehot_kmers("AATATTTTCCCTGGGG")

    preds = torch.stack((p0, p1)).unsqueeze(0)

    tgt = torch.stack( (
        util.tgt_to_kmers("AATATTTCCCTGGGG"),
        util.tgt_to_kmers("AAAATTTCCCCGGGG"),
    )).unsqueeze(0)

    lfunc = nn.CrossEntropyLoss()

    assert preds.shape == (1, 2, 4, util.FEATURE_DIM)
    assert tgt.shape == (1, 2, 4)