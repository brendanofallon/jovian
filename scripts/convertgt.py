
import sys
import os
import torch
import lz4
from dnaseq2seq import util

TRUNCATE_LEN=148

def convert_file(tgtpath, dest):
    tgt_seq = util.tensor_from_file(tgtpath, "cpu")
    tgt_kmers = util.tgt_to_kmers(tgt_seq[:, :, 0:TRUNCATE_LEN]).float()
    with lz4.frame.open(dest, "wb") as fh:
        torch.save(tgt_kmers, fh)

def main(pregendir):
    converted = 0
    for i, f in enumerate(os.listdir(pregendir)):
        if f.startswith("tgt_"):
            fid = f.replace(".lz4", "").replace(".pt", "").replace("tgt_", "")
            convert_file(f"{pregendir}/{f}", f"{pregendir}/tgkmers_{fid}.pt.lz4")
            converted += 1
        if i%100 == 0:
            print(f"Scanned {i}, converted {converted}")

if __name__=="__main__":
    main("/Users/brendanofallon/src/dnaseq2seq/test_pregen/")
