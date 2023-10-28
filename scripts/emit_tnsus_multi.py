#!/usr/bin/env python

import sys
import multiprocessing as mp
from subprocess import run
from pathlib import Path
import random
import os

sys.path.append("/uufs/chpc.utah.edu/common/home/u0379426/src/jovian/dnaseq2seq")

import call


def gensus(args):
    SPLITBEDS, bampath = args
    giab = Path(bampath).name.split("_")[0]
    prefix = Path(bampath).name.replace(".cram", "")
    input_bed = SPLITBEDS[giab]
    dest = f"/uufs/chpc.utah.edu/common/home/u0379426/vast/giab_label_beds/{prefix}_labeltnsus.bed"
    print(f"BAM: {bampath} giab: {giab} prefix: {prefix} input bed: {input_bed}  dest: {dest}")
    reference_path = os.getenv("REF_GENOME")
    ofh = open(dest, "w")
    for line in open(input_bed).readlines():
        toks = line.strip().split("\t")
        chrom = toks[0]
        window_start = int(toks[1])
        window_end = int(toks[2])
        region_type = toks[3]

        if region_type == "tn" and random.random() < 0.5:
            pos = list(call.cluster_positions(
                call.gen_suspicious_spots(bampath, chrom, window_start, window_end, reference_path), maxdist=100,
            ))
            if pos:
                ofh.write(line.strip() + "-sus\n")
            else:
                ofh.write(line.strip() + "\n")
        else:
            ofh.write(line.strip() + "\n")
    ofh.flush()
    ofh.close()


def main(group, bams):
    SPLITBEDS={
        "HG001": f"/scratch/general/vast/u0379426/giab_label_beds/chrsplit{group}/HG001_labels_chrs{group}.bed",
        "HG002": f"/scratch/general/vast/u0379426/giab_label_beds/chrsplit{group}/HG002_labels_chrs{group}.bed",
        "HG003": f"/scratch/general/vast/u0379426/giab_label_beds/chrsplit{group}/HG003_labels_chrs{group}.bed",
        "HG004": f"/scratch/general/vast/u0379426/giab_label_beds/chrsplit{group}/HG004_labels_chrs{group}.bed",
        "HG005": f"/scratch/general/vast/u0379426/giab_label_beds/chrsplit{group}/HG005_labels_chrs{group}.bed",
        "HG006": f"/scratch/general/vast/u0379426/giab_label_beds/chrsplit{group}/HG006_labels_chrs{group}.bed",
        "HG007": f"/scratch/general/vast/u0379426/giab_label_beds/chrsplit{group}/HG007_labels_chrs{group}.bed",
        }

    with mp.Pool(24) as pool:
        pool.map(gensus, [(SPLITBEDS, b) for b in bams])

if __name__=="__main__":
    main(sys.argv[1], sys.argv[2:])


