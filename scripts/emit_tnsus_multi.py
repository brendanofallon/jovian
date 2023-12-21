#!/usr/bin/env python

import sys
import multiprocessing as mp
from subprocess import run
from pathlib import Path
import random
import os
import re

sys.path.append("/uufs/chpc.utah.edu/common/home/u0379426/src/jovian/dnaseq2seq")

import call


def parse_giab_sample(name):
    name = str(name)
    match = re.search(r'HG00[0-7]_', name)
    if match:
        return match.group().strip("_")
    elif "NIST_002" in name:
        return "HG002"
    elif "NIST_003" in name:
        return "HG003"
    elif "NIST_004" in name:
        return "HG004"
    elif "NA12878" in name:
        return "HG001"
    elif ("NA24385" in name) or ("GM24385" in name):
        return "HG002"
    elif "NA24149" in name:
        return "HG003"
    elif "NA24143" in name:
        return "HG004"
    elif "GM24631" in name:
        return "HG005"
    elif "NA24631" in name:
        return "HG005"
    elif "NA24694" in name:
        return "HG006"
    elif "NA24695" in name:
        return "HG007"


def gensus(args):
    SPLITBEDS, bampath, group = args
    giab = parse_giab_sample(bampath)
    prefix = Path(bampath).name.replace(".cram", "")
    input_bed = SPLITBEDS[giab]
    dest = f"/uufs/chpc.utah.edu/common/home/u0379426/vast/giab_label_beds/{prefix}_labeltnsus_chr{group}.bed"
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

    with mp.Pool(8) as pool:
        pool.map(gensus, [(SPLITBEDS, b, group) for b in bams])

if __name__=="__main__":
    main(sys.argv[1], sys.argv[2:])


