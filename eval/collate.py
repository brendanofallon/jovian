#!/usr/bin/env python

import sys
import os
import glob

""" 
Takes the results of many independent hap.py runs and collates them all into a single CSV file with additional columns for the sample name and prefix
"""

def emitsample(prefix, path, include_header=False):
    name = os.path.split(path)[1].replace(".csv", "")
    name = name.replace(prefix, "").replace("_sorted", "").replace(".extended", "")
    if name.endswith("_"):
        name = name[0:-1]
    for i, line in enumerate(open(path)):
        line = line.strip()
        if i==0:
            if include_header:
                print("sample,caller," , line)
            else:
                continue
        else:
            print(name +  "," + prefix + "," + line)


def do_happydir(prefix, hdir, include_header):
    extcsv = glob.glob(f"{hdir}/*extended.csv")
    if not extcsv:
        raise ValueError(f"Could not find extended.csv in {hdir}")
    else:
        emitsample(prefix, extcsv[0], include_header)

if __name__=="__main__":
    prefix = sys.argv[1]
    for i, path in enumerate(sys.argv[2:]):
        do_happydir(prefix, path, i==0)


