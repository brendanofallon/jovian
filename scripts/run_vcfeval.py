#!/usr/bin/env python


VCFEVAL="/uufs/chpc.utah.edu/common/home/u0379426/rtg-tools-3.12.1/rtg vcfeval"
REF="/uufs/chpc.utah.edu/common/home/arup-storage4/brendan/ref/human_g1k_v37_decoy_phiXAdaptr.sdf"

import sys
import subprocess
from pathlib import Path
import random
import string
import os
import shutil

GIABROOT="/uufs/chpc.utah.edu/common/home/u0379426/GIAB_NISTv4.2.1_2023-10-26"

GIAB_DATA={
        "na12878_vcf": f"{GIABROOT}/vcf/HG001_GRCh37_1_22_v4.2.1_benchmark.vcf.gz",
        "na24631_vcf": f"{GIABROOT}/vcf/HG005_GRCh37_1_22_v4.2.1_benchmark.vcf.gz",
        "na24385_vcf": f"{GIABROOT}/vcf/HG002_GRCh37_1_22_v4.2.1_benchmark.vcf.gz",
        "hg003_vcf": f"{GIABROOT}/vcf/HG003_GRCh37_1_22_v4.2.1_benchmark.vcf.gz",
        "hg004_vcf": f"{GIABROOT}/vcf/HG004_GRCh37_1_22_v4.2.1_benchmark.vcf.gz",
        "hg005_vcf": f"{GIABROOT}/vcf/HG005_GRCh37_1_22_v4.2.1_benchmark.vcf.gz",
        "hg006_vcf": f"{GIABROOT}/vcf/HG006_GRCh37_1_22_v4.2.1_benchmark.vcf.gz",
        "hg007_vcf": f"{GIABROOT}/vcf/HG007_GRCh37_1_22_v4.2.1_benchmark.vcf.gz",
        "na12878_bed": f"{GIABROOT}/bed/HG001_GRCh37_1_22_v4.2.1_benchmark.bed",
        "na24385_bed": f"{GIABROOT}/bed/HG002_GRCh37_1_22_v4.2.1_benchmark_noinconsistent.bed",
        "hg003_bed": f"{GIABROOT}/bed/HG003_GRCh37_1_22_v4.2.1_benchmark_noinconsistent.bed",
        "hg004_bed": f"{GIABROOT}/bed/HG004_GRCh37_1_22_v4.2.1_benchmark_noinconsistent.bed",
        "na24631_bed": f"{GIABROOT}/bed/HG005_GRCh37_1_22_v4.2.1_benchmark.bed",
        "hg006_bed": f"{GIABROOT}/bed/HG006_GRCh37_1_22_v4.2.1_benchmark.bed",
        "hg005_bed": f"{GIABROOT}/bed/HG005_GRCh37_1_22_v4.2.1_benchmark.bed",
        "hg007_bed": f"{GIABROOT}/bed/HG007_GRCh37_1_22_v4.2.1_benchmark.bed",
        }


def randchars(n=6):
    return "".join(random.choices(string.digits + string.ascii_letters, k=n))



def find_giab(name):
    if ("NA12878" in name) or ("878_" in name):
        return GIAB_DATA['na12878_bed'], GIAB_DATA['na12878_vcf']
    elif "GM24631" in name:
        return GIAB_DATA['na24631_bed'], GIAB_DATA['na24631_vcf']
    elif ("24385" in name) or ("HG002" in name) or ("385_" in name):
        return GIAB_DATA['na24385_bed'], GIAB_DATA['na24385_vcf']
    elif "HG003" in name or ("NIST_003" in name) :
        return GIAB_DATA['hg003_bed'], GIAB_DATA['hg003_vcf']
    elif "HG004" in name or ("NIST_004" in name):
        return GIAB_DATA['hg004_bed'], GIAB_DATA['hg004_vcf']
    elif "HG005" in name:
        return GIAB_DATA['hg005_bed'], GIAB_DATA['hg005_vcf']
    elif "HG006" in name or "NA24694" in name:
        return GIAB_DATA['hg006_bed'], GIAB_DATA['hg006_vcf']
    elif "HG007" in name or "NA24695":
        return GIAB_DATA['hg007_bed'], GIAB_DATA['hg007_vcf']
    raise ValueError(f"Couldn't find a match for input file {name}")


def run_vcfeval(bed, baseline_vcf, calls_vcf, output_mode='split'):
    dest = Path(calls_vcf).name.replace(".vcf", "").replace(".gz", "") + "_vcfevalresults"
    cmd = f"{VCFEVAL} -t {REF} -b {baseline_vcf} -c {calls_vcf} --vcf-score-field QUAL --ref-overlap --all-records --bed-regions {bed} --evaluation-regions {bed} -o {dest} --output-mode {output_mode}"
    sys.stderr.write(f"Executing: {cmd} \n")
    subprocess.run(cmd, shell=True)
    return dest

def tabix_if_needed(f):
    if not str(f).endswith(".gz"):
        dest = str(f) + ".gz"
        cmd = f"bgzip -c {f} > {dest}"
        sys.stderr.write(f"BGzipping: {cmd} \n")
        subprocess.run(cmd, shell=True)
        cmd = f"tabix {dest}"
        subprocess.run(cmd, shell=True)
        return dest
    else:
        return f

def intersect_beds(a, b):
    dest = Path(a).name.replace(".bed", "") + f"_intersection_{randchars()}.bed"
    cmd = f"bedtools intersect -a {a} -b {b} > {dest}"
    sys.stderr.write(f"BED intersection: {cmd} \n")
    subprocess.run(cmd, shell=True)
    return dest

def main(bed, vcf):

    vcf = Path(tabix_if_needed(vcf)).absolute()

    fname = Path(vcf).name
    confbed, baselinevcf = find_giab(fname)
    
    confbedint = Path(intersect_beds(bed, confbed)).absolute()
    
    destdir = run_vcfeval(confbedint, baselinevcf, vcf)
    #os.chdir(destdir)
    #destdir = run_vcfeval(confbedint, baselinevcf, vcf, output_mode='combine')
    #os.rename(f"{destdir}/output.vcf.gz", "output.vcf.gz")
    #os.rename(f"{destdir}/output.vcf.gz.tbi", "output.vcf.gz.tbi")
    #shutil.rmtree(destdir)
    
    os.unlink(confbedint)


    



if __name__=="__main__":
    main(sys.argv[1], sys.argv[2])

