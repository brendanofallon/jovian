#!/bin/bash


set -x 

VER='NISTv4.2.1'
DATE=$(date +%Y-%m-%d)
RES=GIAB_${VER}_${DATE}

mkdir $RES
cd $RES
mkdir bed
mkdir vcf

cd bed
curl -LO https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/NA12878_HG001/${VER}/GRCh37/HG001_GRCh37_1_22_v4.2.1_benchmark.bed
curl -LO https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/AshkenazimTrio/HG002_NA24385_son/${VER}/GRCh37/HG002_GRCh37_1_22_v4.2.1_benchmark_noinconsistent.bed
curl -LO https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/AshkenazimTrio/HG003_NA24149_father/${VER}/GRCh37/HG003_GRCh37_1_22_v4.2.1_benchmark_noinconsistent.bed
curl -LO https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/AshkenazimTrio/HG004_NA24143_mother/${VER}/GRCh37/HG004_GRCh37_1_22_v4.2.1_benchmark_noinconsistent.bed
curl -LO https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/ChineseTrio/HG005_NA24631_son/${VER}/GRCh37/HG005_GRCh37_1_22_v4.2.1_benchmark.bed
curl -LO https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/ChineseTrio/HG006_NA24694_father/${VER}/GRCh37/HG006_GRCh37_1_22_v4.2.1_benchmark.bed
curl -LO https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/ChineseTrio/HG007_NA24695_mother/${VER}/GRCh37/HG007_GRCh37_1_22_v4.2.1_benchmark.bed


cd ../vcf

# Ute
curl -LO https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/NA12878_HG001/latest/GRCh37/HG001_GRCh37_1_22_v4.2.1_benchmark.vcf.gz
curl -LO https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/NA12878_HG001/latest/GRCh37/HG001_GRCh37_1_22_v4.2.1_benchmark.vcf.gz.tbi

# AJ Trio
curl -LO https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/AshkenazimTrio/HG002_NA24385_son/${VER}/GRCh37/HG002_GRCh37_1_22_v4.2.1_benchmark.vcf.gz
curl -LO https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/AshkenazimTrio/HG002_NA24385_son/${VER}/GRCh37/HG002_GRCh37_1_22_v4.2.1_benchmark.vcf.gz.tbi

curl -LO https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/AshkenazimTrio/HG003_NA24149_father/${VER}/GRCh37/HG003_GRCh37_1_22_v4.2.1_benchmark.vcf.gz
curl -LO https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/AshkenazimTrio/HG003_NA24149_father/${VER}/GRCh37/HG003_GRCh37_1_22_v4.2.1_benchmark.vcf.gz.tbi

curl -LO https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/AshkenazimTrio/HG004_NA24143_mother/${VER}/GRCh37/HG004_GRCh37_1_22_v4.2.1_benchmark.vcf.gz
curl -LO https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/AshkenazimTrio/HG004_NA24143_mother/${VER}/GRCh37/HG004_GRCh37_1_22_v4.2.1_benchmark.vcf.gz.tbi

# Chinese Trio
curl -LO https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/ChineseTrio/HG005_NA24631_son/${VER}/GRCh37/HG005_GRCh37_1_22_v4.2.1_benchmark.vcf.gz
curl -LO https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/ChineseTrio/HG005_NA24631_son/${VER}/GRCh37/HG005_GRCh37_1_22_v4.2.1_benchmark.vcf.gz.tbi

curl -LO https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/ChineseTrio/HG006_NA24694_father/${VER}/GRCh37/HG006_GRCh37_1_22_v4.2.1_benchmark.vcf.gz
curl -LO https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/ChineseTrio/HG006_NA24694_father/${VER}/GRCh37/HG006_GRCh37_1_22_v4.2.1_benchmark.vcf.gz.tbi

curl -LO https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/ChineseTrio/HG007_NA24695_mother/${VER}/GRCh37/HG007_GRCh37_1_22_v4.2.1_benchmark.vcf.gz
curl -LO https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/ChineseTrio/HG007_NA24695_mother/${VER}/GRCh37/HG007_GRCh37_1_22_v4.2.1_benchmark.vcf.gz.tbi

