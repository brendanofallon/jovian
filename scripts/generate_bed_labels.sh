#!/bin/bash

set -x

REF_GENOME=/uufs/chpc.utah.edu/common/home/u0379426/vast/ref/human_g1k_v37_decoy_phiXAdaptr.fasta.gz
DEST_DIR=$HOME/vast

 #./splitbed.py ~/GIAB_NISTv4.2.1_2023-10-26/bed/HG001_GRCh37_1_22_v4.2.1_benchmark.bed > ~/${DEST_DIR}/HG001_split.bed
 #./splitbed.py ~/GIAB_NISTv4.2.1_2023-10-26/bed/HG002_GRCh37_1_22_v4.2.1_benchmark_noinconsistent.bed > ~/${DEST_DIR}/giab_label_beds/HG002_split.bed
 #./splitbed.py ~/GIAB_NISTv4.2.1_2023-10-26/bed/HG003_GRCh37_1_22_v4.2.1_benchmark_noinconsistent.bed > ~/${DEST_DIR}/giab_label_beds/HG003_split.bed
 #./splitbed.py ~/GIAB_NISTv4.2.1_2023-10-26/bed/HG004_GRCh37_1_22_v4.2.1_benchmark_noinconsistent.bed > ~/${DEST_DIR}/giab_label_beds/HG004_split.bed
 #./splitbed.py ~/GIAB_NISTv4.2.1_2023-10-26/bed/HG005_GRCh37_1_22_v4.2.1_benchmark.bed > ~/${DEST_DIR}/giab_label_beds/HG005_split.bed
 #./splitbed.py ~/GIAB_NISTv4.2.1_2023-10-26/bed/HG006_GRCh37_1_22_v4.2.1_benchmark.bed > ~/${DEST_DIR}/giab_label_beds/HG006_split.bed
 #./splitbed.py ~/GIAB_NISTv4.2.1_2023-10-26/bed/HG007_GRCh37_1_22_v4.2.1_benchmark.bed > ~/${DEST_DIR}/giab_label_beds/HG007_split.bed


 for VCF in /uufs/chpc.utah.edu/common/home/u0379426/GIAB_NISTv4.2.1_2023-10-26/vcf/HG001_GRCh37_1_22_v4.2.1_benchmark.vcf.gz \
    /uufs/chpc.utah.edu/common/home/u0379426/GIAB_NISTv4.2.1_2023-10-26/vcf/HG002_GRCh37_1_22_v4.2.1_benchmark.vcf.gz \
    /uufs/chpc.utah.edu/common/home/u0379426/GIAB_NISTv4.2.1_2023-10-26/vcf/HG003_GRCh37_1_22_v4.2.1_benchmark.vcf.gz \
    /uufs/chpc.utah.edu/common/home/u0379426/GIAB_NISTv4.2.1_2023-10-26/vcf/HG004_GRCh37_1_22_v4.2.1_benchmark.vcf.gz \
    /uufs/chpc.utah.edu/common/home/u0379426/GIAB_NISTv4.2.1_2023-10-26/vcf/HG005_GRCh37_1_22_v4.2.1_benchmark.vcf.gz \
    /uufs/chpc.utah.edu/common/home/u0379426/GIAB_NISTv4.2.1_2023-10-26/vcf/HG006_GRCh37_1_22_v4.2.1_benchmark.vcf.gz \
    /uufs/chpc.utah.edu/common/home/u0379426/GIAB_NISTv4.2.1_2023-10-26/vcf/HG007_GRCh37_1_22_v4.2.1_benchmark.vcf.gz; do
        PREFIX=$(basename $VCF | cut -d "_" -f1)
        echo $PREFIX;
        ./assignclasses.py $VCF ~/GIAB_stratifications/GRCh37_AllTandemRepeatsandHomopolymers_slop5.bed.gz ~/GIAB_stratifications/GRCh37_alllowmapandsegdupregions.bed.gz ${DEST_DIR}/giab_label_beds/${PREFIX}_split.bed > ${DEST_DIR}/giab_label_beds/${PREFIX}_split_labels.bed
    done




for CRAM in /uufs/chpc.utah.edu/common/home/u0379426/vast/bam_wgs/HG002_221406_S1/HG002_221406_S1.cram \
    /uufs/chpc.utah.edu/common/home/u0379426/vast/bam_wgs/HG002_221506/HG002_221506.cram; do
    NAME=$(basename $CRAM | sed -e 's/.cram//g')
    ./emit_sus_regions.py $REF_GENOME $CRAM ~/vast/giab_label_beds/HG002_split_labels.bed > ${NAME}_labels_tnsus.bed
done


for CRAM in /uufs/chpc.utah.edu/common/home/u0379426/vast/bam_wgs/HG003_220807_S5/HG003_220807_S5.cram \
    /uufs/chpc.utah.edu/common/home/u0379426/vast/bam_wgs/HG003_221407_S2/HG003_221407_S2.cram \
    /uufs/chpc.utah.edu/common/home/u0379426/vast/bam_wgs/HG003_221407_S3/HG003_221407_S3.cram \
    NAME=$(basename $CRAM | sed -e 's/.cram//g')
    ./emit_sus_regions.py $REF_GENOME $CRAM ~/vast/giab_label_beds/HG003_split_labels.bed > ${NAME}_labels_tnsus.bed
done


for CRAM in /uufs/chpc.utah.edu/common/home/u0379426/vast/bam_wgs/HG004_220807_S1/HG004_220807_S1.cram \
    /uufs/chpc.utah.edu/common/home/u0379426/vast/bam_wgs/HG004_221406_S3/HG004_221406_S3.cram \
    /uufs/chpc.utah.edu/common/home/u0379426/vast/bam_wgs/HG004_223006_S1/HG004_223006_S1.cram \
   NAME=$(basename $CRAM | sed -e 's/.cram//g')
    ./emit_sus_regions.py $REF_GENOME $CRAM ~/vast/giab_label_beds/HG004_split_labels.bed > ${NAME}_labels_tnsus.bed
done


for CRAM in /uufs/chpc.utah.edu/common/home/u0379426/vast/bam_wgs/HG005_02_S1/HG005_02_S1.cram \
    /uufs/chpc.utah.edu/common/home/u0379426/vast/bam_wgs/HG005_220807_S7/HG005_220807_S7.cram \
    /uufs/chpc.utah.edu/common/home/u0379426/vast/bam_wgs/HG005_222206_S1/HG005_222206_S1.cram \
    /uufs/chpc.utah.edu/common/home/u0379426/vast/bam_wgs/HG005_222206_S2/HG005_222206_S2.cram \
    /uufs/chpc.utah.edu/common/home/u0379426/vast/bam_wgs/HG005_222306_S1/HG005_222306_S1.cram \
    /uufs/chpc.utah.edu/common/home/u0379426/vast/bam_wgs/HG005_S2/HG005_S2.cram \   
   NAME=$(basename $CRAM | sed -e 's/.cram//g')
    ./emit_sus_regions.py $REF_GENOME $CRAM ~/vast/giab_label_beds/HG005_split_labels.bed > ${NAME}_labels_tnsus.bed
done

    /uufs/chpc.utah.edu/common/home/u0379426/vast/bam_wgs/HG006_220807_S6/HG006_220807_S6.cram \
    /uufs/chpc.utah.edu/common/home/u0379426/vast/bam_wgs/HG006_222206_S3/HG006_222206_S3.cram \
    /uufs/chpc.utah.edu/common/home/u0379426/vast/bam_wgs/HG007_220807_S2/HG007_220807_S2.cram \
    /uufs/chpc.utah.edu/common/home/u0379426/vast/bam_wgs/HG007_220807_S8/HG007_220807_S8.cram; do
