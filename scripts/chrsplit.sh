#!/usr/bin/bash

set -x

INBED=$1
PREFIX=$(basename $INBED | cut -d '_' -f1)
grep -e '^1\s\|^2\s\|^19\s\|^20\s' $INBED > ${PREFIX}_labels_chrsA.bed
grep -e '^3\s\|^4\s\|^18\s\|^17\s' $INBED > ${PREFIX}_labels_chrsB.bed
grep -e '^5\s\|^6\s\|^15\s\|^16\s' $INBED > ${PREFIX}_labels_chrsC.bed
grep -e '^7\s\|^8\s\|^13\s\|^14\s' $INBED > ${PREFIX}_labels_chrsD.bed
grep -e '^9\s\|^10\s\|^11\s\|^12\s' $INBED > ${PREFIX}_labels_chrsE.bed

grep -e '^21\s\|^22\s' $INBED > ${PREFIX}_labels_chrs21and22.bed

