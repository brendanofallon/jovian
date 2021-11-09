## Useful Scripts for Variant detection with Transformers

This scripts dir contains useful scripts for data processing and analysis

### run_pregen.sh
example script to pregenerate data for training

### run_train.sh
example script to train with pregenerated data

### run_train_clone.sh


### csv_labels_utils.py
A few useful options for manipulating the labels.csv files used in the pregen step
 - tns: add true negative entries in the labels.csv file
   - output is a labels.csv file with new randomly chosen true negative entries
   - input includes a labels csv, the reference sequence, and bed files that intersect to define high quality, ROI regions
   - assumes that input labels csv file contains all true variants
 - split: split labels.csv into two based on a given list of chromosomes
 - vtype: adds 'vtype' column to lables csv file based on a pretty simple look at 'ref' and 'alt' columns


