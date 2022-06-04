
## NGS Variant detection with Transformers

This repo contains code for detecting variants from next-generation sequencing data (BAM / CRAM files)
 via sequence-to-sequence modeling. The input sequence is a list of pileup columns, and the output is two predicted haplotypes, from which variants can be easily parsed. With this approach, there's not a need for any sophisticated statistical procedures - no HMMs, no de Bruijn graphs, or decisions about variant quality cutoffs, kmer-sizes, etc. The approach allows for true end-to-end deep learning
for variant detection.

### Installation

Just navigate the to repository directory and 

    pip install .

### Dependencies

The following libraries are required:

    pysam
    pytorch
    numpy
    pandas
    pyyaml
    scipy
    tensorboard
    lz4
    pygit2
    ssw_aligner
    scikit-bio
    intervaltree
    pyranges
    

In addition, you must install `https://github.com/kyu999/ssw_aligner` manually - and it requires Cython==0.28.3 (that exact
version seems to be required currently). It's not installable via pip or conda currently. 


### Generating training data


#### Creating training from labelled BAMs (`pregen`)

Training requires converting pileups (regions of BAM files) into tensors, but that process takes a long time so it makes sense to just do it once and save the tensors to disk so they can be used in multiple training runs. This is called `pregen` (for pre-generation of training data). The pregenerated training tensors and 'labels' (true alt sequences) are stored in a single directory. To create pregenerated training data, run

    ./main.py pregen --threads <thread count> -c <conf.yaml> -d /path/to/output/directory

Depending on how may BAMs and how many labeled instances there are, this can take a really long time.

The configuration file `conf.yaml` must have a path to the reference genome, the number of examples to choose from
each region type, and a list of BAMs + labels, like this:

    reference: /path/to/reference/fasta

    vals_per_class:
        'snv': 5000
        'deletion': 500
        'insertion': 500
        'mnv': 1000

    data:
      - bam: /path/to/a/bam
        labels: /path/to/labels.bed
      - bam: /path/to/another/bam
        labels: /path/to/another/labels.bed
      .... more bams/ labels ...

The 'labels.bed' file is a four-column BED file where the fourth column is the region label. The label scheme is completely flexible and up to the user, but in general should match the values in 'vals_per_class' from the configuration file. 


### Performing a training run

To train a new model, run

    ./main.py train -c conf.yaml -d <pregen_data_directory> -n <number of epochs> --checkpoint-freq <model checkpointing frequency> -o output.model

Training can, of course, take a pretty long time as well. If a GPU is available (and CUDA is installed properly) it will be used, significantly speeding up training time. The CUDA device used is displayed in log messages, if you want to verify that the GPU is being used properly. 


### Training a 'classifier'

In order to generate well-calibrated quality scores, it's necessary to train a small classifier that learns to predict true and false positive variants. To do this, you must create a configuration file that links VCFs with examples of true and false positive variants to BAM files. The configuration file specifies one or more samples, each of which has a BAM/CRAM file and one or more sets of true positive and false positive variants. Like this:

    sample1:
      bam: /path/to/sample1.cram
      fps:
      - false_positive_calls.vcf
      - some_falsepositives.vcf
      tps:
      - true_positives.vcf
    
    sample2:
      bam: /path/to/sample2.bam
      fps:
      - sample2_falsepositives.vcf
      tps:
      - sample2_truepositives.vcf


To generate the classifier, run the dnaseq2seq/builddclf.py tool 

### Calling variants

With a trained model 
