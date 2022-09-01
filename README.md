
## NGS Variant detection with Transformers

This repo contains code for detecting variants from next-generation sequencing data (BAM / CRAM files)
 via sequence-to-sequence modeling. The input sequence is a list of pileup columns, and the output is two
predicted haplotypes, from which variants can be easily parsed. With this approach, there's not a need for
any sophisticated statistical procedures - no HMMs, no de Bruijn graphs, or decisions about variant 
quality cutoffs, kmer-sizes, etc. The approach allows for true end-to-end deep learning
for variant detection.

### Installation

Just navigate to the repository directory and 

    pip install  .

You'll need both conda (anaconda or miniconda) and pip installed.

You can also create a docker image with the Dockerfile, with a simple `docker build .`
in the main project directory


### Calling variants

Calling variants requires an alignment file in bam / cram format, a model file, 

    dnaseq2seq/main.py call -r <reference genome fasta> 
      --threads <number of threads to use> 
      -m /path/to/model
      --bed /path/to/BED formatted/file 
      --bam /BAM or CRAM file 
      -v output.vcf

Running the model with the command above will generate variant calls _without_ well
calibrated quality scores, which will likely have high sensitivity but poor precision 
(i.e. lots of false positive calls). Adding a 'classifier' model allows Jovian to 
compute meaning quality scores which greatly improve precision. The classifier also
requires a path to a population database VCF (such as Gnomad). To run with a classifier
just add the following args to the command line:

    -c /path/to/classifier.model
    -f /path/to/population/frequency/vcf


Calling does not utilize a GPU (running forward passes of the model 
accounts for only a small fraction of the total runtime).


### Training a new model


#### Creating training from labelled BAMs (`pregen`)

Training requires converting pileups (regions of BAM files) into tensors, but that process takes a long 
time so it makes sense to just do it once and save the tensors to disk so they can be used in multiple 
training runs. This is called `pregen` (for pre-generation of training data). The pregenerated training 
tensors and 'labels' (true alt sequences) are stored in a single directory. To create pregenerated training 
data, run

    ./main.py pregen --threads <thread count> 
      -c <conf.yaml> 
      -d /path/to/output/directory

Depending on how may BAMs and how many labeled instances there are, this can take a really long time.

The configuration file `conf.yaml` must have a path to the reference genome, the number of examples to 
choose from each region type, and a list of BAMs + labels, like this:

    reference: /path/to/reference/fasta

    vals_per_class:
        'snv': 5000   # keys here should match labels in fourth column of BED files provided below    
        'deletion': 500
        'insertion': 500
        'mnv': 1000

    data:
      - bam: /path/to/a/bam
        vcf: /path/to/truth/variants.vcf
        bed: /path/to/regions/to/sample.bed

      - bam: /path/to/another/bam
        vcf: /path/to/another/vcf
        bed: /path/to/another/regions/bed
      .... more bams/ labels ...

The BED files must be four-column BED files where the fourth column is the region label. 
The label scheme is completely flexible and up to the user, but in general should match the values 
in 'vals_per_class' from the configuration file. 


### Performing a training run

To train a new model, run a command similar to

    dnaseq2seq/main.py train -c conf.yaml 
      -d <pregen_data_directory>
      -n <number of epochs>
      --checkpoint-freq <model checkpointing frequency>
      --learning-rate 0.0001
      --batch-size 512 
      -o output.model


It's possible to continue training from a checkpoint by providing an input model with the

     --input-model <path to checkpoint model> 

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
