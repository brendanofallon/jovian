
## NGS Variant detection with Transformers

This repo contains experimental code for detecting variants from next-generation sequencing data (BAM / CRAM files)
 using Transformers. The transformer architecture allows variant detection to be treated as a sequence-to-sequence 
modeling problem, where the input sequence is a list of pileup columns, and the output is the predicted alt sequence.
WIth this approach, there's not a need for any sophisticated statistical procedures - no HMMs, no de Bruijn graphs, no
arbitrary decisions about variant quality cutoffs, kmer-sizes, etc. The approach allows for true end-to-end deep learning
for variant detection.

This repo is in early dev. It's not even installable at this point (there's no setup.py), but the `main.py` script is 
runnable.

### Dependencies

The following libraries are required:

    pysam
    pytorch
    numpy
    pandas
    pyyaml
    scipy

In addition, you must install `https://github.com/kyu999/ssw_aligner` manually - and it requires Cython==0.28.3 (that exact
version seems to be required currently). It's not installable via pip or conda currently. 


### Generating training data


#### Creating training from labelled BAMs (`pregen`)

Training requires converting pileups (regions of BAM files) into tensors, but that process takes a long time so it makes sense to just do it
once and save the tensors to disk so they can be used in multiple training runs. This is called `pregen` (for pre-generation of training data).
The pregenerated training tensors and 'labels' (true alt sequences) are stored in a single directory. To create pregenerated training data, run

    ./main.py pregen -c <conf.yaml> -d /path/to/output/directory

Depending on how may BAMs and how many labeled instances there are, this can take a really long time. Would be nice if it could be made multithreaded.

The configuration file `conf.yaml` must have a path to the reference genome and a list of BAMs + labels, like this:

    reference: /path/to/reference/fasta
    data:
      - bam: /path/to/a/bam
        labels: /path/to/labels.csv
      - bam: /path/to/another/bam
        labels: /path/to/another/labels.csv
      .... more bams/ labels ...
The 'labels.csv' contains a list of genomic regions, chrom/pos/ref/alt and 'TP/FN/FP' status. *This is the tp_fn_fp.csv" file produced by the Caravel calc_ppa_ppv task*,
no modifications needed. 

### Performing a training run

To train a new model, run

    ./main.py train -c conf.yaml -d <pregen_data_directory> -n <number of epochs> --checkpoint-freq <model checkpointing frequency> -o output.model

Training can, of course, take a pretty long time as well. If a GPU is available (and CUDA is installed properly) it will be used, significantly speeding
up training time. The CUDA device used is displayed in log messages, if you want to verify that the GPU is being used properly. 


### Evaluating a model

Model evaluation is in early stages, but it is possible to run a command like:

    ./main.py evalbam -m <model> --bam <path to BAM> --labels <path to labels>

which will generate a textual summary of the PPA / PPV / TPs / FPs / FNs for each variant class. Additional investigating probably requires debugging



### TODOs

Engineering stuff:

    - When training, retain a few batches / samples for validation. Don't use them for training but report their accuracy once per epoch
    - Log basic stats to csv or similar when training, so its easy to look at train & val loss / gradient norm / % match etc 
    - Save best model (lowest loss?) during training
    - Tensorboard integration?
    - VCF output
    - When calling a given region, do we just want to make one prediction from the model? What if the number of reads in the pileup is much bigger
    than the 'read_depth' dimension of the model - should we repeatedly sample the reads, make multiple predictions, and look for variants that appear
    in (most) of the replicates? Should we randomize on the target position as well?
    
Research stuff:

    - Learning curves - how does validation accuracy improve as more training data is added?
    - Model hyperparameter tweaking - How many encoder layers should we use? What the best embedding dimension? Whats the best output size for the initial linear layers? Do we need those two initial linear layers?
    - Should we somehow encode read identity into the features? Right now, there's no way (I think) for the model to figure that a given basecall in one position came from the same read as a basecall at a different position
    - Maybe there's a better way to do alt masking / prediction? Would a transformer work here as well?
    - 
