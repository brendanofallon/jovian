#!/usr/bin/env python


import logging
import random
from collections import defaultdict, Counter
from pathlib import Path
from string import ascii_letters, digits
import gzip
import lz4.frame
import tempfile
from datetime import datetime
import re

import pysam
import torch


import torch.multiprocessing as mp
from concurrent.futures.process import ProcessPoolExecutor
import numpy as np
import argparse
import yaml

import phaser
import util
import vcf
import loader
from bam import string_to_tensor, target_string_to_tensor, encode_pileup3, reads_spanning, alnstart, ensure_dim, \
    reads_spanning_range
from model import VarTransformer
from train import train, load_train_conf, eval_prediction


logging.basicConfig(format='[%(asctime)s]  %(name)s  %(levelname)s  %(message)s',
                    datefmt='%m-%d %H:%M:%S',
                    level=logging.INFO) # handlers=[RichHandler()])
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda") if hasattr(torch, 'cuda') and torch.cuda.is_available() else torch.device("cpu")


def call(statedict, bam, reference, chrom, pos, **kwargs):
    raise NotImplemented




def load_conf(confyaml):
    logger.info(f"Loading configuration from {confyaml}")
    conf = yaml.safe_load(open(confyaml).read())
    assert 'reference' in conf, "Expected 'reference' entry in training configuration"
    assert 'data' in conf, "Expected 'data' entry in training configuration"
    return conf



def pregen_one_sample(dataloader, batch_size, output_dir):
    """
    Pregenerate tensors for a single sample
    """
    uid = "".join(random.choices(ascii_letters + digits, k=8))
    src_prefix = "src"
    tgt_prefix = "tgt"
    vaf_prefix = "vaftgt"
    metafile = tempfile.NamedTemporaryFile(
        mode="wt", delete=False, prefix="pregen_", dir=".", suffix=".txt"
    )
    logger.info(f"Saving tensors to {output_dir}/")
    for i, (src, tgt, vaftgt, varsinfo) in enumerate(dataloader.iter_once(batch_size)):
        logger.info(f"Saving batch {i} with uid {uid}")
        for data, prefix in zip([src, tgt, vaftgt],
                                [src_prefix, tgt_prefix, vaf_prefix]):
            with lz4.frame.open(output_dir / f"{prefix}_{uid}-{i}.pt.lz4", "wb") as fh:
                torch.save(data, fh)
        for idx, varinfo in enumerate(varsinfo):
            meta_str = "\t".join([
                f"{idx}", f"{uid}-{i}", "\t".join(varinfo), dataloader.csv
            ]) 
            print(meta_str, file=metafile)
        metafile.flush()

    metafile.close()
    return metafile.name


def default_vals_per_class():
    """
    Multiprocess will instantly deadlock if a lambda or any callable not defined on the top level of the module is given
    as the 'factory' argument to defaultdict - but we have to give it *some* callable that defines the behavior when the key
    is not present in the dictionary, so this returns the default "vals_per_class" if a class is encountered that is not 
    specified in the configuration file. I don't think there's an easy way to make this user-settable, unfortunately
    """
    return 0


def pregen(config, **kwargs):
    """
    Pre-generate tensors from BAM files + labels and save them in 'datadir' for quicker use in training
    (this takes a long time)
    """
    conf = load_conf(config)
    batch_size = kwargs.get('batch_size', 64)
    reads_per_pileup = kwargs.get('read_depth', 100)
    samples_per_pos = kwargs.get('samples_per_pos', 3)
    vals_per_class = defaultdict(default_vals_per_class)
    vals_per_class.update(conf['vals_per_class'])

    output_dir = Path(kwargs.get('dir'))
    metadata_file = kwargs.get("metadata_file", None)
    if metadata_file is None:
        str_time = datetime.now().strftime("%Y_%d_%m_%H_%M_%S")
        metadata_file = f"pregen_{str_time}.csv"
    processes = kwargs.get('threads', 1)

    logger.info(f"Generating training data using config from {config} vals_per_class: {vals_per_class}")
    dataloaders = [
            loader.LazyLoader(c['bam'], c['bed'], c['vcf'], conf['reference'], reads_per_pileup, samples_per_pos, vals_per_class)
        for c in conf['data']
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Submitting {len(dataloaders)} jobs with {processes} process(es)")

    meta_headers = ["item", "uid", "chrom", "pos", "ref", "alt", "vaf", "label"]
    with open(metadata_file, "wb") as metafh:
        metafh.write(("\t".join(meta_headers) + "\n").encode())
        if processes == 1:
            for dl in dataloaders:
                sample_metafile = pregen_one_sample(dl, batch_size, output_dir)
                util.concat_metafile(sample_metafile, metafh)
        else:
            futures = []
            with ProcessPoolExecutor(max_workers=processes) as executor:
                for dl in dataloaders:
                    futures.append(executor.submit(pregen_one_sample, dl, batch_size, output_dir))
            for fut in futures:
                sample_metafile = fut.result()
                util.concat_metafile(sample_metafile, metafh)


def callvars(model, aln, reference, chrom, start, end, window_width, max_read_depth):
    """
    Call variants in a region of a BAM file using the given altpredictor and model
    and return a list of vcf.Variant objects
    """
    reads = reads_spanning_range(aln, chrom, start, end)
    if len(reads) < 5:
        raise ValueError(f"Hmm, couldn't find any reads spanning {chrom}:{pos}")
    if len(reads) > max_read_depth:
        reads = random.sample(reads, max_read_depth)
    reads = util.sortreads(reads)
    minref = min(alnstart(r) for r in reads)
    maxref = max(alnstart(r) + r.query_length for r in reads)
    reads_encoded, _ = encode_pileup3(reads, minref, maxref)

    refseq = reference.fetch(chrom, minref, minref + reads_encoded.shape[0])
    reftensor = string_to_tensor(refseq)
    reads_w_ref = torch.cat((reftensor.unsqueeze(1), reads_encoded), dim=1)
    padded_reads = ensure_dim(reads_w_ref, maxref - minref, max_read_depth).unsqueeze(0).to(DEVICE)
    midstart = max(0, start - minref)
    midend = midstart + window_width

    if padded_reads.shape[1] > window_width:
        padded_reads = padded_reads[:, midstart:midend, :, :]

    #masked_reads = padded_reads * fullmask
    seq_preds = model(padded_reads.float().to(DEVICE))
    return seq_preds[0, 0, :, :], seq_preds[0, 1, :, :], start

def _var_type(variant):
    if len(variant.ref) == 1 and len(variant.alt) == 1:
        return 'snv'
    elif len(variant.ref) == 0 and len(variant.alt) > 0:
        return 'ins'
    elif len(variant.ref) > 0 and len(variant.alt) == 0:
        return 'del'
    elif len(variant.ref) > 0 and len(variant.alt) > 0:
        return 'mnv'
    print(f"Whoa, unknown variant type: {variant}")
    return 'unknown'


def _call_vars_region(aln, model, reference, chrom, start, end, max_read_depth, window_size=300):
    """
    For the given region, identify variants by repeatedly calling the model over a sliding window,
    tallying all of the variants called, and them making a choice about which ones are 'real'
    Currently, we exclude all variants in the downstream half of the window, and retain only the remaining
    variants that are called in more than one window

    TODO: Handle haplotypes appropriately
    """
    window_step = 50
    var_retain_window_size = 150
    allvars0 = defaultdict(list)
    allvars1 = defaultdict(list)
    window_start = start - 2 * window_step # We start with regions a bit upstream of the focal / target region
    while window_start <= (end - window_step):
        # print(f"Window {window_start}-{window_start + var_retain_window_size}")
        hap0_t, hap1_t, offset = callvars(model, aln, reference, chrom, window_start, window_start + window_size, window_size,
                                          max_read_depth=max_read_depth)
        hap0 = util.readstr(hap0_t)
        hap1 = util.readstr(hap1_t)
        hap0_probs = hap0_t.detach().numpy().max(axis=-1)
        hap1_probs = hap1_t.detach().numpy().max(axis=-1)
        refseq = reference.fetch(chrom, offset, offset + window_size)
        vars_hap0 = list(vcf.aln_to_vars(refseq, hap0, offset, hap0_probs))
        vars_hap1 = list(vcf.aln_to_vars(refseq, hap1, offset, hap1_probs))
        for v in vars_hap0:
            if v.pos < (window_start + var_retain_window_size):
                allvars0[v.pos].append(v)
        for v in vars_hap1:
            if v.pos < (window_start + var_retain_window_size):
                allvars1[v.pos].append(v)

        window_start += window_step
    # print("Hap0 vars:")
    # for pos, variants in allvars0.items():
    #     print(f"{pos} : {variants}")
    # print("Hap1 vars:")
    # for pos, variants in allvars1.items():
    #     print(f"{pos} : {variants}")

    # Return variants that occur more than once?
    hap0_passing = list(v[0] for k, v in allvars0.items() if len(v) > 1 and start < v[0].pos < end)
    hap1_passing = list(v[0] for k, v in allvars1.items() if len(v) > 1 and start < v[0].pos < end)
    return hap0_passing, hap1_passing


def eval_labeled_bam(config, bam, labels, statedict, truth_vcf, **kwargs):
    """
    Call variants in BAM file with given model at positions given in the labels CSV, emit useful
    summary information about PPA / PPV, etc
    """
    max_read_depth = 100
    feats_per_read = 9
    logger.info(f"Found torch device: {DEVICE}")
    conf = load_conf(config)

    reference = pysam.FastaFile(conf['reference'])
    truth_vcf = pysam.VariantFile(truth_vcf)
    attention_heads = 2
    transformer_dim = 400
    encoder_layers = 8
    embed_dim_factor = 200
    model = VarTransformer(read_depth=max_read_depth,
                                    feature_count=feats_per_read,
                                    out_dim=4,
                                    embed_dim_factor=embed_dim_factor,
                                    nhead=attention_heads,
                                    d_hid=transformer_dim,
                                    n_encoder_layers=encoder_layers,
                                    device=DEVICE)

    model.load_state_dict(torch.load(statedict, map_location=DEVICE))
    model.eval()

    aln = pysam.AlignmentFile(bam)
    results = defaultdict(Counter)
    window_size = 300

    tot_tps = 0
    tot_fps = 0
    tot_fns = 0
    results = defaultdict(Counter)
    for i, line in enumerate(open(labels)):
        tps = []
        fps = []
        fns = []
        toks = line.strip().split("\t")
        chrom = toks[0]
        start = int(toks[1])
        end = int(toks[2])
        label = toks[3]

        fp_varpos = []
        tp_varpos = []
        try:
            vars_hap0, vars_hap1 = _call_vars_region(aln, model, reference, chrom, start, end, max_read_depth=100,
                                                   window_size=300)
        except Exception as ex:
            logger.warning(f"Hmm, exception processing {chrom}:{start}-{end}, skipping it")
            logger.warning(ex)
            continue

        print(f"[{start}]  {chrom}:{start}-{start + window_size} ", end='')

        refwidth = end-start
        refseq = reference.fetch(chrom, start, start + refwidth)
        variants = list(truth_vcf.fetch(chrom, start, start + refwidth))

        # WONT ALWAYS WORK: Grab *ALL* variants and generate a single alt sequence with everything???
        pseudo_altseq = phaser.project_vars(variants, [np.argmax(v.samples[0]['GT']) for v in variants], refseq, start)
        pseudo_vars = list(vcf.aln_to_vars(refseq, pseudo_altseq, start))


        print(f" true: {len(pseudo_vars)}", end='')

        var_types = set()
        for true_var in pseudo_vars:
            var_type = _var_type(true_var)
            var_types.add(var_type)
            # print(f"{true_var} ", end='')
            # print(f" hap0: {true_var in vars_hap0}, hap1: {true_var in vars_hap1}")
            if true_var in vars_hap0 or true_var in vars_hap1:
                tps.append(true_var)
                tot_tps += 1
                results[var_type]['tp'] += 1
                tp_varpos.append(true_var.pos - start)
            else:
                fns.append(true_var)
                tot_fns += 1
                results[var_type]['fn'] += 1
        print(f" {', '.join(var_types)} TP: {len(tps)} FN: {len(fns)}", end='')

        for var0 in vars_hap0:
            var_type = _var_type(var0)
            if var0 not in pseudo_vars:
                fps.append(var0)
                fp_varpos.append(var0.pos - start)
                tot_fps += 1
                results[var_type]['fp'] += 1
        for var1 in vars_hap1:
            var_type = _var_type(var1)
            if var1 not in pseudo_vars and var1 not in vars_hap0:
                fps.append(var1)
                fp_varpos.append(var1.pos - start)
                tot_fps += 1
                results[var_type]['fp'] += 1


        tp_pos = ", ".join(str(s) for s in tp_varpos)
        fp_pos = ", ".join(str(s) for s in fp_varpos[0:10])
        print(f" FP: {len(fps)}\t[{tp_pos}]  [{fp_pos}]")

    for key, val in results.items():
        print(f"{key} : total entries: {sum(val.values())}")
        for t, count in val.items():
            print(f"\t{t} : {count}")


def print_pileup(path, idx, target=None, **kwargs):
    path = Path(path)

    suffix = path.name.split("_")[-1]
    tgtpath = path.parent / f"tgt_{suffix}"
    if tgtpath.exists():
        tgt = util.tensor_from_file(tgtpath, device='cpu')
        logger.info(f"Found target file: {tgtpath}, loaded tensor of shape {tgt.shape}")
        for i in range(tgt.shape[1]):
            t = tgt[idx, i, :]
            bases = util.tgt_str(tgt[idx, i, :])
            print(bases)
    else:
        logger.info(f"No tgt file found (look for {tgtpath})")

    src = util.tensor_from_file(path, device='cpu')
    logger.info(f"Loaded tensor with shape {src.shape}")
    s = util.to_pileup(src[idx, :, :, :])
    print(s)





def alphanumeric_no_spaces(name):
    if re.match(r"[a-zA-Z0-9_-]+", name):
        return name
    else:
        raise argparse.ArgumentTypeError(f"{name} is not an alphanumeric plus '_' or '-' without spaces")


def main():
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers()

    genparser = subparser.add_parser("pregen", help="Pre-generate tensors from BAMs")
    genparser.add_argument("-c", "--config", help="Training configuration yaml", required=True)
    genparser.add_argument("-d", "--dir", help="Output directory", default=".")
    genparser.add_argument("-s", "--sim", help="Generate simulated data", action='store_true')
    genparser.add_argument("-b", "--batch-size", help="Number of pileups to include in a single file (basically the batch size)", default=64, type=int)
    genparser.add_argument("-n", "--start-from", help="Start numbering from here", type=int, default=0)
    genparser.add_argument("-t", "--threads", help="Number of processes to use", type=int, default=1)
    # genparser.add_argument("-vpc", "--vals-per-class", help="The number of instances for each variant class in a label file; it will be set automatically if not specified", type=int, default=1000)
    genparser.add_argument("-mf", "--metadata-file", help="The metadata file that records each row in the encoded tensor files and the variant from which that row is derived. The name pregen_{time}.csv will be used if not specified.")
    genparser.set_defaults(func=pregen)

    printpileupparser = subparser.add_parser("print", help="Print a tensor pileup")
    printpileupparser.add_argument("-p", "--path", help="Path to saved tensor data", required=True)
    printpileupparser.add_argument("-i", "--idx", help="Index of item in batch to emit", required=True, type=int)
    printpileupparser.set_defaults(func=print_pileup)

    evalbamparser = subparser.add_parser("evalbam", help="Evaluate a BAM with labels")
    evalbamparser.add_argument("-c", "--config", help="Training configuration yaml", required=True)
    evalbamparser.add_argument("-m", "--statedict", help="Stored model", required=True)
    evalbamparser.add_argument("-b", "--bam", help="Input BAM file", required=True)
    evalbamparser.add_argument("-v", "--truth-vcf", help="Truth VCF", required=True)
    evalbamparser.add_argument("-l", "--labels", help="CSV file with truth variants", required=True)
    evalbamparser.set_defaults(func=eval_labeled_bam)

    trainparser = subparser.add_parser("train", help="Train a model")
    trainparser.add_argument("-n", "--epochs", type=int, help="Number of epochs to train for", default=100)
    trainparser.add_argument("-i", "--input-model", help="Start with parameters from given state dict")
    trainparser.add_argument("-o", "--output-model", help="Save trained state dict here", required=True)
    trainparser.add_argument("-ch", "--checkpoint-freq", help="Save model checkpoints frequency (0 to disable)", default=10, type=int)
    trainparser.add_argument("-lr", "--learning-rate", help="Initial learning rate", default=0.001, type=float)
    trainparser.add_argument("-c", "--config", help="Training configuration yaml", required=True)
    trainparser.add_argument("-d", "--datadir", help="Pregenerated data dir", default=None)
    trainparser.add_argument("-vd", "--val-dir", help="Pregenerated data for validation", default=None)
    trainparser.add_argument("-t", "--threads", help="Max number of threads to use for decompression (torch may use more)", default=4, type=int)
    trainparser.add_argument("-md", "--max-decomp-batches",
                             help="Max number batches to decompress and store in memory at once", default=4, type=int)
    trainparser.add_argument("-b", "--batch-size", help="The batch size, default is 64", type=int, default=64)
    trainparser.add_argument("-da", "--data-augmentation", help="Specify data augmentation options: 'shortening', 'shuffling', 'downsampling'. You can provide multiple options. Default is None", nargs="+", default=None)
    trainparser.add_argument("-fa", "--fraction-to-augment", help="Fraction of sample batches to augment. Needed with '--data-augmentation' option. Default is 0.25", default=0.25, type=float)
    trainparser.add_argument("-rn", "--wandb-run-name", type=alphanumeric_no_spaces, default=None,
                             help="Weights & Biases run name, must be alphanumeric plus '_' or '-'")
    trainparser.add_argument("--wandb-notes", type=str, default=None,
                             help="Weights & Biases run notes, longer description of run (like 'git commit -m')")
    trainparser.add_argument("--loss", help="Loss function to use, use 'ce' for CrossEntropy or 'sw' for Smith-Waterman", choices=['ce', 'sw'], default='ce')
    trainparser.set_defaults(func=train)

    callparser = subparser.add_parser("call", help="Call variants")
    callparser.add_argument("-m", "--statedict", help="Stored model", required=True)
    callparser.add_argument("-r", "--reference", help="Path to Fasta reference genome", required=True)
    callparser.add_argument("-b", "--bam", help="Input BAM file", required=True)
    callparser.add_argument("--chrom", help="Chromosome", required=True)
    callparser.add_argument("--pos", help="Position", required=True, type=int)
    callparser.set_defaults(func=call)

    args = parser.parse_args()
    args.cl_args = vars(args).copy()  # command line copy for logging
    args.func(**vars(args))


if __name__ == "__main__":
    main()
