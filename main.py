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

import pysam
import torch
import pandas as pd

import torch.multiprocessing as mp
from concurrent.futures.process import ProcessPoolExecutor
import numpy as np
import argparse
import yaml

import bwasim
import model
import sim
import util
import vcf
import loader
from bam import string_to_tensor, target_string_to_tensor, encode_pileup3, reads_spanning, alnstart, ensure_dim
from model import VarTransformer, VarTransformerAltMask
from train import train, load_train_conf, eval_prediction


logging.basicConfig(format='[%(asctime)s]  %(name)s  %(levelname)s  %(message)s',
                    datefmt='%m-%d %H:%M:%S',
                    level=logging.INFO) # handlers=[RichHandler()])
logger = logging.getLogger(__name__)


DEVICE = torch.device("cuda") if hasattr(torch, 'cuda') and torch.cuda.is_available() else torch.device("cpu")


def call(statedict, bam, reference, chrom, pos, **kwargs):
    max_read_depth = 300
    feats_per_read = 8

    altpredictor = AltPredictor(0, 7)
    altpredictor.load_state_dict(torch.load("altpredictor3.sd"))
    altpredictor.to(DEVICE)

    model = VarTransformer(read_depth=max_read_depth, feature_count=feats_per_read, out_dim=4, nhead=6, d_hid=300, n_encoder_layers=2).to(DEVICE)
    model.load_state_dict(torch.load(statedict))
    model.eval()

    bam = pysam.AlignmentFile(bam)
    reads = reads_spanning(bam, chrom, pos, max_reads=max_read_depth)
    if len(reads) < 5:
        raise ValueError(f"Hmm, couldn't find any reads spanning {chrom}:{pos}")

    minref = min(alnstart(r) for r in reads)
    maxref = max(alnstart(r) + r.query_length for r in reads)
    reads_encoded, _ = encode_pileup3(reads, minref, maxref)

    refseq = pysam.FastaFile(reference).fetch(chrom, minref, minref + reads_encoded.shape[0])
    reftensor = string_to_tensor(refseq)
    reads_w_ref = torch.cat((reftensor.unsqueeze(1), reads_encoded), dim=1)
    padded_reads = ensure_dim(reads_w_ref, maxref-minref, max_read_depth).unsqueeze(0)

    fullmask = create_altmask(altpredictor, padded_reads)
    masked_reads = padded_reads * fullmask
    
    print(util.to_pileup(padded_reads[0, :, :, :]))
    seq_preds, _ = model(masked_reads.flatten(start_dim=2))
    pred1str = util.readstr(seq_preds[0, :, :])
    print("\n")
    print(refseq)
    print(pred1str)

    midwith = 100
    for v in vcf.aln_to_vars(refseq[len(refseq) // 2 - midwith // 2:len(refseq) // 2 + midwith // 2],
                            pred1str[len(refseq)//2 - midwith//2:len(refseq)//2 + midwith//2],
                             offset=minref + len(refseq)//2 - midwith//2 + 1):
        print(v)





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
    return 500


def pregen(config, **kwargs):
    """
    Pre-generate tensors from BAM files + labels and save them in 'datadir' for quicker use in training
    (this takes a long time)
    """
    conf = load_conf(config)
    batch_size = kwargs.get('batch_size', 64)
    reads_per_pileup = kwargs.get('read_depth', 300)
    samples_per_pos = kwargs.get('samples_per_pos', 10)
    vals_per_class = defaultdict(default_vals_per_class)
    vals_per_class.update(conf['vals_per_class'])

    output_dir = Path(kwargs.get('dir'))
    metadata_file = kwargs.get("metadata_file", None)
    if metadata_file is None:
        str_time = datetime.now().strftime("%Y_%d_%m_%H_%M_%S")
        metadata_file = f"pregen_{str_time}.csv"
    processes = kwargs.get('threads', 1)
    if kwargs.get("sim"):
        batches = 50
        logger.info(f"Generating simulated data with batch size {batch_size} and {batches} total batches")
        dataloaders = [loader.BWASimLoader(DEVICE,
                                     regions=conf['regions'],
                                     refpath=conf['reference'],
                                     readsperpileup=200,
                                     readlength=145,
                                     error_rate=0.02,
                                     clip_prob=0.01)]
        dataloaders[0].batches_in_epoch = batches
    else:
        logger.info(f"Generating training data using config from {config} vals_per_class: {vals_per_class}")
        dataloaders = [
                loader.LazyLoader(c['bam'], c['labels'], conf['reference'], reads_per_pileup, samples_per_pos, vals_per_class)
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


def callvars(model, aln, reference, chrom, pos, window_width, max_read_depth):
    """
    Call variants in a region of a BAM file using the given altpredictor and model
    and return a list of vcf.Variant objects
    """
    reads = reads_spanning_range(aln, chrom, pos - 10, pos + 10)
    reads = util.sortreads(random.sample(reads, min(len(reads), maxreads)))
    if len(reads) < 5:
        raise ValueError(f"Hmm, couldn't find any reads spanning {chrom}:{pos}")

    minref = min(alnstart(r) for r in reads)
    maxref = max(alnstart(r) + r.query_length for r in reads)
    reads_encoded, _ = encode_pileup3(reads, minref, maxref)

    refseq = reference.fetch(chrom, minref, minref + reads_encoded.shape[0])
    reftensor = string_to_tensor(refseq)
    reads_w_ref = torch.cat((reftensor.unsqueeze(1), reads_encoded), dim=1)
    padded_reads = ensure_dim(reads_w_ref, maxref - minref, max_read_depth).unsqueeze(0).to(DEVICE)

    focal_min = padded_reads.shape[1] // 2 - window_width // 2 + 0
    focal_max = padded_reads.shape[1] // 2 + window_width // 2 + 0
    trimmed_reads = padded_reads[:, focal_min:focal_max, :, :]
    print(f"Window {minref} [{minref + focal_min}  {pos}  {minref + focal_max}]  {maxref}")
    refseq = refseq[focal_min:focal_max]
    #masked_reads = padded_reads * fullmask
    seq_preds, _ = model(trimmed_reads.float().to(DEVICE))
    pred1str = util.readstr(seq_preds[0, :, :])

    # for i, r, p in zip(range(minref + focal_min, minref + focal_max), refseq, pred1str):
    #     print(f"{i}\t{r}\t{p} {'*' if r != p else ''}")

    variants = [v for v in vcf.aln_to_vars(refseq,
                             pred1str,
                             offset=minref + focal_min)]
    return variants, seq_preds.squeeze(0),  minref + focal_min


def eval_labeled_bam(config, bam, labels, statedict, **kwargs):
    """
    Call variants in BAM file with given model at positions given in the labels CSV, emit useful
    summary information about PPA / PPV, etc
    """
    max_read_depth = 300
    feats_per_read = 9
    logger.info(f"Found torch device: {DEVICE}")
    conf = load_conf(config)

    reference = pysam.FastaFile(conf['reference'])

    model = VarTransformerAltMask(read_depth=max_read_depth, feature_count=feats_per_read, out_dim=4, nhead=8, d_hid=400, n_encoder_layers=4, device=DEVICE).to(DEVICE)
    model.load_state_dict(torch.load(statedict, map_location=DEVICE))
    model.eval()

    aln = pysam.AlignmentFile(bam)
    results = defaultdict(Counter)

    window_size = 60

    for i, row in pd.read_csv(labels).iterrows():
        pos = int(row.pos)
        if row.ngs_vaf < 0.05:
            vafgroup = "< 0.05"
        elif 0.05 <= row.ngs_vaf < 0.25:
            vafgroup = "0.05 - 0.25"
        elif 0.25 <= row.ngs_vaf < 0.50:
            vafgroup = "0.25 - 0.50"
        else:
            vafgroup = "> 0.50"

        labeltype = f"{row.vtype}-{row.status} {vafgroup}"
        if results[labeltype]['rows'] > 20:
            continue
        try:
            variants, seq_preds, pred_start = callvars(model, aln, reference, str(row.chrom), pos, window_size, max_read_depth=max_read_depth)
        except Exception as ex:
            logger.warning(f"Hmm, exception processing {row.chrom}:{row.pos}, skipping it")
            logger.warning(ex)
            continue
        refwidth = seq_preds.shape[0]
        refseq = reference.fetch(str(row.chrom), pred_start, pred_start + seq_preds.shape[0])
        if row.status == 'TP' or row.status == 'FN':
            true_altseq = refseq[0:pos-pred_start-1] + row.alt + refseq[pos - pred_start + len(row.ref)-1:]
        else:
            true_altseq = refseq

        tps, fps, fns = eval_prediction(refseq, true_altseq, seq_preds,
                                        midwidth=min(refwidth, window_size))
        print(f"{row.chrom}:{row.pos} {row.ref} -> {row.alt}\t{row.status} VAF: {row.ngs_vaf:.4f} TP: {len(tps)} FP: {len(fps)} FN: {len(fns)}")
        # print(f"TPs: {tps}")
        # print(f"FPs: {fps}")
        # print(f"TP: {total_tps} FP: {total_fps} FNs: {total_fns}")
        results[labeltype]['TP'] += len(tps)
        results[labeltype]['FP'] += len(fps)
        results[labeltype]['FN'] += len(fns)
        results[labeltype]['rows'] += 1

    for key, val in results.items():
        print(f"{key} : total entries: {val['rows']}")
        for t, count in val.items():
            print(f"\t{t} : {count}")


def print_pileup(path, idx, target=None, **kwargs):
    src = util.tensor_from_file(path, device='cpu')
    logger.info(f"Loaded tensor with shape {src.shape}")
    s = util.to_pileup(src[idx, :, :, :])
    print(s)


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
    trainparser.add_argument("-da", "--data-augmentation", action="store_true", help="Specify --data-augmentation to perform data augmentation via diff loaders, default is false", default=False)
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
    args.func(**vars(args))


if __name__ == "__main__":
    main()
