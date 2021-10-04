#!/usr/bin/env python


import logging
import random
from collections import defaultdict, Counter
from pathlib import Path
from string import ascii_letters, digits
import gzip
import lz4.frame

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
from model import VarTransformer, AltPredictor, VarTransformerAltMask
from train import train, load_train_conf


logging.basicConfig(format='[%(asctime)s]  %(name)s  %(levelname)s  %(message)s',
                    datefmt='%m-%d %H:%M:%S',
                    level=logging.INFO) # handlers=[RichHandler()])
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda:0") if hasattr(torch, 'cuda') and torch.cuda.is_available() else torch.device("cpu")


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


def eval_prediction(refseqstr, altseq, predictions, midwidth=100):
    """
    Given a target sequence and two predicted sequences, attempt to determine if the correct *Variants* are
    detected from the target. This uses the vcf.align_seqs(seq1, seq2) method to convert string sequences to Variant
    objects, then compares variant objects
    :param tgt:
    :param predictions:
    :param midwidth:
    :return: Sets of TP, FP, and FN vars
    """
    midstart = len(refseqstr) // 2 - midwidth // 2
    midend  =  len(refseqstr) // 2 + midwidth // 2
    known_vars = []
    for v in vcf.aln_to_vars(refseqstr, altseq):
        if midstart < v.pos < midend:
            known_vars.append(v)

    pred_vars = []
    for v in vcf.aln_to_vars(refseqstr, util.readstr(predictions)):
        if midstart < v.pos < midend:
            v.qual = predictions[v.pos:v.pos + max(1, min(len(v.ref), len(v.alt))), :].max(dim=1)[0].min().item()
            pred_vars.append(v)

    tps = [] # True postive - real and detected variant
    fns = [] # False negatives - real variant but not detected
    fps = [] # False positives - detected but not a real variant
    for true_var in known_vars:
        if true_var in pred_vars:
            tps.append(true_var)
        else:
            fns.append(true_var)

    for detected_var in pred_vars:
        if detected_var not in known_vars:
            #logger.info(f"FP: {detected_var}")
            fps.append(detected_var)

    return tps, fps, fns



def create_altmask(altmaskmodel, src):
    """
    Create a new tensor of the same dimension as src that weights individual reads by their probability
    of supporting a variant.
    :param altmaskmodel: A function that returns a pytorch Tensor with dimension [batch, read], containing
                        a value 0-1 for each read
    :param src: Encoded pileup of dimension [batch, seq pos, read, feature]
    :returns: Tensor with same dimension as src, with values 0-1 in read dimension replicated into features
    """
    predicted_altmask = altmaskmodel(src.to(DEVICE))
    amin = predicted_altmask.min(dim=1)[0].unsqueeze(1).expand((-1, predicted_altmask.shape[1]))
    amx = 0.95 / (predicted_altmask - amin).max(dim=1)[0]
    predicted_altmask = (predicted_altmask - amin) * amx.unsqueeze(1).expand((-1, predicted_altmask.shape[1])) # + amin
    predicted_altmask = torch.cat((torch.ones(src.shape[0], 1).to(DEVICE), predicted_altmask[:, 1:]), dim=1)
    predicted_altmask = predicted_altmask.clamp(0.001, 1.0)
    aex = predicted_altmask.unsqueeze(-1).unsqueeze(-1)
    fullmask = aex.expand(src.shape[0], src.shape[2], src.shape[1],
                          src.shape[3]).transpose(1, 2).to(DEVICE)
    return fullmask


def eval_sim(statedict, config, **kwargs):
    max_read_depth = 200
    feats_per_read = 7
    logger.info(f"Found torch device: {DEVICE}")
    conf = load_train_conf(config)
    regions = bwasim.load_regions(conf['regions'])
    in_dim = (max_read_depth ) * feats_per_read

    altpredictor = AltPredictor(0, 7)
    altpredictor.load_state_dict(torch.load("altpredictor3.sd"))
    altpredictor.to(DEVICE)

    model = VarTransformer(in_dim=in_dim, out_dim=4, nhead=6, d_hid=300, n_encoder_layers=2).to(DEVICE)
    model.load_state_dict(torch.load(statedict))
    model.eval()

    batch_size = 100
    for varfunc in [bwasim.make_het_del, bwasim.make_het_ins, bwasim.make_het_snv, bwasim.make_mnv, bwasim.make_novar]:
        label = str(varfunc.__name__).split("_")[-1]
        for vaf in [0.99, 0.50, 0.25, 0.10, 0.05]:
            src, tgt, vaftgt, altmask = bwasim.make_batch(batch_size,
                                                  regions,
                                                  conf['reference'],
                                                  numreads=200,
                                                  readlength=140,
                                                  var_funcs=[varfunc],
                                                  vaf_func=lambda : vaf,
                                                  error_rate=0.01,
                                                  clip_prob=0)

            fullmask = create_altmask(altpredictor, src)
            masked_src = src.to(DEVICE) * fullmask
            seq_preds, vaf_preds = model(masked_src)

            tp_total = 0
            fp_total = 0
            fn_total = 0
            for b in range(src.shape[0]):
                tps, fps, fns = eval_prediction(util.readstr(src[b, :, 0, :]),  util.tgt_str(tgt[b, :]), seq_preds[b, :, :], midwidth=50)
                tp_total += len(tps)
                fp_total += len(fps)
                fn_total += len(fns)

            print(f"{label}, {vaf:.3f}, {tp_total}, {fp_total}, {fn_total}")
            # print(f"VAF: {vaf} PPA: {tp_total / (tp_total + fn_total):.3f} PPV: {tp_total / (tp_total + fp_total):.3f}")


def pregen_one_sample(dataloader, batch_size, output_dir):
    """
    Pregenerate tensors for a single sample
    """
    uid = "".join(random.choices(ascii_letters + digits, k=8))
    src_prefix = "src"
    tgt_prefix = "tgt"
    vaf_prefix = "vaftgt"

    logger.info(f"Saving tensors to {output_dir}/")
    for i, (src, tgt, vaftgt, _) in enumerate(dataloader.iter_once(batch_size)):
        logger.info(f"Saving batch {i} with uid {uid}")
        for data, prefix in zip([src, tgt, vaftgt],
                                [src_prefix, tgt_prefix, vaf_prefix]):
            with lz4.frame.open(output_dir / f"{prefix}_{uid}-{i}.pt.lz4", "wb") as fh:
                torch.save(data, fh)

def pregen(config, **kwargs):
    """
    Pre-generate tensors from BAM files + labels and save them in 'datadir' for quicker use in training
    (this takes a long time)
    """
    conf = load_train_conf(config)
    batch_size = kwargs.get('batch_size', 64)
    reads_per_pileup = kwargs.get('read_depth', 300)
    samples_per_pos = kwargs.get('samples_per_pos', 10)
    output_dir = Path(kwargs.get('dir'))
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
        logger.info(f"Generating training data using config from {config}")
        dataloaders = [
            loader.LazyLoader([(c['bam'], c['labels'])], conf['reference'], reads_per_pileup, samples_per_pos)
            for c in conf['data']
        ]

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Submitting {len(dataloaders)} jobs with {processes} process(es)")
    if processes == 1:
        for dl in dataloaders:
            pregen_one_sample(dl, batch_size, output_dir)
    else:
        futures = []
        with ProcessPoolExecutor(max_workers=processes) as executor:
            for dl in dataloaders:
                futures.append(executor.submit(pregen_one_sample, dl, batch_size, output_dir))
            for fut in futures:
                fut.result()


    # output_dir = Path(kwargs.get('dir'))
    # output_dir.mkdir(parents=True, exist_ok=True)
    # src_prefix = "src"
    # tgt_prefix = "tgt"
    # vaf_prefix = "vaftgt"
    # existing_tgt = list(output_dir.glob(tgt_prefix + "*"))
    # startval = kwargs.get('start_from', 0)
    # if existing_tgt:
    #     idxs = [int(t.name.replace(".pt", "").split("_")[-1]) for t in existing_tgt]
    #     startval = max(max(idxs), startval)
    #     logger.info(f"Found existing data in directory {output_dir}, starting indexes at {startval}")
    # logger.info(f"Saving tensors to {output_dir}/..")
    # for i, (src, tgt, vaftgt, _) in enumerate(dataloader.iter_once(batch_size), start=startval):
    #     logger.info(f"Saving batch {i}")
    #     torch.save(src, output_dir / f"{src_prefix}_{i}.pt")
    #     torch.save(tgt, output_dir / f"{tgt_prefix}_{i}.pt")
    #     torch.save(vaftgt, output_dir / f"{vaf_prefix}_{i}.pt")


def callvars(altpredictor, model, aln, reference, chrom, pos, max_read_depth):
    """
    Call variants in a region of a BAM file using the given altpredictor and model
    and return a list of vcf.Variant objects
    """
    reads = reads_spanning(aln, chrom, pos, max_reads=max_read_depth)
    if len(reads) < 5:
        raise ValueError(f"Hmm, couldn't find any reads spanning {chrom}:{pos}")


    minref = min(alnstart(r) for r in reads)
    maxref = max(alnstart(r) + r.query_length for r in reads)
    reads_encoded, _ = encode_pileup3(reads, minref, maxref)


    refseq = reference.fetch(chrom, minref, minref + reads_encoded.shape[0])
    reftensor = string_to_tensor(refseq)
    reads_w_ref = torch.cat((reftensor.unsqueeze(1), reads_encoded), dim=1)
    padded_reads = ensure_dim(reads_w_ref, maxref - minref, max_read_depth).unsqueeze(0).to(DEVICE)


    #masked_reads = padded_reads * fullmask
    seq_preds, _ = model(padded_reads.to(DEVICE))
    pred1str = util.readstr(seq_preds[0, :, :])

    variants = [v for v in vcf.aln_to_vars(refseq,
                             pred1str,
                             offset=minref)]
    return variants, seq_preds.squeeze(0)


def eval_labeled_bam(config, bam, labels, statedict, **kwargs):
    """
    Call variants in BAM file with given model at positions given in the labels CSV, emit useful
    summary information about PPA / PPV, etc
    """
    max_read_depth = 300
    feats_per_read = 8
    logger.info(f"Found torch device: {DEVICE}")
    conf = load_train_conf(config)

    reference = pysam.FastaFile(conf['reference'])

    altpredictor = None

    model = VarTransformerAltMask(read_depth=max_read_depth, feature_count=feats_per_read, out_dim=4, nhead=6, d_hid=300, n_encoder_layers=2, device=DEVICE).to(DEVICE)
    model.load_state_dict(torch.load(statedict))
    model.eval()

    aln = pysam.AlignmentFile(bam)
    results = defaultdict(Counter)
    for i, row in pd.read_csv(labels).iterrows():
        if row.ngs_vaf < 0.05:
            vafgroup = "< 0.05"
        elif 0.05 <= row.ngs_vaf < 0.15:
            vafgroup = "0.05 - 0.15"
        else:
            vafgroup = "> 0.15"

        labeltype = f"{row.vtype}-{row.status} {vafgroup}"
        if results[labeltype]['rows'] > 100:
            continue
        try:
            variants, seq_preds = callvars(altpredictor, model, aln, reference, str(row.chrom), int(row.pos), max_read_depth=max_read_depth)
        except Exception as ex:
            logger.warning(f"Hmm, exception processing {row.chrom}:{row.pos}, skipping it")
            logger.warning(ex)
            continue
        refwidth = seq_preds.shape[0]
        refseq = reference.fetch(str(row.chrom), int(row.pos) - refwidth//2, int(row.pos) + refwidth//2)
        if row.status == 'TP' or row.status == 'FN':
            true_altseq = refseq[0:refwidth//2 - 1] + row.alt + refseq[refwidth//2 + len(row.ref) - 1:]
        else:
            true_altseq = refseq

        tps, fps, fns = eval_prediction(refseq, true_altseq, seq_preds,
                                        midwidth=150)
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
    genparser.set_defaults(func=pregen)

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
    trainparser.set_defaults(func=train)

    callparser = subparser.add_parser("call", help="Call variants")
    callparser.add_argument("-m", "--statedict", help="Stored model", required=True)
    callparser.add_argument("-r", "--reference", help="Path to Fasta reference genome", required=True)
    callparser.add_argument("-b", "--bam", help="Input BAM file", required=True)
    callparser.add_argument("--chrom", help="Chromosome", required=True)
    callparser.add_argument("--pos", help="Position", required=True, type=int)
    callparser.set_defaults(func=call)

    evalparser = subparser.add_parser("eval", help="Evaluate a model on some known or simulated data")
    evalparser.add_argument("-m", "--statedict", help="Stored model", required=True)
    evalparser.add_argument("-c", "--config", help="Training configuration yaml", required=True)
    evalparser.set_defaults(func=eval_sim)

    args = parser.parse_args()
    args.func(**vars(args))


if __name__ == "__main__":
    main()
