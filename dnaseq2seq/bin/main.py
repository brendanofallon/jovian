#!/usr/bin/env python

import sys
import logging
import random
from collections import defaultdict
from pathlib import Path
from string import ascii_letters, digits
import lz4.frame
import tempfile
from datetime import datetime
import re
import torch

from concurrent.futures.process import ProcessPoolExecutor
import argparse
import yaml

from dnaseq2seq import util
from dnaseq2seq import loader
from dnaseq2seq.train import train
from dnaseq2seq.call import call


logging.basicConfig(format='[%(asctime)s]  %(name)s  %(levelname)s  %(message)s',
                    datefmt='%m-%d %H:%M:%S',
                    level=logging.INFO) # handlers=[RichHandler()])
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda") if hasattr(torch, 'cuda') and torch.cuda.is_available() else torch.device("cpu")

def load_conf(confyaml):
    logger.info(f"Loading configuration from {confyaml}")
    conf = yaml.safe_load(open(confyaml).read())
    assert 'reference' in conf, "Expected 'reference' entry in training configuration"
    assert 'data' in conf, "Expected 'data' entry in training configuration"
    return conf



def default_vals_per_class():
    """
    Multiprocess will instantly deadlock if a lambda or any callable not defined on the top level of the module is given
    as the 'factory' argument to defaultdict - but we have to give it *some* callable that defines the behavior when the key
    is not present in the dictionary, so this returns the default "vals_per_class" if a class is encountered that is not 
    specified in the configuration file. I don't think there's an easy way to make this user-settable, unfortunately
    """
    return 0


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


def pregen(config, **kwargs):
    """
    Pre-generate tensors from BAM files + labels and save them in 'datadir' for quicker use in training
    (this takes a long time)
    """
    conf = load_conf(config)
    batch_size = kwargs.get('batch_size', 64)
    reads_per_pileup = kwargs.get('read_depth', 100)
    samples_per_pos = kwargs.get('samples_per_pos', 2)
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
    callparser.add_argument("-m", "--model-path", help="Stored model", required=True)
    callparser.add_argument("-c", "--classifier-path", help="Stored variant classifier model", default=None, type=str)
    callparser.add_argument("-r", "--reference-fasta", help="Path to Fasta reference genome", required=True)
    callparser.add_argument("-b", "--bam", help="Input BAM file", required=True)
    callparser.add_argument("-d", "--bed", help="bed file defining regions to call", required=False)
    callparser.add_argument("-g", "--region", help="Region to call variants in, of form chr:start-end", required=False)
    callparser.add_argument("-f", "--freq-file", help="Population frequency file for classifier")
    callparser.add_argument("-v", "--vcf-out", help="Output vcf file", required=True)
    callparser.add_argument("-t", "--threads", help="Number of processes to use", type=int, default=1)
    callparser.set_defaults(func=call)

    args = parser.parse_args()
    args.cmdline = " ".join(sys.argv[1:])
    args.cl_args = vars(args).copy()  # command line copy for logging
    if len(vars(args)) == 0 or getattr(args, 'func') is None:
        print(args.usage())
    else:
        args.func(**vars(args))


if __name__ == "__main__":
    main()
