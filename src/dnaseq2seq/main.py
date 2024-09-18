#!/usr/bin/env python

import sys
import logging
import os

import re
import torch
import sklearn

import argparse

from dnaseq2seq import util as util
from dnaseq2seq import loader as loader
from dnaseq2seq import __version__ as VERSION
from dnaseq2seq import util as util

logging.basicConfig(format='[%(asctime)s] %(process)d  %(name)s  %(levelname)s  %(message)s',
                    datefmt='%m-%d %H:%M:%S',
                    level=os.environ.get('JV_LOGLEVEL', logging.INFO),
                    handlers=[
                        logging.StreamHandler(),  # Output logs to stdout
                    ])

logger = logging.getLogger(__name__)

def do_pregen(*args, **kwargs):
    from dnaseq2seq.pregen import pregen
    pregen(*args, **kwargs)

def do_call(*args, **kwargs):
    from dnaseq2seq.call import call
    call(*args, **kwargs)

def do_train(*args, **kwargs):
    from dnaseq2seq.train import train
    train(*args, **kwargs)


def alphanumeric_no_spaces(name):
    if re.match(r"[a-zA-Z0-9_-]+", name):
        return name
    else:
        raise argparse.ArgumentTypeError(f"{name} is not an alphanumeric plus '_' or '-' without spaces")

def main():
    logger.debug("Turning on DEBUG log level")
    logger.info("ONT branch")
    logger.info(f"Jenever version {VERSION}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Scikit-learn version: {sklearn.__version__}")
    parser = argparse.ArgumentParser(description='NGS variant detection with transformers')
    subparser = parser.add_subparsers()

    genparser = subparser.add_parser("pregen", help="Pre-generate tensors from BAMs")
    genparser.add_argument("-c", "--config", help="Configuration yaml", required=True)
    genparser.add_argument("-d", "--dir", help="Output directory", default=".")
    genparser.add_argument("-rd", "--read-depth", help="Max read depth / tensor dim", default=128, type=int)
    genparser.add_argument("-s", "--sim", help="Generate simulated data", action='store_true')
    genparser.add_argument("-b", "--batch-size", help="Number of pileups to include in a single file (basically the batch size)", default=64, type=int)
    genparser.add_argument("-n", "--start-from", help="Start numbering from here", type=int, default=0)
    genparser.add_argument("-t", "--threads", help="Number of processes to use", type=int, default=1)
    genparser.add_argument("-j", "--jitter", help="Jitter each region by at most this amount (default 0 is no jitter)", type=int, default=0)
    # genparser.add_argument("-vpc", "--vals-per-class", help="The number of instances for each variant class in a label file; it will be set automatically if not specified", type=int, default=1000)
    genparser.add_argument("-mf", "--metadata-file", help="The metadata file that records each row in the encoded tensor files and the variant from which that row is derived. The name pregen_{time}.csv will be used if not specified.")
    genparser.set_defaults(func=do_pregen)

    printpileupparser = subparser.add_parser("print", help="Print a tensor pileup")
    printpileupparser.add_argument("-p", "--path", help="Path to saved tensor data", required=True)
    printpileupparser.add_argument("-i", "--idx", help="Index of item in batch to emit", required=True, type=int)
    printpileupparser.set_defaults(func=util.print_pileup)

    trainparser = subparser.add_parser("train", help="Train a model")
    trainparser.add_argument("-c", "--config", help="Configuration yaml", required=False)
    trainparser.add_argument("-n", "--epochs", type=int, help="Number of epochs to train for", default=None)
    trainparser.add_argument("-i", "--input-model", help="Start with parameters from given state dict")
    trainparser.add_argument("-o", "--output-model", help="Save trained state dict here", required=True)
    trainparser.add_argument("-ch", "--checkpoint-freq", help="Save model checkpoints frequency (0 to disable)", type=int)
    trainparser.add_argument("-lr", "--learning-rate", help="Initial learning rate", default=None, type=float)
    trainparser.add_argument("-s", "--samples-per-epoch", help="Number of samples to process before emitting stats", type=int, default=None)
    trainparser.add_argument("-d", "--datadir", help="Pregenerated data dir", default=None)
    trainparser.add_argument("-vd", "--val-dir", help="Pregenerated data for validation", default=None)
    trainparser.add_argument("-t", "--threads", help="Max number of threads to use for decompression (torch may use more)", default=None, type=int)
    trainparser.add_argument("-md", "--max-decomp-batches",
                             help="Max number batches to decompress and store in memory at once", default=4, type=int)
    trainparser.add_argument("-b", "--batch-size", help="The batch size, default is 64", type=int, default=None)
    trainparser.add_argument("-rn", "--run-name", type=alphanumeric_no_spaces, default=None,
                             help="Run name, must be alphanumeric plus '_' or '-'")
    trainparser.add_argument("--notes", type=str, default=None,
                             help="Run notes, longer description of run (like 'git commit -m')")
    trainparser.set_defaults(func=do_train)

    callparser = subparser.add_parser("call", help="Call variants")
    callparser.add_argument("-m", "--model-path", help="Stored model", required=True)
    callparser.add_argument("-c", "--classifier-path", help="Stored variant classifier model", default=None, type=str)
    callparser.add_argument("-r", "--reference-fasta", help="Path to Fasta reference genome", required=True)
    callparser.add_argument("-b", "--bam", help="Input BAM file", required=True)
    callparser.add_argument("-d", "--bed", help="bed file defining regions to call", required=True)
    callparser.add_argument("-g", "--region", help="Region to call variants in, of form chr:start-end", required=False)
    callparser.add_argument("-f", "--freq-file", help="Population frequency file for classifier")
    callparser.add_argument("-v", "--vcf-out", help="Output vcf file", required=True)
    callparser.add_argument("-t", "--threads", help="Number of processes to use", type=int, default=1)
    callparser.add_argument("-td", "--temp-dir", help="Temporary data storage location", default=os.environ.get("JV_TMPDIR", "."))
    callparser.add_argument("-mx", "--max-batch-size", help="Max number of regions to process at once", type=int, default=128)
    callparser.add_argument("-np", "--no-progress", help="Turn off progress bars", action='store_true')


    callparser.set_defaults(func=do_call)

    args = parser.parse_args()
    if len(vars(args)) == 0 or not hasattr(args, 'func'):
        parser.print_help()
    else:
        args.cmdline = " ".join(sys.argv[1:])
        args.cl_args = vars(args).copy()  # command line copy for logging
        args.func(**vars(args))


if __name__ == "__main__":
    main()
