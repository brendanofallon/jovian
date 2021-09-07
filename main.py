#!/usr/bin/env python


import logging
from datetime import datetime
from collections import defaultdict

import pysam
import torch
from torch import nn
import torch.multiprocessing as mp
import numpy as np
import argparse
import yaml

import bwasim
import model
import sim
import util
import vcf
import loader
from bam import target_string_to_tensor, encode_pileup2, reads_spanning, alnstart
from model import VarTransformer, AltPredictor

logging.basicConfig(format='[%(asctime)s]  %(name)s  %(levelname)s  %(message)s',
                    datefmt='%m-%d %H:%M:%S',
                    level=logging.INFO) # handlers=[RichHandler()])
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda:0") if hasattr(torch, 'cuda') and torch.cuda.is_available() else torch.device("cpu")




def trim_pileuptensor(src, tgt, width):
    """
    Trim or zero-pad the sequence dimension of src and target (first dimension of src, second of target)
    so they're equal to width
    :param src: Data tensor of shape [sequence, read, features]
    :param tgt: Target tensor of shape [haplotype, sequence]
    :param width: Size to trim to
    """
    assert src.shape[0] == tgt.shape[1], f"Unequal src and target lengths ({src.shape[0]} vs. {tgt.shape[1]}), not sure how to deal with this :("
    if src.shape[0] < width:
        z = torch.zeros(width - src.shape[0], src.shape[1], src.shape[2])
        src = torch.cat((src, z))
        t = torch.zeros(tgt.shape[0], width - tgt.shape[1])
        tgt = torch.cat((tgt, t), dim=1)
    else:
        start = src.shape[0] // 2 - width // 2
        src = src[start:start+width, :, :]
        tgt = tgt[:, start:start+width]

    return src, tgt


def make_loader(bampath, refpath, csv, max_to_load=1e9, max_reads_per_aln=100):
    allsrc = []
    alltgt = []
    count = 0
    seq_len = 150
    logger.info(f"Creating new data loader from {bampath}")
    counter = defaultdict(int)
    classes = []
    for enc, tgt, status, vtype in loader.load_from_csv(bampath, refpath, csv, max_reads_per_aln=max_reads_per_aln):
        label_class = "-".join((status, vtype))
        classes.append(label_class)
        counter[label_class] += 1
        src, tgt = trim_pileuptensor(enc, tgt, seq_len)
        assert src.shape[0] == seq_len, f"Src tensor #{count} had incorrect shape after trimming, found {src.shape[0]} but should be {seq_len}"
        assert tgt.shape[1] == seq_len, f"Tgt tensor #{count} had incorrect shape after trimming, found {tgt.shape[1]} but should be {seq_len}"
        allsrc.append(src)
        alltgt.append(tgt)
        count += 1
        if count % 100 == 0:
            logger.info(f"Loaded {count} tensors from {csv}")
        if count == max_to_load:
            logger.info(f"Stopping tensor load after {max_to_load}")
            break
    logger.info(f"Loaded {count} tensors from {csv}")
    logger.info("Class breakdown is: " + " ".join(f"{k}={v}" for k,v in counter.items()))
    weights = np.array([1.0 / counter[c] for c in classes])
    return loader.WeightedLoader(torch.stack(allsrc), torch.stack(alltgt).long(), weights, DEVICE)


def make_multiloader(inputs, refpath, threads, max_to_load, max_reads_per_aln):
    """
    Create multiple ReadLoaders in parallel for each element in Inputs
    :param inputs: List of (BAM path, labels csv) tuples
    :param threads: Number of threads to use
    :param max_reads_per_aln: Max number of reads for each pileup
    :return: List of loaders
    """
    results = []
    logger.info(f"Loading training data for {len(inputs)} samples with {threads} processes (max to load = {max_to_load})")
    with mp.Pool(processes=threads) as pool:
        for bam, labels_csv in inputs:
            result = pool.apply_async(make_loader, (bam, refpath, labels_csv, max_to_load, max_reads_per_aln))
            results.append(result)
        pool.close()
        return loader.MultiLoader([l.get(timeout=2*60*60) for l in results])


def train_epoch(model, optimizer, criterion, vaf_criterion, loader, batch_size, max_alt_reads, altpredictor=None):
    """
    Train for one epoch, which is defined by the loader but usually involves one pass over all input samples
    :param model: Model to train
    :param optimizer: Optimizer to update params
    :param criterion: Loss function
    :param loader: Provides training data
    :param batch_size:
    :return: Sum of losses over each batch, plus fraction of matching bases for ref and alt seq
    """
    epoch_loss_sum = 0
    vafloss_sum = 0
    for unsorted_src, tgt_seq, tgtvaf, altmask in loader.iter_once(batch_size):
        predicted_altmask = altpredictor(unsorted_src)
        predicted_altmask = torch.cat((torch.ones(unsorted_src.shape[0], 1).to(DEVICE), predicted_altmask[:, 1:]), dim=1)
        aex = predicted_altmask.unsqueeze(-1).unsqueeze(-1)
        fullmask = aex.expand(unsorted_src.shape[0], unsorted_src.shape[2], unsorted_src.shape[1], unsorted_src.shape[3]).transpose(1, 2)
        src = unsorted_src * fullmask

        optimizer.zero_grad()

        seq_preds, vaf_preds = model(src)

        focalwidth = 100
        focalstart = seq_preds.shape[1] // 2 - focalwidth // 2
        focalend = seq_preds.shape[1] // 2 + focalwidth // 2

        loss = criterion(seq_preds[:, focalstart:focalend, :].flatten(start_dim=0, end_dim=1), tgt_seq[:, focalstart:focalend].flatten())

        # vafloss = vaf_criterion(vaf_preds.double().squeeze(1), tgtvaf.double())
        with torch.no_grad():
            width = 20
            mid = seq_preds.shape[1] // 2
            midmatch = (torch.argmax(seq_preds[:, mid-width//2:mid+width//2, :].flatten(start_dim=0, end_dim=1),
                                     dim=1) == tgt_seq[:, mid-width//2:mid+width//2].flatten()
                         ).float().mean()



        loss.backward(retain_graph=True)
        # vafloss.backward()
        optimizer.step()
        epoch_loss_sum += loss.detach().item()
        # vafloss_sum += vafloss.detach().item()

    return epoch_loss_sum, midmatch.item(), vafloss_sum


def train_epochs(epochs,
                 dataloader,
                 max_read_depth=50,
                 feats_per_read=8,
                 init_learning_rate=0.001,
                 checkpoint_freq=0,
                 statedict=None,
                 model_dest=None,
                 eval_batches=None):
    in_dim = (max_read_depth) * feats_per_read
    model = VarTransformer(in_dim=in_dim, out_dim=4, nhead=6, d_hid=200, n_encoder_layers=2).to(DEVICE)
    logger.info(f"Creating model with {sum(p.numel() for p in model.parameters() if p.requires_grad)} params")
    if statedict is not None:
        logger.info(f"Initializing model with state dict {statedict}")
        model.load_state_dict(torch.load(statedict))
    model.train()
    batch_size = 128

    altpredictor = AltPredictor(0, 7)
    altpredictor.load_state_dict(torch.load("altpredictor.sd"))
    altpredictor.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    vaf_crit = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=init_learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.995)
    try:
        for epoch in range(epochs):
            starttime = datetime.now()
            loss, refmatch, vafloss = train_epoch(model,
                                                  optimizer,
                                                  criterion,
                                                  vaf_crit,
                                                  dataloader,
                                                  batch_size=batch_size,
                                                  max_alt_reads=max_read_depth,
                                                  altpredictor=altpredictor)
            elapsed = datetime.now() - starttime

            logger.info(f"Epoch {epoch} Secs: {elapsed.total_seconds():.2f} lr: {scheduler.get_last_lr()[0]:.4f} loss: {loss:.4f} Ref match: {refmatch:.4f}  vafloss: {vafloss:.4f} ")
            scheduler.step()


            if epoch > 0 and checkpoint_freq > 0 and (epoch % checkpoint_freq == 0):
                modelparts = str(model_dest).rsplit(".", maxsplit=1)
                checkpoint_name = modelparts[0] + f"_epoch{epoch}" + modelparts[1]
                logger.info(f"Saving model state dict to {checkpoint_name}")
                torch.save(model.to('cpu').state_dict(), checkpoint_name)


            if eval_batches is not None:
                for vartype, (src, tgt, vaftgt, altmask) in eval_batches.items():
                    # Use 'altpredictor' to mask out non-alt reads
                    src = src.to(DEVICE)
                    predicted_altmask = altpredictor(src.to(DEVICE))
                    predicted_altmask = torch.cat((torch.ones(src.shape[0], 1).to(DEVICE), predicted_altmask[:, 1:]), dim=1)
                    aex = predicted_altmask.unsqueeze(-1).unsqueeze(-1)
                    fullmask = aex.expand(src.shape[0], src.shape[2], src.shape[1],
                                          src.shape[3]).transpose(1, 2).to(DEVICE)
                    src = src * fullmask

                    with torch.no_grad():
                        altpreds = altpredictor(src.to(DEVICE))
                        minalt = altpreds.min().item()
                        maxalt = altpreds.max().item()

                    predictions, vafpreds = model(src.to(DEVICE))
                    tps, fps, fns = eval_batch(src, tgt, predictions)
                    logger.info(f"Eval: Min alt mask: {minalt:.3f} max: {maxalt:.3f}")
                    if tps > 0:
                        logger.info(f"Eval: {vartype} PPA: {(tps / (tps + fns)):.3f} PPV: {(tps / (tps + fps)):.3f}")
                    else:
                        logger.info(f"Eval: {vartype} PPA: No TPs found :(")


        logger.info(f"Training completed after {epoch} epochs")
    except KeyboardInterrupt:
        pass

    if model_dest is not None:
        logger.info(f"Saving model state dict to {model_dest}")
        torch.save(model.to('cpu').state_dict(), model_dest)


def load_train_conf(confyaml):
    logger.info(f"Loading configuration from {confyaml}")
    conf = yaml.safe_load(open(confyaml).read())
    assert 'reference' in conf, "Expected 'reference' entry in training configuration"
    assert 'data' in conf, "Expected 'data' entry in training configuration"
    return conf


def train(config, output_model, input_model, epochs, max_to_load, **kwargs):
    logger.info(f"Found torch device: {DEVICE}")
    conf = load_train_conf(config)
    train_sets = [(c['bam'], c['labels']) for c in conf['data']]
    #dataloader = make_multiloader(train_sets, conf['reference'], threads=6, max_to_load=max_to_load, max_reads_per_aln=200)
    # dataloader = loader.SimLoader(DEVICE, seqlen=100, readsperbatch=100, readlength=80, error_rate=0.01, clip_prob=0.01)
    eval_batches = create_eval_batches(25, conf)
    dataloader = loader.BWASimLoader(DEVICE,
                                     regions=conf['regions'],
                                     refpath=conf['reference'],
                                     readsperpileup=100,
                                     readlength=145,
                                     error_rate=0.01,
                                     clip_prob=0)
    train_epochs(epochs,
                 dataloader,
                 max_read_depth=100,
                 feats_per_read=7,
                 statedict=input_model,
                 model_dest=output_model,
                 eval_batches=eval_batches)


def call(statedict, bam, reference, chrom, pos, **kwargs):
    max_read_depth = 200
    feats_per_read = 8
    in_dim = max_read_depth * feats_per_read
    model = VarTransformer(in_dim=in_dim, out_dim=4, nhead=5, d_hid=200, n_encoder_layers=2).to(DEVICE)
    model.load_state_dict(torch.load(statedict))
    model.eval()

    bam = pysam.AlignmentFile(bam)
    reads = reads_spanning(bam, chrom, pos, max_reads=max_read_depth)
    if len(reads) < 5:
        raise ValueError(f"Hmm, couldn't find any reads spanning {chrom}:{pos}")

    reads_encoded = encode_pileup2(reads).to(DEVICE)
    minref = min(alnstart(r) for r in reads)
    refseq = pysam.FastaFile(reference).fetch(chrom, minref, minref + reads_encoded.shape[0])
    reft = target_string_to_tensor(refseq).unsqueeze(0).unsqueeze(0).to(DEVICE)
    src = sort_by_ref(reft, reads_encoded.unsqueeze(0))
    
    print(util.to_pileup(src[0, :, :, :]))
    seq1preds, seq2preds = model(src.flatten(start_dim=2))
    pred1str = util.readstr(seq1preds[0, :, :])
    pred2str = util.readstr(seq2preds[0, :, :])
    print("\n")
    print(refseq)
    print(pred1str)
    print(pred2str)
    print("".join('*' if a==b else 'x' for a,b in zip(refseq, pred2str)))
    print(refseq)
    midwith = 100
    for v in vcf.aln_to_vars(refseq[len(refseq) // 2 - midwith // 2:len(refseq) // 2 + midwith // 2],
                            pred2str[len(refseq)//2 - midwith//2:len(refseq)//2 + midwith//2],
                             offset=minref + len(refseq)//2 - midwith//2 + 1):
        print(v)


def eval_prediction(refseq, tgt, predictions, midwidth=100):
    """
    Given a target sequence and two predicted sequences, attempt to determine if the correct *Variants* are
    detected from the target. This uses the vcf.align_seqs(seq1, seq2) method to convert string sequences to Variant
    objects, then compares variant objects
    :param tgt:
    :param predictions:
    :param midwidth:
    :return: Sets of TP, FP, and FN vars
    """
    rseq1 = util.tgt_str(tgt)
    midstart = tgt.shape[-1] // 2 - midwidth // 2
    midend = tgt.shape[-1] // 2 + midwidth // 2
    refseqstr = util.readstr(refseq)
    known_vars = []

    # print("Ref / target alignment:")
    # aln = vcf.align_sequences(refseqstr[midstart:midend], rseq1[midstart:midend])
    # print(aln.aligned_query_sequence)
    # print(aln.aligned_target_sequence)
    for v in vcf.aln_to_vars(refseqstr[midstart:midend], rseq1[midstart:midend]):
        known_vars.append(v)

    # print("Ref / predictions alignment:")
    # aln = vcf.align_sequences(refseqstr[midstart:midend], util.readstr(predictions[midstart:midend, :]))
    # print(aln.aligned_query_sequence)
    # print(aln.aligned_target_sequence)
    pred_vars = []
    for v in vcf.aln_to_vars(refseqstr[midstart:midend], util.readstr(predictions[midstart:midend, :])):
        v.qual = predictions[v.pos:v.pos + max(1, min(len(v.ref), len(v.alt))), :].max(dim=1)[0].min().item()
        pred_vars.append(v)

    tps = [] # True postive - real and detected variant
    fns = [] # False negatives - real variant but not detected
    fps = [] # False positives - detected but not a real variant
    for true_var in known_vars:
        if true_var in pred_vars:
            logger.info(f"TP: {v}")
            tps.append(true_var)
        else:
            fns.append(true_var)

    for detected_var in pred_vars:
        if detected_var not in known_vars:
            logger.info(f"FP: {detected_var}")
            fps.append(detected_var)

    return tps, fps, fns


def eval_batch(src, tgt, predictions):
    """
    Run evaluation on a single batch
    :param src: Model input (with batch dimension as first dim and ref sequence as first element in dimension 2)
    :param tgt: Model targets / true alt sequence
    :param predictions: Model prediction
    :return: Total number of TP, FP, and FN variants
    """
    tp_total = 0
    fp_total = 0
    fn_total = 0
    for b in range(src.shape[0]):
        refseq = src[b, :, 0, :]
        assert refseq[:, 0:4].sum() == refseq.shape[0], f"Probable incorrect refseq index, sum did not match sequence length!"
        tps, fps, fns = eval_prediction(refseq, tgt[b, :], predictions[b, :, :])
        tp_total += len(tps)
        fp_total += len(fps)
        fn_total += len(fns)
    return tp_total, fp_total, fn_total


def create_eval_batches(batch_size, config):
    """
    Create batches of variants for evaluation
    :param batch_size: Number of pileups per batch
    :param config: Config
    :return: Mapping from variant type -> (batch src, tgt, vaftgt, altmask)
    """
    num_reads = 100
    read_length = 146
    base_error_rate = 0.01
    if type(config) == str:
        conf = load_train_conf(config)
    else:
        conf = config
    regions = bwasim.load_regions(conf['regions'])
    eval_batches = {}
    logger.info(f"Generating evaluation batches of size {batch_size}")
    vaffunc = loader.betavaf
    eval_batches['del'] = bwasim.make_batch(batch_size,
                                                      regions,
                                                      conf['reference'],
                                                      numreads=num_reads,
                                                      readlength=read_length,
                                                      var_funcs=[bwasim.make_het_del],
                                                      vaf_func=vaffunc,
                                                      error_rate=base_error_rate,
                                                      clip_prob=0)
    eval_batches['ins'] = bwasim.make_batch(batch_size,
                                                regions,
                                                conf['reference'],
                                                numreads=num_reads,
                                                readlength=read_length,
                                                vaf_func=vaffunc,
                                                var_funcs=[bwasim.make_het_ins],
                                                error_rate=base_error_rate,
                                                clip_prob=0)
    eval_batches['snv'] = bwasim.make_batch(batch_size,
                                                regions,
                                                conf['reference'],
                                                numreads=num_reads,
                                                readlength=read_length,
                                                vaf_func=vaffunc,
                                                var_funcs=[bwasim.make_het_snv],
                                                error_rate=base_error_rate,
                                                clip_prob=0)
    eval_batches['mnv'] = bwasim.make_batch(batch_size,
                                                regions,
                                                conf['reference'],
                                                numreads=num_reads,
                                                readlength=read_length,
                                                vaf_func=vaffunc,
                                                var_funcs=[bwasim.make_het_del],
                                                error_rate=base_error_rate,
                                                clip_prob=0)
    logger.info("Done generating evaluation batches")
    return eval_batches

def eval_sim(statedict, config, **kwargs):
    max_read_depth = 100
    feats_per_read = 7
    logger.info(f"Found torch device: {DEVICE}")
    conf = load_train_conf(config)
    regions = bwasim.load_regions(conf['regions'])
    in_dim = (max_read_depth ) * feats_per_read

    altpredictor = AltPredictor(0, 7)
    altpredictor.load_state_dict(torch.load("altpredictor2.sd"))
    altpredictor.to(DEVICE)

    model = VarTransformer(in_dim=in_dim, out_dim=4, nhead=6, d_hid=200, n_encoder_layers=2).to(DEVICE)
    model.load_state_dict(torch.load(statedict))
    model.eval()

    batch_size = 50
    for varfunc in [bwasim.make_het_del, bwasim.make_het_ins, bwasim.make_het_snv, bwasim.make_mnv]:
        label = str(varfunc.__name__).split("_")[-1]
        for vaf in [0.99, 0.50, 0.25, 0.10, 0.05]:
            src, tgt, vaftgt, altmask = bwasim.make_batch(batch_size,
                                                  regions,
                                                  conf['reference'],
                                                  numreads=100,
                                                  readlength=100,
                                                  var_funcs=[varfunc],
                                                  vaf_func=lambda : vaf,
                                                  error_rate=0.01,
                                                  clip_prob=0)

        predicted_altmask = altpredictor(src.to(DEVICE))
        predicted_altmask = torch.cat((torch.ones(src.shape[0], 1).to(DEVICE), predicted_altmask[:, 1:]), dim=1)
        aex = predicted_altmask.unsqueeze(-1).unsqueeze(-1)
        fullmask = aex.expand(src.shape[0], src.shape[2], src.shape[1],
                          src.shape[3]).transpose(1, 2).to(DEVICE)
        masked_src = src * fullmask
        seq_preds, vaf_preds = model(masked_src)

        tp_total = 0
        fp_total = 0
        fn_total = 0
        for b in range(src.shape[0]):
            # print(util.to_pileup(src[b, :, :, :], altmask[b, :]))
            # print(util.readstr(src[b, :, 0, :]))
            # print(util.readstr(seq_preds[b, :, :]))
            tps, fps, fns = eval_prediction(src[b, :, 0, :], tgt[b, :], seq_preds[b, :, :], midwidth=50)
            tp_total += len(tps)
            fp_total += len(fps)
            fn_total += len(fns)
            # print(f"\tItem {b} TP: {len(tps)} FP: {len(fps)}  FN: {len(fns)}")

        print(f"{label}, {vaf:.3f}, {tp_total}, {fp_total}, {fn_total}")
        # print(f"VAF: {vaf} PPA: {tp_total / (tp_total + fn_total):.3f} PPV: {tp_total / (tp_total + fp_total):.3f}")


def main():
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers()
    trainparser = subparser.add_parser("train", help="Train a model")
    trainparser.add_argument("-n", "--epochs", type=int, help="Number of epochs to train for", default=100)
    trainparser.add_argument("-i", "--input-model", help="Start with parameters from given state dict")
    trainparser.add_argument("-o", "--output-model", help="Save trained state dict here", required=True)
    trainparser.add_argument("-ch", "--checkpoint-freq", help="Save model checkpoints frequency (0 to disable)", default=0, type=int)
    trainparser.add_argument("-m", "--max-to-load", help="Max number of input tensors to load", type=int, default=1e9)
    trainparser.add_argument("-c", "--config", help="Training configuration yaml", required=True)
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

# def testaltpredictor():
#     conf = load_train_conf("simtrain.yaml")
#     regions = bwasim.load_regions(conf['regions'])
#     ap = model.AltPredictor(-1, 7)
#     ap.load_state_dict(torch.load("altpredictor.sd"))
#     src, tgt, vaftgt, altmask = bwasim.make_batch(10,
#                                                   regions,
#                                                   conf['reference'],
#                                                   numreads=100,
#                                                   readlength=55,
#                                                   var_funcs=[bwasim.make_het_snv],
#                                                   error_rate=0.01,
#                                                   clip_prob=0)
#
#     preds = ap(src)
#     for b in range(src.shape[0]):
#         print(f"\n Batch {b}")
#         for pred, actual in zip(preds[b, :], altmask[b, :]):
#             print(f"{pred:.3f}\t{actual:.3f}")

if __name__ == "__main__":
    # testaltpredictor()
    main()
