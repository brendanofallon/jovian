
import concurrent.futures
import os
import time

import datetime
import logging
import string
import random
import itertools
from collections import defaultdict
from functools import partial
from pathlib import Path

import torch
import torch.multiprocessing as mp
import pysam
import numpy as np

from model import VarTransformer
import buildclf
import vcf
import util
import bam

logger = logging.getLogger(__name__)


DEVICE = torch.device("cuda") if hasattr(torch, 'cuda') and torch.cuda.is_available() else torch.device("cpu")

import warnings
warnings.filterwarnings(action='ignore')


def randchars(n=6):
    """ Generate a random string of letters and numbers """
    return ''.join(random.choices(string.ascii_letters + string.digits, k=n))


def gen_suspicious_spots(bamfile, chrom, start, stop, reference_fasta):
    """
    Generator for positions of a BAM / CRAM file that may contain a variant. This should be pretty sensitive and
    trigger on anything even remotely like a variant
    This uses the pysam Pileup 'engine', which seems less than ideal but at least it's C code and is probably
    fast. May need to update to something smarter if this
    :param bamfile: The alignment file in BAM format.
    :param chrom: Chromosome containing region
    :param start: Start position of region
    :param stop: End position of region (exclusive)
    :param reference_fasta: Reference sequences in fasta
    """
    aln = pysam.AlignmentFile(bamfile, reference_filename=reference_fasta)
    ref = pysam.FastaFile(reference_fasta)
    refseq = ref.fetch(chrom, start, stop)
    assert len(refseq) == stop - start, f"Ref sequence length doesn't match start - stop coords start: {chrom}:{start}-{stop}, ref len: {len(refseq)}"
    for col in aln.pileup(chrom, start=start, stop=stop, stepper='nofilter', multiple_iterators=False):
        # The pileup returned by pysam actually starts long before the first start position, but we only want to
        # report positions in the actual requested window
        if start <= col.reference_pos < stop:
            refbase = refseq[col.reference_pos - start]
            indel_count = 0
            base_mismatches = 0

            for i, read in enumerate(col.pileups):
                if read.indel != 0:
                    indel_count += 1

                if read.query_position is not None:
                    base = read.alignment.query_sequence[read.query_position]
                    if base != refbase:  # May want to check quality before adding a mismatch?
                        base_mismatches += 1

                if indel_count > 1 or base_mismatches > 2:
                    yield col.reference_pos
                    break



def load_model(model_path):
    
    #96M params
    # encoder_attention_heads = 8
    # decoder_attention_heads = 10
    # dim_feedforward = 512
    # encoder_layers = 10
    # decoder_layers = 10
    # embed_dim_factor = 160

    #50M params
    #encoder_attention_heads = 8
    #decoder_attention_heads = 4 
    #dim_feedforward = 512
    #encoder_layers = 8
    #decoder_layers = 6
    #embed_dim_factor = 120 

    #50M 'small decoder'
    #decoder_attention_heads = 4
    #decoder_layers = 2
    #dim_feedforward = 512
    #embed_dim_factor = 120
    #encoder_attention_heads = 10
    #encoder_layers = 10

    # 35M params
    #encoder_attention_heads = 8 # was 4
    #decoder_attention_heads = 4 # was 4
    #dim_feedforward = 512
    #encoder_layers = 6
    #decoder_layers = 4 # was 2
    #embed_dim_factor = 120 # was 100

    model_info = torch.load(model_path, map_location=DEVICE)
    statedict = model_info['model']
    modelconf = model_info['conf']
    new_state_dict = {}
    for key in statedict.keys():
      new_key = key.replace('_orig_mod.', '')
      new_state_dict[new_key] = statedict[key]
    statedict = new_state_dict

    #modelconf = {
    #        "max_read_depth": 150,
    #        "feats_per_read": 10,
    #        "decoder_layers": 10,
   #         "decoder_attention_heads": 10,
   #         "encoder_layers": 10,
   #         "encoder_attention_heads": 8,
   #         "dim_feedforward": 512,
   #         "embed_dim_factor": 160,
   #         }

    model = VarTransformer(read_depth=modelconf['max_read_depth'],
                           feature_count=modelconf['feats_per_read'],
                           kmer_dim=util.FEATURE_DIM,  # Number of possible kmers
                           n_encoder_layers=modelconf['encoder_layers'],
                           n_decoder_layers=modelconf['decoder_layers'],
                           embed_dim_factor=modelconf['embed_dim_factor'],
                           encoder_attention_heads=modelconf['encoder_attention_heads'],
                           decoder_attention_heads=modelconf['decoder_attention_heads'],
                           d_ff=modelconf['dim_feedforward'],
                           device=DEVICE)

    model.load_state_dict(statedict)

    #model.half()
    model.eval()
    model.to(DEVICE)
    
    model = torch.compile(model, fullgraph=True)
    
    return model


def read_bed_regions(bedpath):
    """
    Generate chrom, start, end regions from a BED formatted file
    """
    with open(bedpath) as fh:
        for line in fh:
            line = line.strip()
            if len(line) == 0 or line.startswith("#"):
                continue
            toks = line.split("\t")
            chrom, start, end = toks[0], int(toks[1]), int(toks[2])
            assert end > start, f"End position {end} must be strictly greater start {start}"
            yield chrom, start, end


def split_large_regions(regions, max_region_size):
    """
    Split any regions greater than max_region_size into regions smaller than max_region_size
    """
    for chrom, start, end in regions:
        while start < end:
            yield chrom, start, min(end, start + max_region_size)
            start += max_region_size


def cluster_positions(poslist, maxdist=100):
    """
    Iterate over the given list of positions (numbers), and generate ranges containing
    positions not greater than 'maxdist' in size
    """
    cluster = []
    end_pad_bases = 8
    for pos in poslist:
        if len(cluster) == 0 or pos - min(cluster) < maxdist:
            cluster.append(pos)
        else:
            yield min(cluster) - end_pad_bases, max(cluster) + end_pad_bases
            cluster = [pos]

    if len(cluster) == 1:
        yield cluster[0] - end_pad_bases, cluster[0] + end_pad_bases
    elif len(cluster) > 1:
        yield min(cluster) - end_pad_bases, max(cluster) + end_pad_bases


def cluster_positions_for_window(window, bamfile, reference_fasta, maxdist=100):
    """
    Generate a list of ranges containing a list of posistions from the given window
    returns: list of (chrom, index, start, end) tuples
    """
    chrom, window_idx, window_start, window_end = window
    
    cpname = mp.current_process().name
    logger.debug(
        f"{cpname}: Generating regions from window {window_idx}: "
        f"{window_start}-{window_end} on chromosome {chrom}"
    )
    return [
        (chrom, window_idx, start, end) 
        for start, end in cluster_positions(
            gen_suspicious_spots(bamfile, chrom, window_start, window_end, reference_fasta),
            maxdist=maxdist,
        )
    ]


def call(model_path, bam, bed, reference_fasta, vcf_out, classifier_path=None, **kwargs):
    """
    Use model in statedict to call variants in bam in genomic regions in bed file.
    Steps:
      1. build model
      2. break bed regions into windows with start positions determined by window_spacing and end positions
         determined by window_overlap (the last window in each bed region will likely be shorter than others)
      3. call variants in each window
      4. join variants after searching for any duplicates
      5. save to vcf file
    :param trans_model_path:
    :param bam:
    :param bed:
    :param reference_fasta:
    :param vcf_out:
    :param clf_model_path:
    """
    start_time = time.perf_counter()
    seed = 52342318
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    tmpdir_root = Path(kwargs.get("temp_dir"))
    threads = kwargs.get('threads', 1)
    max_batch_size = kwargs.get('max_batch_size', 64)
    logger.info(f"Using {threads} threads for encoding")
    logger.info(f"Found torch device: {DEVICE}")

    if 'cuda' in str(DEVICE):
        for idev in range(torch.cuda.device_count()):
            logger.info(f"CUDA device {idev} name: {torch.cuda.get_device_name({idev})}")

    var_freq_file = kwargs.get('freq_file')
    if var_freq_file:
        logger.info(f"Loading variant pop frequency file from {var_freq_file}")
    else:
        logger.info(f"No variant frequency file specified, this might not work depending on the classifier requirements")

    tmpdir = tmpdir_root / f"jv_tmpdata_{randchars()}"
    logger.info(f"Saving temp data in {tmpdir}")
    os.makedirs(tmpdir, exist_ok=False)

    logger.info(f"The model will be loaded from path {model_path}")

    vcf_header = vcf.create_vcf_header(sample_name="sample", lowcov=20, cmdline=kwargs.get('cmdline'))
    vcf_template = pysam.VariantFile("/dev/null", mode='w', header=vcf_header)
    vcf_out_fh = open(vcf_out, mode='w')
    logger.info(f"Writing variants to {Path(vcf_out).absolute()}")
    vcf_out_fh.write(str(vcf_header))

    call_vars_in_blocks(
        bamfile=bam,
        bed=bed,
        reference_fasta=reference_fasta,
        model_path=model_path,
        classifier_path=classifier_path,
        threads=threads,
        tmpdir=tmpdir,
        max_batch_size=max_batch_size,
        var_freq_file=var_freq_file,
        vcf_out=vcf_out_fh,
        vcf_template=vcf_template,
    )

    vcf_out_fh.close()
    logger.info(f"All variants saved to {vcf_out}")
    tmpdir.rmdir()
    end_time = time.perf_counter()
    elapsed_seconds = end_time - start_time
    if elapsed_seconds > 3600:
        logger.info(f"Total running time of call subcommand is: {elapsed_seconds / 3600 :.2f} hours")
    elif elapsed_seconds > 120:
        logger.info(f"Total running time of call subcommand is: {elapsed_seconds / 60 :.2f} minutes")
    else:
        logger.info(f"Total running time of call subcommand is: {elapsed_seconds :.2f} seconds")


def call_vars_in_blocks(
    bamfile, bed, reference_fasta, model_path, classifier_path, threads, tmpdir, max_batch_size, var_freq_file, vcf_out, vcf_template,
):
    """
    Split the input BED file into chunks and call variants in each chunk sequentially
    :param chrom: the chromosome name
    :param bamfile: the alignment file
    :param bed: the BED file for the given chromosome
    :param reference_fasta: the reference file
    :param model_path: the path to model
    :param classifier_path: the path to classifier
    :param threads: the number of CPUs to use
    :param tmpdir: a temporary directory

    :return: a VCF file with called variants for the given chromosome.
    """
    max_read_depth = 150
    logger.info(f"Max read depth: {max_read_depth}")
    logger.info(f"Max batch size: {max_batch_size}")

    # Iterate over the input BED file, splitting larger regions into chunks
    # of at most 'max_region_size'
    logger.info(f"Creating windows from {bed}")
    windows = [
        (chrom, window_idx, window_start, window_end) 
        for window_idx, (chrom, window_start, window_end) in enumerate(
            split_large_regions(read_bed_regions(bed), max_region_size=10000)
        )
    ]
    model = load_model(model_path)
    regions_per_block = max(4, threads)
    start_block = 0
    while start_block < len(windows):
        logger.info(f"Processing block {start_block}-{start_block + regions_per_block} of {len(windows)}  {start_block / len(windows) * 100 :.2f}% done")
        process_block(windows[start_block:start_block + regions_per_block],
                      bamfile=bamfile,
                      model=model,
                      reference_fasta=reference_fasta,
                      classifier_path=classifier_path,
                      tmpdir=tmpdir,
                      threads=threads,
                      max_read_depth=max_read_depth,
                      window_size=150,
                      vcf_out=vcf_out,
                      vcf_template=vcf_template,
                      max_batch_size=max_batch_size,
                      var_freq_file=var_freq_file)
        start_block += regions_per_block


def process_block(raw_regions,
              bamfile,
              model,
              reference_fasta,
              classifier_path,
              tmpdir,
              threads,
              max_read_depth,
              window_size,
              vcf_out,
              vcf_template,
              max_batch_size,
              var_freq_file=None):
    """
    For each region in the given list of regions, generate input tensors and then call variants on that data
    Input tensor generation is done in parallel and the result tensors are saved to disk in 'tmpdir'
    Calling is done on the main thread and loads the tensors one at a time from disk

    """
    sus_start = datetime.datetime.now()

    # First, find 'suspect' regions. This part is pretty fast
    cluster_positions_func = partial(
        cluster_positions_for_window,
        bamfile=bamfile,
        reference_fasta=reference_fasta,
        maxdist=100,
    )
    sus_regions = mp.Pool(threads).map(cluster_positions_func, raw_regions)
    sus_regions = util.merge_overlapping_regions( list(itertools.chain(*sus_regions)))
    sus_tot_bp = sum(r[3] - r[2] for r in sus_regions)
    enc_start = datetime.datetime.now()

    # Next, encode each suspect region and save it to disk. This is slow
    logger.info(f"Found {len(sus_regions)} suspicious regions with {sus_tot_bp}bp in {(enc_start - sus_start).total_seconds() :.3f} seconds")
    encoded_paths = encode_regions(bamfile, reference_fasta, sus_regions, tmpdir, threads, max_read_depth, window_size, batch_size=64, window_step=25)
    enc_elapsed = datetime.datetime.now() - enc_start
    logger.info(f"Encoded {len(encoded_paths)} regions in {enc_elapsed.total_seconds() :.2f}")

    # Now call variants over the encoded regions
    aln = pysam.AlignmentFile(bamfile)
    reference = pysam.FastaFile(reference_fasta)
    if var_freq_file:
        var_freq_file = pysam.VariantFile(var_freq_file)
    classifier_model = buildclf.load_model(classifier_path) if classifier_path else None

    var_records = call_multi_paths(encoded_paths, model, reference, aln, classifier_model, vcf_template, var_freq_file, max_batch_size)

    # Write the variant records to the VCF file
    for var in sorted(var_records, key=lambda x: x.pos):
        vcf_out.write(str(var))
    vcf_out.flush()


@torch.no_grad()
def call_multi_paths(encoded_paths, model, reference, aln, classifier_model, vcf_template, var_freq_file, max_batch_size):
    """
    Call variants from the encoded regions and return them as a list of variant records suitable for writing to a VCF file
    No more than max_batch_size are processed in a single batch
    """
    # Accumulate regions until we have at least this many
    min_samples_callbatch = 96

    batch_encoded = []
    batch_start_pos = []
    batch_reftoks = []
    batch_regions = []
    batch_count = 0
    call_start = datetime.datetime.now()
    window_count = 0
    var_records = []  # Stores all variant records so we can sort before writing
    window_idx = -1
    path = None  # Avoid rare case with logging when there is no path set

    for path in encoded_paths:
        # Load the data, parsing location + encoded data from file
        data = torch.load(path, map_location='cpu')
        chrom, start, end = data['region']
        window_idx = int(data['window_idx'])
        batch_encoded.append(data['encoded_pileup'])
        batch_start_pos.extend(data['start_positions'])
        batch_reftoks.extend(data['ref_tokens'])
        batch_regions.extend((chrom, start, end) for _ in range(len(data['start_positions'])))
        os.unlink(path)
        window_count += len(batch_start_pos)
        if len(batch_start_pos) > min_samples_callbatch:
            logger.debug(f"Calling variants on window {window_idx} path: {path}")
            batch_count += 1
            if len(batch_encoded) > 1:
                allencoded = torch.concat(batch_encoded, dim=0)
            else:
                allencoded = batch_encoded[0]
            hap0, hap1 = call_and_merge(allencoded, batch_start_pos, batch_reftoks, batch_regions, model, reference,
                                        max_batch_size)
            var_records.extend(
                vars_hap_to_records(hap0, hap1, aln, reference, classifier_model, vcf_template, var_freq_file)
            )
            batch_encoded = []
            batch_start_pos = []
            batch_regions = []
            batch_reftoks = []

    # Write last few
    logger.debug(f"Calling variants on window {window_idx} path: {path}")
    if len(batch_start_pos):
        batch_count += 1
        if len(batch_encoded) > 1:
            allencoded = torch.concat(batch_encoded, dim=0)
        else:
            allencoded = batch_encoded[0]
        hap0, hap1 = call_and_merge(allencoded, batch_start_pos, batch_reftoks, batch_regions, model, reference, max_batch_size)
        var_records.extend(
            vars_hap_to_records(hap0, hap1, aln, reference, classifier_model, vcf_template, var_freq_file)
        )

    call_elapsed = datetime.datetime.now() - call_start
    logger.info(
        f"Called variants in {window_count} windows over {batch_count} batches from {len(encoded_paths)} paths in {call_elapsed.total_seconds() :.2f} seconds"
    )
    return var_records


def encode_regions(bamfile, reference_fasta, regions, tmpdir, n_threads, max_read_depth, window_size, batch_size, window_step):
    """
    Encode and save all the regions in the regions list in parallel, and return the list of files to the saved data
    """
    torch.set_num_threads(1) # Required to avoid deadlocks when creating tensors
    encode_func = partial(
        encode_and_save_region,
        bamfile=bamfile,
        refpath=reference_fasta,
        destdir=tmpdir,
        max_read_depth=max_read_depth,
        window_size=window_size,
        min_reads=5,
        batch_size=batch_size,
        window_step=window_step,
    )

    futures = []
    result_paths = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_threads) as pool:
        for chrom, window_idx, start, end in regions:
            fut = pool.submit(encode_func, region=(chrom, window_idx, int(start), int(end)))
            futures.append(fut)

        for fut in futures:
            result = fut.result(timeout=300) # Timeout in 5 mins - typical time for encoding is 1-3 seconds
            if result is not None:
                result_paths.append(result)
            else:
                logger.error("Weird, found an empty result object, ignoring it... but this could mean missed variant calls")

    torch.set_num_threads(n_threads)
    return result_paths


def encode_and_save_region(bamfile, refpath, destdir, region, max_read_depth, window_size, min_reads, batch_size, window_step):
    """
    Encode the reads in the given region and save the data along with the region and start offsets to a file
    and return the absolute path of the file

    Somewhat confusingly, the 'region' argument must be a tuple of  (chrom, index, start, end)
    """
    chrom, window_idx, start, end = region
    aln = pysam.AlignmentFile(bamfile, reference_filename=refpath)
    reference = pysam.FastaFile(refpath)
    all_encoded = []
    all_starts = []
    all_reftoks = []
    logger.debug(f"Encoding region {chrom}:{start}-{end} idx: {window_idx}")
    for encoded_region, start_positions, ref_toks in _encode_region(aln, reference, chrom, start, end, max_read_depth,
                                                     window_size=window_size, min_reads=min_reads, batch_size=batch_size, window_step=window_step):
        all_encoded.append(encoded_region)
        all_starts.extend(start_positions)
        all_reftoks.extend(ref_toks)
    logger.debug(f"Done encoding region {chrom}:{start}-{end}, created {len(all_starts)} windows")
    if len(all_encoded) > 1:
        encoded = torch.concat(all_encoded, dim=0)
    elif len(all_encoded) == 1:
        encoded = all_encoded[0]
    else:
        logger.error(f"Uh oh, did not find any encoded paths!, region is {chrom}:{start}-{end}")
        return None

    data = {
        'encoded_pileup': encoded,
        'region': (chrom, start, end),
        'start_positions': all_starts,
        'ref_tokens': all_reftoks,
        'window_idx': window_idx,
    }
    dest = Path(destdir) / f"enc_chr{chrom}_{window_idx}_{randchars(6)}.pt"
    logger.debug(f"Saving data as {dest.absolute()}")
    torch.save(data, dest)
    return dest.absolute()


def call_and_merge(batch, batch_offsets, batch_reftoks, regions, model, reference, max_batch_size):
    """

    """
    # dists = np.array([r[2] - bo for r, bo in zip(regions, batch_offsets)])
    dists = np.array([r[2] - r[1] for r in regions])
    median_dist = np.median(dists)
    # subbatch_idx = np.zeros(len(dists), dtype=int)
    subbatch_idx = (dists <= median_dist).astype(int)

    byregion = defaultdict(list)

    # n_output_toks = min(150 // util.TGT_KMER_SIZE - 1, max(dists) // util.TGT_KMER_SIZE + 1)
    for sbi in range(max(subbatch_idx) + 1):
        which = np.where(subbatch_idx == sbi)[0]
        subbatch = batch[torch.tensor(which), :, :, :]
        subbatch_offsets = [b for i,b in zip(subbatch_idx, batch_offsets) if i == sbi]
        subbatch_dists =   [d for i,d in zip(subbatch_idx, dists) if i == sbi]
        subbatch_regions = [r for i,r in zip(subbatch_idx, regions) if i == sbi]
        subbatch_reftoks = [r for i, r in zip(subbatch_idx, batch_reftoks) if i == sbi]

        # n_output_toks = min(150 // util.TGT_KMER_SIZE - 1, max_dist // util.TGT_KMER_SIZE + 1)
        n_output_toks = [min(150 // util.TGT_KMER_SIZE, d // util.TGT_KMER_SIZE + 2) for d in subbatch_dists]

        logger.debug(f"Sub-batch size: {len(subbatch_offsets)}   max dist: {max(subbatch_dists)},  n_tokens: {n_output_toks}")
        batchvars = call_batch(subbatch, subbatch_offsets, subbatch_regions, subbatch_reftoks, model, reference, n_output_toks, max_batch_size=max_batch_size)
        # Batchvars is a list of two tuples of lists of Variant objects [ ([vars0...], [vars1...]), ([vars0..], [vars1..]) ], raw calls from each region not aggregated
        for region, bvars in zip(subbatch_regions, batchvars):
            byregion[region].append(bvars)

    hap0 = defaultdict(list)
    hap1 = defaultdict(list)
    for region, rvars in byregion.items():
        chrom, start, end = region
        h0, h1 = merge_genotypes(rvars)
        for k, v in h0.items():
            if start <= v[0].pos < end:
                hap0[k].extend(v)
        for k, v in h1.items():
            if start <= v[0].pos < end:
                hap1[k].extend(v)

    return hap0, hap1


def pop_reftoks(reftoks, n_toks, num_classes=260):
    batchtoks = []
    masks = []
    for i, rtok in enumerate(reftoks):
        if len(rtok) >= n_toks:
            s = util.START_TOKEN.expand(1, -1)
            t = torch.concat((s, torch.nn.functional.one_hot(torch.tensor(rtok[0:n_toks]), num_classes=num_classes)))
            batchtoks.append(t)
            m = torch.zeros(n_toks + 1)
            masks.append(m)
        else:
            num_starts = n_toks - len(rtok)
            s = util.START_TOKEN.expand(num_starts + 1, -1)
            if len(rtok) > 0:
                t = torch.concat((s, torch.nn.functional.one_hot(torch.tensor(rtok), num_classes=num_classes)))
            else:
                t = s
            batchtoks.append(t)
            m = torch.concat((torch.ones(num_starts) * float("-inf"), torch.zeros(len(rtok) + 1)), dim=0)
            masks.append(m)

    t = torch.stack(batchtoks)
    # Expand to haplotype dim
    t = t.unsqueeze(1).expand(-1, 2, -1, -1).float()

    masks = torch.stack(masks).float()

    return t, masks

def trim_preds(seq_preds, probs, offsets):
    assert seq_preds.shape[0] == offsets.shape[0]
    trimmed_preds = []
    trimmed_probs = []
    for i in range(seq_preds.shape[0]):
        offset = offsets[i].item()
        h0 = torch.argmax(seq_preds[i, 0, offset:, :], dim=-1)
        h1 = torch.argmax(seq_preds[i, 1, offset:, :], dim=-1)
        trimmed_preds.append((h0, h1))
        trimmed_probs.append(
            (probs[i, 0, offset:], probs[i, 1, offset:])
        )

    return trimmed_preds, trimmed_probs


def _call_safe(encoded_reads, model, reftoks, n_output_toks, max_batch_size, enable_amp=True):
    """
    Predict the sequence for the encoded reads, but dont submit more than 'max_batch_size' samples
    at once
    """
    seq_preds = []
    probs = []
    start = 0

    while start < encoded_reads.shape[0]:
        end = start + max_batch_size
        batch_reftoks = reftoks[start:end]
        batch_tokcounts = n_output_toks[start:end]
        maxrt = min(20, max(len(r) for r in batch_reftoks))
        pretoks, premask = pop_reftoks(batch_reftoks, maxrt, num_classes=util.FEATURE_DIM)
        reftok_offset = (premask == float("-inf")).sum(dim=1)
        toks_to_generate = max(5, max(batch_tokcounts))

        logger.debug(
            f"Requested {n_output_toks} total output toks, found max {maxrt} reftokens, will generate {toks_to_generate} tokens"
        )
        with torch.amp.autocast(device_type='cuda', enabled=enable_amp):
            preds, prbs = util.predict_sequence(
                                encoded_reads[start:end, :, :, :].to(DEVICE),
                                model,
                                pretoks,
                                n_output_toks=toks_to_generate,
                                padding_mask=premask,
                                device=DEVICE
                            )
        trimmed_preds, trimmed_probs = trim_preds(preds, prbs.detach().cpu(), reftok_offset)
        seq_preds.extend(trimmed_preds)
        probs.extend(trimmed_probs)
        start += max_batch_size

    return seq_preds, probs


def call_batch(encoded_reads, offsets, regions, reftoks, model, reference, n_output_toks, max_batch_size):
    """
    Call variants in a batch (list) of regions, by running a forward pass of the model and
    then aligning the predicted sequences to the reference genome and picking out any
    mismatching parts
    :returns : List of variants called in both haplotypes for every item in the batch as a list of 2-tuples
    """
    assert encoded_reads.shape[0] == len(regions), f"Expected the same number of reads as regions, but got {encoded_reads.shape[0]} reads and {len(regions)}"
    assert len(offsets) == len(regions), f"Should be as many offsets as regions, but found {len(offsets)} and {len(regions)}"

    seq_preds, probs = _call_safe(encoded_reads, model, reftoks, n_output_toks, max_batch_size)

    calledvars = []
    for offset, (chrom, start, end), b in zip(offsets, regions, range(len(seq_preds))):
        hap0 = util.kmer_idx_to_str(seq_preds[b][0], util.i2s)
        hap1 = util.kmer_idx_to_str(seq_preds[b][1], util.i2s)

        # o = ' ' * (offset - start + 120)
        # print(f"{chrom}:{start}-{end}\tH0:{o}{hap0}")
        # print(f"{chrom}:{start}-{end}\tH1:{o}{hap1}")

        probs0 = np.exp(util.expand_to_bases(probs[b][0].numpy()))
        probs1 = np.exp(util.expand_to_bases(probs[b][1].numpy()))

        refseq = reference.fetch(chrom, offset, offset + len(hap0))
        vars_hap0 = list(v for v in vcf.aln_to_vars(refseq, hap0, chrom, offset, probs=probs0) if start <= v.pos <= end)
        vars_hap1 = list(v for v in vcf.aln_to_vars(refseq, hap1, chrom, offset, probs=probs1) if start <= v.pos <= end)

        print("vars0:" + ', '.join(f"{v.chrom}:{v.pos} {v.ref}-{v.alt}" for v in vars_hap0))
        print("vars1:" + ', '.join(f"{v.chrom}:{v.pos} {v.ref}-{v.alt}" for v in vars_hap1))

        #print(f"Offset: {offset}\twindow {start}-{end} frame: {start % 4} hap0: {vars_hap0}\n       hap1: {vars_hap1}")
        #calledvars.append((vars_hap0, vars_hap1))
        calledvars.append((vars_hap0[0:5], vars_hap1[0:5]))

    return calledvars


def merge_multialts(v0, v1):
    """
    Merge two VcfVar objects into a single one with two alts

    ATCT   G
    A     GCAC
      -->
    ATCT   G,GCACT

    """
    assert v0.pos == v1.pos
    #assert v0.het and v1.het
    if v0.ref == v1.ref:
        v0.alts = (v0.alts[0], v1.alts[0])
        v0.qual = (v0.qual + v1.qual) / 2  # Average quality??
        v0.samples['sample']['GT'] = (1,2)
        return v0

    else:
        shorter, longer = sorted([v0, v1], key=lambda x: len(x.ref))
        extra_ref = longer.ref[len(shorter.ref):]
        newalt = shorter.alts[0] + extra_ref
        longer.alts = (longer.alts[0], newalt)
        longer.qual = (longer.qual + shorter.qual) / 2
        longer.samples['sample']['GT'] = (1, 2)
        return longer


def het(rec):
    return rec.samples[0]['GT'] == (0,1) or rec.samples[0]['GT'] == (1,0)


def merge_overlaps(overlaps, min_qual):
    """
     Attempt to merge overlapping VCF records in a sane way
     As a special case if there is only one input record we just return that
    """
    if len(overlaps) == 1:
        return [overlaps[0]]
    overlaps = list(filter(lambda x: x.qual > min_qual, overlaps))
    if len(overlaps) == 1:
        # An important case where two variants overlap but one of them is low quality
        # Should the remaining variant be het or hom?
        return [overlaps[0]]
    elif len(overlaps) == 0:
        return []

    result = []
    overlaps = sorted(overlaps, key=lambda x: x.qual, reverse=True)[0:2]  # Two highest quality alleles
    overlaps[0].samples['sample']['GT'] = (None, 1)
    overlaps[1].samples['sample']['GT'] = (1, None)
    result.extend(sorted(overlaps, key=lambda x: x.pos))
    return result


def collect_phasegroups(vars_hap0, vars_hap1, aln, reference, minimum_safe_distance=100):
    allkeys = sorted(set(list(k for k in vars_hap0.keys()) + list(k for k in vars_hap1.keys())), key=lambda x: x[1])

    all_vcf_vars = []
    group0 = defaultdict(list)
    group1 = defaultdict(list)
    prevpos = -1000
    prevchrom = None
    for k in allkeys:
        chrom, pos, ref, alt = k
        if (chrom != prevchrom) or (pos - prevpos > minimum_safe_distance):
            vcf_vars = vcf.vcf_vars(
                vars_hap0=group0,
                vars_hap1=group1,
                aln=aln,
                reference=reference
            )
            all_vcf_vars.extend(vcf_vars)

            group0 = defaultdict(list)
            group1 = defaultdict(list)
            if k in vars_hap0:
                group0[k].extend(vars_hap0[k])
            if k in vars_hap1:
                group1[k].extend(vars_hap1[k])
            prevpos = pos
        else:
            if k in vars_hap0:
                group0[k].extend(vars_hap0[k])
            if k in vars_hap1:
                group1[k].extend(vars_hap1[k])
        prevchrom = chrom

    vcf_vars = vcf.vcf_vars(
        vars_hap0=group0,
        vars_hap1=group1,
        aln=aln,
        reference=reference
    )
    all_vcf_vars.extend(vcf_vars)
    return all_vcf_vars


def vars_hap_to_records(vars_hap0, vars_hap1, aln, reference, classifier_model, vcf_template, var_freq_file):
    """
    Convert variant haplotype objects to variant records,
    """

    # Merging vars can sometimes cause a poor quality variant to clobber a very high quality one, to avoid this
    # we hard-filter out very poor quality variants that overlap other, higher-quality variants
    # This value defines the min qual to be included when merging overlapping variants
    min_merge_qual = 0.01

    vcf_vars = collect_phasegroups(vars_hap0, vars_hap1, aln, reference, minimum_safe_distance=100)

    # covert variants to pysam vcf records
    vcf_records = [
        vcf.create_vcf_rec(var, vcf_template)
        for var in sorted(vcf_vars, key=lambda x: x.pos)
    ]

    if not vcf_records:
        return []

    for rec in vcf_records:
        if rec.ref == rec.alts[0]:
            logger.warning(f"Whoa, found a REF == ALT variant: {rec}")

        if classifier_model:
            rec.info["RAW_QUAL"] = rec.qual
            rec.qual = buildclf.predict_one_record(classifier_model, rec, aln, var_freq_file)

    merged = []
    overlaps = [vcf_records[0]]
    for rec in vcf_records[1:]:
        if overlaps and util.records_overlap(overlaps[-1], rec):
            overlaps.append(rec)
        elif overlaps: #
            result = merge_overlaps(overlaps, min_qual=min_merge_qual)
            merged.extend(result)
            overlaps = [rec]
        else:
            overlaps = [rec]

    if overlaps:
        merged.extend(merge_overlaps(overlaps, min_qual=min_merge_qual))
    else:
        merged.append(rec)

    return merged



def add_ref_bases(encbases, refseq, max_read_depth):
    """
    Add the reference sequence as read 0
    """
    ref_encoded = bam.string_to_tensor(refseq)
    return torch.cat((ref_encoded.unsqueeze(1), encbases), dim=1)[:, 0:max_read_depth, :]


def _encode_region(aln, reference, chrom, start, end, max_read_depth, window_size=150, min_reads=5, batch_size=64, window_step=25):
    """
    Generate batches of tensors that encode read data from the given genomic region, along with position information. Each
    batch of tensors generated should be suitable for input into a forward pass of the model - but the data will be on the
    CPU.
    Each item in the batch represents a pileup in a single 'window' into the given region of size 'window_size', and
    subsequent elements are encoded from a sliding window that advances by 'window_step' after each item.
    If start=100, window_size is 50, and window_step is 10, then the items will include data from regions:
    100-150
    110-160
    120-170,
    etc

    The start positions for each item in the batch are returned in the 'batch_offsets' element, which is required
    when variant calling to determine the genomic coordinates of the called variants.

    If the full region size is small this will probably generate just a single batch, but if the region is very large
    (or batch_size is small) this could generate multiple batches

    :param window_size: Size of region in bp to generate for each item
    :returns: Generator for tuples of (batch tensor, list of start positions)
    """
    window_start = int(start - 0.7 * window_size)  # We start with regions a bit upstream of the focal / target region
    batch = []
    batch_offsets = []
    batch_toks = []
    readwindow = bam.ReadWindow(aln, chrom, start - 150, end + window_size)
    logger.debug(f"Encoding region {chrom}:{start}-{end}")
    returned_count = 0
    while window_start <= (end - 0.2 * window_size):
        refseq = reference.fetch(chrom, window_start, window_start + window_size)
        try:
            #logger.debug(f"Getting reads from  readwindow: {window_start} - {window_start + window_size}")
            enc_reads = readwindow.get_window(window_start, window_start + window_size, max_reads=max_read_depth)
            encoded_with_ref = add_ref_bases(enc_reads, refseq, max_read_depth=max_read_depth)
            tok_bases = ((start - window_start) // 4) * 4
            window_toks = util.bases_to_kvec(refseq[0:max(0, tok_bases)], util.s2i)
            batch_toks.append(window_toks)
            batch.append(encoded_with_ref)
            batch_offsets.append(window_start)
            #logger.debug(f"Added item to batch from window_start {window_start}")
        except bam.LowReadCountException:
            logger.info(
                f"Bam window {chrom}:{window_start}-{window_start + window_size} "
                f"had too few reads for variant calling (< {min_reads})"
            )
        window_start += window_step
        if len(batch) >= batch_size:
            encodedreads = torch.stack(batch, dim=0).cpu().float()
            returned_count += 1
            yield encodedreads, batch_offsets, batch_toks
            batch = []
            batch_offsets = []
            batch_toks = []

    # Last few
    if batch:
        encodedreads = torch.stack(batch, dim=0).cpu().float() # Keep encoded tensors on cpu for now
        returned_count += 1
        yield encodedreads, batch_offsets, batch_toks

    if not returned_count:
        logger.info(f"Region {chrom}:{start}-{end} has only low coverage areas, not encoding data")


def merge_genotypes(genos):
    """
    Genos is a list of 2-tuples representing haplotypes called in overlapping windows. This function
    attempts to merge the haplotypes in a way that makes sense, given the possibility of conflicting information

    """
    allvars = sorted(list(v for g in genos for v in g[0]) + list(v for g in genos for v in g[1]), key=lambda v: v.pos)
    allkeys = set()
    varsnodups = []
    for v in allvars:
        if v.key not in allkeys:
            allkeys.add(v.key)
            varsnodups.append(v)

    results = [[], []]

    prev_het = None
    prev_het_index = None
    for p in varsnodups:
        homcount = 0
        hetcount = 0
        for g in genos:
            a = p.key in [v.key for v in g[0]]
            b = p.key in [v.key for v in g[1]]
            if a and b:
                homcount += 1
            elif a or b:
                hetcount += 1
        if homcount > hetcount:
            results[0].append(p)
            results[1].append(p)
        elif prev_het is None:
            results[0].append(p)
            prev_het = p
            prev_het_index = 0
        else:
            # There was a previous het variant, so figure out where this new one should go
            # determine if p should be in cis or trans with prev_het
            cis = 0
            trans = 0
            for g in genos:
                g0keys = [v.key for v in g[0]]
                g1keys = [v.key for v in g[1]]
                if (p.key in g0keys and prev_het.key in g0keys) or (p.key in g1keys and prev_het.key in g1keys):
                    cis += 1
                elif (p.key in g0keys and prev_het.key in g1keys) or (p.key in g1keys and prev_het.key in g0keys):
                    trans += 1
            if trans >= cis: # If there's a tie, assume trans. This covers the case where cis==0 and trans==0, because trans is safer
                results[1 - prev_het_index].append(p)
                prev_het = p
                prev_het_index = 1 - prev_het_index
            else:
                results[prev_het_index].append(p)
                prev_het = p
                prev_het_index = prev_het_index

    # Build dictionaries with correct haplotypes...
    allvars0 = dict()
    allvars1 = dict()
    for v in results[0]:
        allvars0[v.key] = [t for t in allvars if t.key == v.key]
    for v in results[1]:
        allvars1[v.key] = [t for t in allvars if t.key == v.key]
    return allvars0, allvars1

