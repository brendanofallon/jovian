"""
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any
later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program.
If not, see <https://www.gnu.org/licenses/>.
"""
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
    #encoder_attention_heads = 8
    #decoder_attention_heads = 10 
    #dim_feedforward = 512
    #encoder_layers = 10
    #decoder_layers = 10 
    #embed_dim_factor = 160 

    #50M params
    encoder_attention_heads = 8
    decoder_attention_heads = 4 
    dim_feedforward = 512
    encoder_layers = 8
    decoder_layers = 6
    embed_dim_factor = 120 


    # 35M params
    #encoder_attention_heads = 8 # was 4
    #decoder_attention_heads = 4 # was 4
    #dim_feedforward = 512
    #encoder_layers = 6
    #decoder_layers = 4 # was 2
    #embed_dim_factor = 120 # was 100

    model = VarTransformer(read_depth=100,
                            feature_count=10,
                            kmer_dim=util.FEATURE_DIM, # Number of possible kmers
                            n_encoder_layers=encoder_layers,
                            n_decoder_layers=decoder_layers,
                            embed_dim_factor=embed_dim_factor,
                            encoder_attention_heads=encoder_attention_heads,
                            decoder_attention_heads=decoder_attention_heads,
                            d_ff=dim_feedforward,
                            device=DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    model.to(DEVICE)
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
    tmpdir_root = Path(kwargs.get("temp_dir"))
    threads = kwargs.get('threads', 1)
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
    bamfile, bed, reference_fasta, model_path, classifier_path, threads, tmpdir, var_freq_file, vcf_out, vcf_template,
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
    max_read_depth = 100

    # Iterate over the input BED file, splitting larger regions into chunks
    # of at most 'max_region_size'
    logger.info(f"Creating windows from {bed}")
    windows = [
        (chrom, window_idx, window_start, window_end) 
        for window_idx, (chrom, window_start, window_end) in enumerate(
            split_large_regions(read_bed_regions(bed), max_region_size=10000)
        )
    ]

    regions_per_block = max(4, threads)
    start_block = 0
    while start_block < len(windows):
        logger.info(f"Processing block {start_block}-{start_block + regions_per_block} of {len(windows)}  {start_block / len(windows) * 100 :.2f}% done")
        process_block(windows[start_block:start_block + regions_per_block],
                      bamfile=bamfile,
                      model_path=model_path,
                      reference_fasta=reference_fasta,
                      classifier_path=classifier_path,
                      tmpdir=tmpdir,
                      threads=threads,
                      max_read_depth=max_read_depth,
                      window_size=150,
                      vcf_out=vcf_out,
                      vcf_template=vcf_template,
                      var_freq_file=var_freq_file)
        start_block += regions_per_block


def process_block(raw_regions,
              bamfile,
              model_path,
              reference_fasta,
              classifier_path,
              tmpdir,
              threads,
              max_read_depth,
              window_size,
              vcf_out,
              vcf_template,
              var_freq_file=None):
    """
    For each region in the given list of regions, generate input tensors and then call variants on that data
    Input tensor generation is done in parallel and the result tensors are saved to disk in 'tmpdir'
    Calling is done on the main thread and loads the tensors one at a time from disk

    """
    sus_start = datetime.datetime.now()
    cluster_positions_func = partial(
        cluster_positions_for_window,
        bamfile=bamfile,
        reference_fasta=reference_fasta,
        maxdist=100,
    )
    sus_regions = mp.Pool(threads).map(cluster_positions_func, raw_regions)
    sus_regions = list(itertools.chain(*sus_regions))
    sus_tot_bp = sum(r[3] - r[2] for r in sus_regions)
    enc_start = datetime.datetime.now()
    logger.info(f"Found {len(sus_regions)} suspicious regions with {sus_tot_bp}bp in {(enc_start - sus_start).total_seconds() :.3f} seconds")
    encoded_paths = encode_regions(bamfile, reference_fasta, sus_regions, tmpdir, threads, max_read_depth, window_size, batch_size=64, window_step=25)
    enc_elapsed = datetime.datetime.now() - enc_start
    logger.info(f"Encoded {len(encoded_paths)} regions in {enc_elapsed.total_seconds() :.2f}")
    model = load_model(model_path)
    aln = pysam.AlignmentFile(bamfile)
    reference = pysam.FastaFile(reference_fasta)

    if var_freq_file:
        var_freq_file = pysam.VariantFile(var_freq_file)
    classifier_model = buildclf.load_model(classifier_path) if classifier_path else None

    min_samples_callbatch = 16 # Accumulate regions until we have at least this many
    batch_encoded = []
    batch_start_pos = []
    batch_regions = []
    call_start = datetime.datetime.now()
    batch_count = 0
    window_count = 0
    var_records = [] # Stores all variant records so we can sort before writing
    for path in encoded_paths:
        # Load the data, parsing location + encoded data from file
        data = torch.load(path, map_location='cpu')
        chrom, start, end = data['region']
        window_idx = int(data['window_idx'])
        batch_encoded.append(data['encoded_pileup'])
        batch_start_pos.extend(data['start_positions'])
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
            hap0, hap1 = call_and_merge(allencoded, batch_start_pos, batch_regions, model, reference)
            var_records.extend(
                vars_hap_to_records(
                    chrom, window_idx, hap0, hap1, aln, reference, classifier_model, vcf_template, var_freq_file
                )
            )
            batch_encoded = []
            batch_start_pos = []
            batch_regions = []

    # Write last few
    logger.debug(f"Calling variants on window {window_idx} path: {path}")
    if len(batch_start_pos):
        batch_count += 1
        if len(batch_encoded) > 1:
            allencoded = torch.concat(batch_encoded, dim=0)
        else:
            allencoded = batch_encoded[0]
        hap0, hap1 = call_and_merge(allencoded, batch_start_pos, batch_regions, model, reference)
        var_records.extend(
            vars_hap_to_records(
                chrom, window_idx, hap0, hap1, aln, reference, classifier_model, vcf_template, var_freq_file
            )
        )

    for var in sorted(var_records, key=lambda x: x.pos):
        vcf_out.write(str(var))
    vcf_out.flush()
    call_elapsed = datetime.datetime.now() - call_start
    logger.info(f"Called variants in {window_count} windows over {batch_count} batches from {len(encoded_paths)} paths in {call_elapsed.total_seconds() :.2f} seconds")



def encode_regions(bamfile, reference_fasta, regions, tmpdir, n_threads, max_read_depth, window_size, batch_size, window_step):
    """
    Encode and save all the regions in the regions list in parallel, and return the list of files to the saved data
    """
    torch.set_num_threads(1) # Required to avoid deadlocks when creating tensors
    encode_func = partial(
        encode_and_save_region,
        bamfile=bamfile,
        refpath=reference_fasta,
        tmpdir=tmpdir,
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


def encode_and_save_region(bamfile, refpath, tmpdir, region, max_read_depth, window_size, min_reads, batch_size, window_step):
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
    logger.debug(f"Encoding region {chrom}:{start}-{end} idx: {window_idx}")
    for encoded_region, start_positions in _encode_region(aln, reference, chrom, start, end, max_read_depth,
                                                     window_size=window_size, min_reads=min_reads, batch_size=batch_size, window_step=window_step):
        all_encoded.append(encoded_region)
        all_starts.extend(start_positions)
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
        'window_idx': window_idx,
    }
    dest = Path(tmpdir) / f"enc_chr{chrom}_{window_idx}_{randchars(4)}.pt"
    logger.debug(f"Saving data as {dest.absolute()}")
    torch.save(data, dest)
    return dest.absolute()


def call_and_merge(batch, batch_offsets, regions, model, reference):
    max_dist = max(r[2] - bo for r, bo in zip(regions, batch_offsets))
    n_output_toks = min(150 // util.TGT_KMER_SIZE - 1, (max_dist) // util.TGT_KMER_SIZE + 1)
    logger.debug(f"window max dist: {max_dist}, min batch offset: {min(batch_offsets)}, n_tokens: {n_output_toks}")
    logger.debug(f"Calling batch of size {len(batch_offsets)}")
    batchvars = call_batch(batch, batch_offsets, regions, model, reference, n_output_toks)

    byregion = defaultdict(list)
    for region, bvars in zip(regions, batchvars):
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


def vars_hap_to_records(
    chrom, window_idx, vars_hap0, vars_hap1, aln, reference, classifier_model, vcf_template, var_freq_file
):
    """
    Convert variant haplotypes to variant records and write the records to a temporary VCF file.
    """
    vcf_vars = vcf.vcf_vars(
        vars_hap0=vars_hap0,
        vars_hap1=vars_hap1,
        chrom=chrom,
        window_idx=window_idx,
        aln=aln,
        reference=reference
    )

    # covert variants to pysam vcf records
    vcf_records = [
        vcf.create_vcf_rec(var, vcf_template)
        for var in sorted(vcf_vars, key=lambda x: x.pos)
    ]

    for rec in vcf_records:
        if classifier_model:
            rec.info["RAW_QUAL"] = rec.qual
            rec.qual = buildclf.predict_one_record(classifier_model, rec, aln, var_freq_file)

    return vcf_records


def call_batch(encoded_reads, offsets, regions, model, reference, n_output_toks):
    """
    Call variants in a batch (list) of regions, by running a forward pass of the model and
    then aligning the predicted sequences to the reference genome and picking out any
    mismatching parts
    :returns : List of variants called in both haplotypes for every item in the batch as a list of 2-tuples
    """
    assert encoded_reads.shape[0] == len(regions), f"Expected the same number of reads as regions, but got {encoded_reads.shape[0]} reads and {len(regions)}"
    assert len(offsets) == len(regions), f"Should be as many offsets as regions, but found {len(offsets)} and {len(regions)}"
    seq_preds, probs = util.predict_sequence(encoded_reads.to(DEVICE), model, n_output_toks=n_output_toks, device=DEVICE)
    probs = probs.detach().cpu().numpy()
    calledvars = []
    for offset, (chrom, start, end), b in zip(offsets, regions, range(seq_preds.shape[0])):
        hap0_t, hap1_t = seq_preds[b, 0, :, :], seq_preds[b, 1, :, :]
        hap0 = util.kmer_preds_to_seq(hap0_t, util.i2s)
        hap1 = util.kmer_preds_to_seq(hap1_t, util.i2s)
        probs0 = np.exp(util.expand_to_bases(probs[b, 0, :]))
        probs1 = np.exp(util.expand_to_bases(probs[b, 1, :]))

        refseq = reference.fetch(chrom, offset, offset + len(hap0))
        vars_hap0 = list(v for v in vcf.aln_to_vars(refseq, hap0, offset, probs=probs0) if start <= v.pos <= end)
        vars_hap1 = list(v for v in vcf.aln_to_vars(refseq, hap1, offset, probs=probs1) if start <= v.pos <= end)
        #print(f"Offset: {offset}\twindow {start}-{end} frame: {start % 4} hap0: {vars_hap0}\n       hap1: {vars_hap1}")
        #calledvars.append((vars_hap0, vars_hap1))
        calledvars.append((vars_hap0[0:4], vars_hap1[0:4]))

    return calledvars


def add_ref_bases(encbases, reference, chrom, start, end, max_read_depth):
    """
    Add the reference sequence as read 0
    """
    refseq = reference.fetch(chrom, start, end)
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
    window_start = int(start - 0.5 * window_size)  # We start with regions a bit upstream of the focal / target region
    batch = []
    batch_offsets = []
    readwindow = bam.ReadWindow(aln, chrom, start - 100, end + window_size)
    logger.debug(f"Encoding region {chrom}:{start}-{end}")
    returned_count = 0
    while window_start <= (end - 0.2 * window_size):
        try:
            #logger.debug(f"Getting reads from  readwindow: {window_start} - {window_start + window_size}")
            enc_reads = readwindow.get_window(window_start, window_start + window_size, max_reads=max_read_depth)
            encoded_with_ref = add_ref_bases(enc_reads, reference, chrom, window_start, window_start + window_size,
                                             max_read_depth=max_read_depth)
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
            yield encodedreads, batch_offsets
            batch = []
            batch_offsets = []

    # Last few
    if batch:
        encodedreads = torch.stack(batch, dim=0).cpu().float() # Keep encoded tensors on cpu for now
        returned_count += 1
        yield encodedreads, batch_offsets

    if not returned_count:
        logger.info(f"Region {chrom}:{start}-{end} has only low coverage areas, not encoding data")


# def _call_vars_region(
#     chrom, window_idx, start, end, aln, model, reference, max_read_depth, window_size=300, min_reads=5, window_step=25,
# ):
#     """
#     For the given region, identify variants by repeatedly calling the model over a sliding window,
#     tallying all of the variants called, and passing back all call and repeat count info
#     for further exploration
#     Currently:
#     - exclude all variants in the downstream half of the window
#     - retain all remaining var calls noting how many time each one was called, qualities, etc.
#     - call with no repeats are mostly false positives but they are retained
#     - haplotype 0 and 1 for each step are set by comparing with repeat vars from previous steps
#
#     TODO:
#       - add prob info from alt sequence to vars?
#       - add depth derived from tensor to vars?
#       - create new prob from all duplicate calls?
#     """
#     cpname = mp.current_process().name
#     logger.debug(
#         f"{cpname}: Processing region: {chrom}:{start}-{end} on window {window_idx}"
#     )
#
#     step_count = 0  # initialize
#     # var_retain_window_size = 145
#     batch_size = 128
#
#     enctime_total = datetime.timedelta(0)
#     calltime_total = datetime.timedelta(0)
#     encstart = datetime.datetime.now()
#     hap0 = defaultdict(list)
#     hap1 = defaultdict(list)
#     for batch, batch_offsets in _encode_region(aln, reference, chrom, start, end,
#                                                max_read_depth,
#                                                window_size=window_size,
#                                                min_reads=min_reads,
#                                                batch_size=batch_size,
#                                                window_step=window_step):
#         logger.debug(f"{cpname}: Forward pass for batch with starts {min(batch_offsets)} - {max(batch_offsets)}")
#         logger.debug(f"Encoded {len(batch)} windows for region {start}-{end} size: {end - start}")
#         enctime_total += (datetime.datetime.now() - encstart)
#         callstart = datetime.datetime.now()
#         n_output_toks = min(150 // util.TGT_KMER_SIZE - 1, (end - min(batch_offsets)) // util.TGT_KMER_SIZE + 1)
#         logger.debug(f"window end: {end}, min batch offset: {min(batch_offsets)}, n_tokens: {n_output_toks}")
#         batchvars = call_batch(batch, batch_offsets, [(start, end) for _ in range(batch.shape[0])], model, reference, chrom, n_output_toks)
#
#         h0, h1 = merge_genotypes(batchvars)
#
#         for k, v in h0.items():
#             hap0[k].extend(v)
#         for k, v in h1.items():
#             hap1[k].extend(v)
#         calltime_total += (datetime.datetime.now() - callstart)
#
#         step_count += batch.shape[0]
#
#     # Only return variants that are actually in the window
#     hap0_passing = {k: v for k, v in hap0.items() if start <= v[0].pos <= end}
#     hap1_passing = {k: v for k, v in hap1.items() if start <= v[0].pos <= end}
#
#     logger.debug(f"{cpname}: Enc time total: {enctime_total.total_seconds()}  calltime total: {calltime_total.total_seconds()}")
#     return chrom, window_idx, hap0_passing, hap1_passing


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


