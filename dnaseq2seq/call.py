
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
import queue
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

REGION_STOP_TOKEN = "stop"

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
    Generate a list of ranges containing a list of positions from the given window
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
    torch.set_num_threads(1) # Per-process ?
    mp.set_start_method('spawn')
    start_time = time.perf_counter()
    tmpdir_root = Path(kwargs.get("temp_dir"))
    threads = kwargs.get('threads', 1)
    max_batch_size = kwargs.get('max_batch_size', 64)
    logger.info(f"Using {threads} threads for encoding")
    logger.info(f"Found torch device: {DEVICE}")

    if 'cuda' in str(DEVICE):
        for idev in range(torch.cuda.device_count()):
            logger.info(f"CUDA device {idev} name: {torch.cuda.get_device_name({idev})}")

    logger.info(f"The model will be loaded from path {model_path}")

    vcf_header_extras = kwargs.get('cmdline')


    call_vars_in_parallel(
        bampath=bam,
        bed=bed,
        refpath=reference_fasta,
        model_path=model_path,
        classifier_path=classifier_path,
        threads=threads,
        max_batch_size=max_batch_size,
        vcf_out=vcf_out,
        vcf_header_extras=vcf_header_extras,
    )

    logger.info(f"All variants saved to {vcf_out}")
    end_time = time.perf_counter()
    elapsed_seconds = end_time - start_time
    if elapsed_seconds > 3600:
        logger.info(f"Total running time of call subcommand is: {elapsed_seconds / 3600 :.2f} hours")
    elif elapsed_seconds > 120:
        logger.info(f"Total running time of call subcommand is: {elapsed_seconds / 60 :.2f} minutes")
    else:
        logger.info(f"Total running time of call subcommand is: {elapsed_seconds :.2f} seconds")


def call_vars_in_parallel(
    bampath, bed, refpath, model_path, classifier_path, threads, max_batch_size, vcf_out, vcf_header_extras,
):
    regions_queue = mp.Queue(maxsize=1024)  # Hold BED file regions, generated in main process and sent to 'generate_tensors' workers
    tensors_queue = mp.Queue(maxsize=threads * 100)  # Holds tensors generated in 'generate tensors' workers, consumed by accumulate_regions_and_call


    # This one processes the input BED file and find 'suspect regions', and puts them in the regions_queue
    region_finder = mp.Process(target=find_regions, args=(regions_queue, bed, bampath, refpath, threads))
    region_finder.start()

    region_workers = [mp.Process(target=generate_tensors,
                                 args=(regions_queue, tensors_queue, bampath, refpath))
                      for _ in range(threads)]

    for p in region_workers:
        p.start()

    model_proc = mp.Process(target=accumulate_regions_and_call,
                            args=(model_path, tensors_queue, refpath, bampath, classifier_path, max_batch_size, vcf_out, vcf_header_extras, threads))
    model_proc.start()

    region_finder.join()
    logger.info("Done finding regions")

    for p in region_workers:
        p.join()
    logger.info("Region workers are done")

    model_proc.join()
    logger.info("All done")



def find_regions(regionq, inputbed, bampath, refpath, n_signals):
    """
    Read the input BED formatted file and merge / split the regions into big chunks
    Then finc regions that may contain a variant, and add all of these
    to the region_queue
    """
    torch.set_num_threads(2) # Must be here for it to work for this process
    region_count = 0
    tot_size_bp = 0
    sus_region_bp = 0
    sus_region_count = 0
    for idx, (chrom, window_start, window_end) in enumerate(split_large_regions(read_bed_regions(inputbed), max_region_size=10000)):
        region_count += 1
        tot_size_bp += window_end - window_start

        sus_regions = cluster_positions_for_window(
            (chrom, idx, window_start, window_end),
            bamfile=bampath,
            reference_fasta=refpath,
            maxdist=100,
        )

        sus_regions = util.merge_overlapping_regions(sus_regions)
        for r in sus_regions:
            sus_region_count += 1
            sus_region_bp += r[-1] - r[-2]
            regionq.put(r)

        if idx % 1 == 0:
            logger.info(f"Read {region_count} raw regions with suspect regions: {sus_region_count} tot bp: {sus_region_bp} ")

    logger.info("Done generating sus regions")
    for i in range(n_signals):
        regionq.put(REGION_STOP_TOKEN)


def encode_region(bampath, refpath, region, max_read_depth, window_size, min_reads, batch_size, window_step):
    """
    Encode the reads in the given region and save the data along with the region and start offsets to a file
    and return the absolute path of the file

    Somewhat confusingly, the 'region' argument must be a tuple of  (chrom, index, start, end)
    """
    chrom, idx, start, end = region
    torch.set_num_threads(2) # Must be here for it to work for this process
    logger.debug(f"Entering func for region {chrom}:{start}-{end}")
    aln = pysam.AlignmentFile(bampath, reference_filename=refpath)
    reference = pysam.FastaFile(refpath)
    all_encoded = []
    all_starts = []
    logger.debug(f"Encoding region {chrom}:{start}-{end}")
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
    }
    return data

def generate_tensors(region_queue: mp.Queue, output_queue: mp.Queue, bampath, refpath, max_read_depth=150, window_size=150):
    """
    Consume regions from the region_queue and generate input tensors for each and put them into the output_queue
    """
    min_reads = 5 # Abort if there are fewer than this many reads
    batch_size = 32 # Tensors hold this many regions
    window_step = 25

    encoded_region_count = 0
    while True:
        region = region_queue.get()
        if region == REGION_STOP_TOKEN:
            logger.debug("Region worker found end token")
            output_queue.put(None)
            break
        else:
            logger.debug(f"Encoding region {region}")
            data = encode_region(bampath, refpath, region, max_read_depth, window_size, min_reads, batch_size=batch_size, window_step=window_step)
            if data is not None:
                data['encoded_pileup'].share_memory_()
                encoded_region_count += 1
                output_queue.put(data)
            else:
                logger.warning(f"Whoa, got back None from encode_region")

    logger.debug("Region worker will wait a bit to shut down....")
    time.sleep(2)
    logger.info(f"Region worker {os.getpid()} is shutting down after generating {encoded_region_count} encoded regions")


@torch.no_grad()
def call_multi_paths(datas, model, reference, aln, classifier_model, vcf_template, max_batch_size):
    """
    Call variants from the encoded regions and return them as a list of variant records suitable for writing to a VCF file
    No more than max_batch_size are processed in a single batch
    """
    # Accumulate regions until we have at least this many
    # Bigger number here use more memory but allow for more efficient processing downstream
    min_samples_callbatch = 256

    batch_encoded = []
    batch_start_pos = []
    batch_regions = []
    batch_count = 0
    call_start = datetime.datetime.now()
    window_count = 0
    var_records = []  # Stores all variant records

    for data in datas:
        # Load the data, parsing location + encoded data from file
        chrom, start, end = data['region']
        batch_encoded.append(data['encoded_pileup'])
        batch_start_pos.extend(data['start_positions'])
        batch_regions.extend((chrom, start, end) for _ in range(len(data['start_positions'])))
        window_count += len(batch_start_pos)
        if len(batch_start_pos) > min_samples_callbatch:
            batch_count += 1
            if len(batch_encoded) > 1:
                allencoded = torch.concat(batch_encoded, dim=0)
            else:
                allencoded = batch_encoded[0]
            allencoded = allencoded.float()
            hap0, hap1 = call_and_merge(allencoded, batch_start_pos, batch_regions, model, reference,
                                        max_batch_size)
            var_records.extend(
                vars_hap_to_records(chrom, -1, hap0, hap1, aln, reference, classifier_model, vcf_template)
            )
            batch_encoded = []
            batch_start_pos = []
            batch_regions = []

    # Write last few
    if len(batch_start_pos):
        batch_count += 1
        if len(batch_encoded) > 1:
            allencoded = torch.concat(batch_encoded, dim=0)
        else:
            allencoded = batch_encoded[0]
        allencoded = allencoded.float()
        hap0, hap1 = call_and_merge(allencoded, batch_start_pos, batch_regions, model, reference, max_batch_size)
        var_records.extend(
            vars_hap_to_records(chrom, -1, hap0, hap1, aln, reference, classifier_model, vcf_template)
        )

    call_elapsed = datetime.datetime.now() - call_start
    logger.info(
        f"Called variants in {window_count} windows over {batch_count} batches from {len(datas)} paths in {call_elapsed.total_seconds() :.2f} seconds"
    )
    return var_records


def accumulate_regions_and_call(modelpath: str,
                                inputq: mp.Queue,
                                refpath: str,
                                bampath: str,
                                classifier_path,
                                max_batch_size: int,
                                vcf_out_path: str,
                                header_extras: str,
                                n_region_workers: int):
    """
    Continually poll the input queue to find new encoded regions, and call variants over those regions
    This function is typically called inside a subprocess and runs until it finds a None entry in the queue
    Variants are written to the output VCF file here
    """

    torch.set_num_threads(4)
    model = load_model(modelpath)
    model.eval()
    classifier = buildclf.load_model(classifier_path)

    vcf_header = vcf.create_vcf_header(sample_name="sample", lowcov=20, cmdline=header_extras)
    vcf_template = pysam.VariantFile("/dev/null", mode='w', header=vcf_header)
    logger.info(f"Writing variants to {Path(vcf_out_path).absolute()}")

    vcf_out = open(vcf_out_path, "w")
    vcf_out.write(str(vcf_header))

    reference = pysam.FastaFile(refpath)
    aln = pysam.AlignmentFile(bampath, reference_filename=refpath)

    datas = []
    
    # Not sure what the optimum is here - we accumulate tensors until we have at least this many, then process them in batches
    # If this number is too large, we will wait for too long before submitting the next batch to the GPU
    # If its to small, then we end up sending lots of tiny little batches to the GPU, which also isn't efficient
    max_datas = n_region_workers * 2
    n_finished_workers = 0
    max_consecutive_timeouts = 10
    timeouts = 0
    vbuff = []
    max_vbuff_size = 1000 # Max number of variants to buffer
    while True:
        try:
            data = inputq.get(timeout=1) # Timeout is 10 seconds, if we go this long without getting a new object
            timeouts = 0
        except queue.Empty:
            timeouts += 1
            data = None
            logger.info(f"Got a timeout in model queue, have {timeouts} total")
        except FileNotFoundError:
            # This might be some weird bug... occasionally we get FileNotFound errors polling the queue, but they
            # don't seem to have any effect?
            logger.debug("Got a FNF error polling tensor input queue, ignoring it...")
            continue

        if data is not None:
            datas.append(data)

        if (data is None and len(datas)) or len(datas) > max_datas:
            records = call_multi_paths(datas, model, reference, aln, classifier, vcf_template, max_batch_size=max_batch_size)

            # Store the variants in a buffer so we can sort big groups of them (no strong guarantees about sort order for
            # variants coming out of queue)
            vbuff.extend(records)
            if len(vbuff) > max_vbuff_size:
                # Write the variant records to the VCF file
                for var in sorted(vbuff, key=lambda x: x.pos):
                    vcf_out.write(str(var))
                vcf_out.flush()
                vbuff = []

            datas = []

        # vbuff might not be empty
        for var in sorted(vbuff, key=lambda x: x.pos):
            vcf_out.write(str(var))
        vcf_out.flush()

        if timeouts == max_consecutive_timeouts:
            logger.info(f"Found {max_consecutive_timeouts} timeouts, aborting model processing queue")
            break

        if data is None:
            n_finished_workers += 1
            logger.debug(f"Found a stop token, have {n_finished_workers} of {n_region_workers} are done")

        if n_finished_workers == n_region_workers:
            logger.debug(f"All region workers are done, datas length is {len(datas)}, exiting..")
            break

    vcf_out.close()
    logger.info("Calling worker is exiting")


def call_and_merge(batch, batch_offsets, regions, model, reference, max_batch_size):
    """
    Generate haplotypes for the batch, identify variants in each, and then 'merge genotypes' across the overlapping
    windows with the ad-hoc algo in the merge_genotypes function. This also filters out any variants not found
    in the 'regions' tuple

    Note that this function contains additional, unused logic to divide batch up into smaller regions and send those
    through the model individually. This might be useful if different regions have very different numbers of predicted
    tokens, for instance. However, since the time spent in forward passes of the model are almost constant in batch size
    this doesn't offer much speedup in a naive implementation.

    :param batch: Tensor of encoded regions
    :param batch_offsets: List of start positions, must have same length as batch.shape[0]
    :param regions: List of genomic regions, must have length equal to batch_offsets
    :param model: Model for haplotype prediction
    :param reference: pysam.FastaFile with reference genome
    :param max_batch_size: Maximum number of regions to call in one go
    :returns Tuple(List, List) of variants found on each haplotype
    """
    logger.info(f"Predicting batch of size {batch.shape[0]} for chrom {regions[0][0]}:{regions[0][1]}-{regions[-1][2]}")
    dists = np.array([r[2] - bo for r, bo in zip(regions, batch_offsets)])

    # Identify distinct sub-batches for calling. In the future we might populate this with different indexes
    # Setting everything to 0 effectively turns it off for now
    subbatch_idx = np.zeros(len(dists), dtype=int)

    byregion = defaultdict(list)
    max_dist = max(dists)
    # n_output_toks = min(150 // util.TGT_KMER_SIZE - 1, max(dists) // util.TGT_KMER_SIZE + 1)
    for sbi in range(max(subbatch_idx) + 1):
        which = np.where(subbatch_idx == sbi)[0]
        subbatch = batch[torch.tensor(which), :, :, :]
        subbatch_offsets = [b for i,b in zip(subbatch_idx, batch_offsets) if i == sbi]
        subbatch_dists =   [d for i,d in zip(subbatch_idx, dists) if i == sbi]
        subbatch_regions = [r for i,r in zip(subbatch_idx, regions) if i == sbi]
        n_output_toks = min(150 // util.TGT_KMER_SIZE - 1, max_dist // util.TGT_KMER_SIZE + 1)

        logger.debug(f"Sub-batch size: {len(subbatch_offsets)}   max dist: {max(subbatch_dists)},  n_tokens: {n_output_toks}")
        batchvars = call_batch(subbatch, subbatch_offsets, subbatch_regions, model, reference, n_output_toks, max_batch_size=max_batch_size)

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


def collect_phasegroups(vars_hap0, vars_hap1, chrom, aln, reference, minimum_safe_distance=100):
    allkeys = sorted(list(k for k in vars_hap0.keys()) + list(k for k in vars_hap1.keys()), key=lambda x: x[0])

    all_vcf_vars = []
    group0 = defaultdict(list)
    group1 = defaultdict(list)
    prevpos = -1000
    for k in allkeys:
        if k[0] - prevpos > minimum_safe_distance:
            vcf_vars = vcf.vcf_vars(
                vars_hap0=group0,
                vars_hap1=group1,
                chrom=chrom,
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
            prevpos = k[0]
        else:
            if k in vars_hap0:
                group0[k].extend(vars_hap0[k])
            if k in vars_hap1:
                group1[k].extend(vars_hap1[k])

    vcf_vars = vcf.vcf_vars(
        vars_hap0=group0,
        vars_hap1=group1,
        chrom=chrom,
        aln=aln,
        reference=reference
    )
    all_vcf_vars.extend(vcf_vars)
    return all_vcf_vars

def vars_hap_to_records(
    chrom, window_idx, vars_hap0, vars_hap1, aln, reference, classifier_model, vcf_template, var_freq_file=None
):
    """
    Convert variant haplotype objects to variant records
    Performs merging of overlapping variants and collection o phase groups
    """

    # Merging vars can sometimes cause a poor quality variant to clobber a very high quality one, to avoid this
    # we hard-filter out very poor quality variants that overlap other, higher-quality variants
    # This value defines the min qual to be included when merging overlapping variants
    min_merge_qual = 0.01

    vcf_vars = collect_phasegroups(vars_hap0, vars_hap1, chrom, aln, reference, minimum_safe_distance=100)

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


def _call_safe(encoded_reads, model, n_output_toks, max_batch_size, enable_amp=True):
    """
    Predict the sequence for the encoded reads, but dont submit more than 'max_batch_size' samples
    at once
    """
    seq_preds = None
    probs = None
    start = 0
    
    while start < encoded_reads.shape[0]:
        end = start + max_batch_size
        with torch.amp.autocast(device_type='cuda', enabled=enable_amp):
            preds, prbs = util.predict_sequence(encoded_reads[start:end, :, :, :].to(DEVICE), model,
                                            n_output_toks=n_output_toks, device=DEVICE)
        if seq_preds is None:
            seq_preds = preds
        else:
            seq_preds = torch.concat((seq_preds, preds), dim=0)
        if probs is None:
            probs = prbs.detach().cpu().numpy()
        else:
            probs = np.concatenate((probs, prbs.detach().cpu().numpy()), axis=0)
        start += max_batch_size
    return seq_preds, probs


def call_batch(encoded_reads, offsets, regions, model, reference, n_output_toks, max_batch_size):
    """
    Call variants in a batch (list) of regions, by running a forward pass of the model and
    then aligning the predicted sequences to the reference genome and picking out any
    mismatching parts
    :returns : List of variants called in both haplotypes for every item in the batch as a list of 2-tuples
    """
    assert encoded_reads.shape[0] == len(regions), f"Expected the same number of reads as regions, but got {encoded_reads.shape[0]} reads and {len(regions)}"
    assert len(offsets) == len(regions), f"Should be as many offsets as regions, but found {len(offsets)} and {len(regions)}"
    #encoded_reads = encoded_reads.bfloat16()

    seq_preds, probs = _call_safe(encoded_reads, model, n_output_toks, max_batch_size)

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
        calledvars.append((vars_hap0[0:5], vars_hap1[0:5]))

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
    window_start = int(start - 0.7 * window_size)  # We start with regions a bit upstream of the focal / target region
    batch = []
    batch_offsets = []
    readwindow = bam.ReadWindow(aln, chrom, start - 150, end + window_size)
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
            encodedreads = torch.stack(batch, dim=0).cpu()
            returned_count += 1
            yield encodedreads, batch_offsets
            batch = []
            batch_offsets = []

    # Last few
    if batch:
        encodedreads = torch.stack(batch, dim=0).cpu() # Keep encoded tensors on cpu for now
        returned_count += 1
        yield encodedreads, batch_offsets

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

