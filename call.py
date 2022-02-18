import datetime
import logging
from collections import defaultdict
import random

import torch
import pysam
import pyranges as pr

from model import VarTransformer
import vcf
import util
import bam

logger = logging.getLogger(__name__)


DEVICE = torch.device("cuda") if hasattr(torch, 'cuda') and torch.cuda.is_available() else torch.device("cpu")


class LowReadCountException(Exception):
    """
    Region of bam file has too few spanning reads for variant detection
    """
    pass


def gen_suspicious_spots(aln, chrom, start, stop, refseq):
    """
    Generator for positions of a BAM / CRAM file that may contain a variant. This should be pretty sensitive and
    trigger on anything even remotely like a variant
    This uses the pysam Pileup 'engine', which seems less than ideal but at least it's C code and is probably
    fast. May need to update to something smarter if this
    :param aln: pysam.AlignmentFile with reads
    :param chrom: Chromosome containing region
    :param start: Start position of region
    :param stop: End position of region (exclusive)
    :param refseq: Reference sequence of position
    """
    assert len(refseq) == stop - start, f"Ref sequence length doesn't match start - stop coords"
    for col in aln.pileup(chrom, start=start, stop=stop, stepper='nofilter'):
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


def bed_to_windows(pr_bed, bed_slack=0, window_spacing=1000, window_overlap=0):
    """
    Make generator yielding windows of spacing window_spacing with right side overlap window_overlap
    Windows will typically be smaller than window_spacing at right end of bed intervals
    Also return total window count (might indicate how long it could take?)
    :param pr_bed: PyRange object representing bed file (columns Chromosome, Start, End)
    :param bed_slack: bases to slack both sides of each bed region
    :param window_spacing: spacing between the start of each window
    :param window_overlap: right side overlap between windows
    :return: yields Chromosome, Start, End of window
    """
    # merge and slack/pad bed file regions
    pr_slack = pr_bed.slack(bed_slack)
    df_windows = pr_slack.window(window_spacing).df
    df_windows["End"] = df_windows["End"] + window_overlap
    df_windows = pr.PyRanges(df_windows).intersect(pr_slack).df

    window_count = len(df_windows)
    windows = ((win.Chromosome, win.Start, win.End) for i, win in df_windows.iterrows())
    return windows, window_count


def reconcile_current_window(prev_win, current_win):
    """
    modify variant parameters in current window depending on any overlapping variants in previous window
    :param prev_win: variant dict for previous window (to left of current)
    :param current_win: variant dict for current window (most recent variants called)
    :return: modified variant dict for current window
    """
    overlap_vars = set(prev_win) & set(current_win)

    # swap haplotypes if supported by previous window
    same_hap_var_count, opposite_hap_var_count = 0, 0
    for v in overlap_vars:
        if prev_win[v].het and current_win[v].het:
            if prev_win[v].haplotype == current_win[v].haplotype:
                same_hap_var_count += 1
            else:
                opposite_hap_var_count += 1
    if opposite_hap_var_count > same_hap_var_count:  # swap haplotypes
        for k, v in current_win.items():
            current_win[k].genotype = tuple(reversed(current_win[k].genotype))
            if v.het and v.haplotype == 0:
                v.haplotype = 1
            elif v.het and v.haplotype == 1:
                v.haplotype = 0

    for var in overlap_vars:
        # if hom in both windows
        #   - just mark as DUPLICATE
        if not prev_win[var].het and not current_win[var].het:
            current_win[var].duplicate = True
        # if het in both windows and same genotype order ( 0|1 or 1|0 )
        #   - change phase set (PS) of current window to previous window
        #   - mark var as DUPLICATE in current window
        if prev_win[var].het and current_win[var].het and prev_win[var].genotype == current_win[var].genotype:
            current_win[var].duplicate = True
            for v in current_win:
                current_win[v].phase_set = prev_win[var].phase_set
        # if het in both windows and different haplotype (hap0 or hap1)
        #   - change phase set (PS) of current window to prev window
        #   - mark var as DUPLICATE in current window
        #   - reverse genotype of all current window vars (i.e., (0,1) to (1,0))
        if prev_win[var].het and current_win[var].het and prev_win[var].genotype != current_win[var].genotype:
            current_win[var].duplicate = True
    return current_win


def load_model(model_path):
    logger.info(f"Loading model from path {model_path}")
    attention_heads = 4
    encoder_layers = 6
    transformer_dim = 200
    embed_dim_factor = 125
    model = VarTransformer(read_depth=100,
                           feature_count=9,
                           out_dim=4,
                           embed_dim_factor=embed_dim_factor,
                           nhead=attention_heads,
                           d_hid=transformer_dim,
                           n_encoder_layers=encoder_layers,
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
            yield toks[0], int(toks[1]), int(toks[2])


def split_large_regions(regions, max_region_size):
    """
    Split any regions greater than max_region_size into regions smaller than max_region_size
    """
    for chrom, start, end in regions:
        while start < end:
            yield chrom, start, min(end, start + max_region_size)
            start += max_region_size


def cluster_positions(poslist, maxdist=500):
    """
    Iterate over the given list of positions (numbers), and generate ranges containing
    positions not greater than 'maxdist' in size
    """
    cluster = []
    for pos in poslist:
        if len(cluster) == 0 or pos - min(cluster) < maxdist:
            cluster.append(pos)
        else:
            yield min(cluster), max(cluster)
            cluster = [pos]

    if len(cluster) == 1:
        yield cluster[0] - 10, cluster[0] + 10
    elif len(cluster) > 1:
        yield min(cluster), max(cluster)


def call(model_path, bam, bed, reference_fasta, vcf_out, bed_slack=0, window_spacing=1000, window_overlap=0, **kwargs):
    """
    Use model in statedict to call variants in bam in genomic regions in bed file.
    Steps:
      1. build model
      2. break bed regions into windows with start positions determined by window_spacing and end positions
         determined by window_overlap (the last window in each bed region will likely be shorter than others)
      3. call variants in each window
      4. join variants after searching for any duplicates
      5. save to vcf file
    :param statedict:
    :param bam:
    :param bed:
    :param reference_fasta:
    :param vcf_out:
    :param bed_slack:
    :param window_spacing:
    :param window_overlap:
    :param kwargs:
    :return:
      """
    max_read_depth = 100
    logger.info(f"Found torch device: {DEVICE}")

    model = load_model(model_path)
    reference = pysam.FastaFile(reference_fasta)
    aln = pysam.AlignmentFile(bam, reference_filename=reference_fasta)

    vcf_file = vcf.init_vcf(vcf_out, sample_name="sample", lowcov=30)

    totbases = util.count_bases(bed)
    bases_processed = 0 
    #  Iterate over the input BED file, splitting larger regions into chunks of at most 'max_region_size'
    for i, (chrom, window_start, window_end) in enumerate(split_large_regions(read_bed_regions(bed), max_region_size=1000)):
        prog = bases_processed / totbases
        logger.info(f"Processing {chrom}:{window_start}-{window_end}   [{100 * prog :.2f}%]")
        refseq = reference.fetch(chrom, window_start, window_end)

        # Search the region for positions that may contain a variant, and cluster those into ranges
        # For each range, run the variant calling procedure in a sliding window
        for start, end in cluster_positions(gen_suspicious_spots(pysam.AlignmentFile(bam, reference_filename=reference_fasta), chrom, window_start, window_end, refseq), maxdist=500):
            logger.info(f"Running model for {start}-{end} ({end - start} bp) inside {window_start}-{window_end}")
            vars_hap0, vars_hap1 = _call_vars_region(aln, model, reference,
                                                     chrom, start-3, end+2,
                                                     max_read_depth,
                                                     window_size=300,
                                                     window_step=33)

            vcf_vars = vcf.vcf_vars(vars_hap0=vars_hap0, vars_hap1=vars_hap1, chrom=chrom, window_idx=i, aln=aln,
                             reference=reference)
            vcf.vars_to_vcf(vcf_file, sorted(vcf_vars, key=lambda x: x.pos))

        bases_processed += window_end - window_start
    vcf_file.close()


def call_batch(batch, batch_pos_offsets, model, reference, chrom, window_size):
    """
    Call variants in a batch (list) of regions, by running a forward pass of the model and
    then aligning the predicted sequences to the reference genome and picking out any
    mismatching parts
    :returns : List of variants called in both haplotypes for every item in the batch as a list of 2-tuples
    """
    logger.info(f"Forward pass of batch with size {len(batch)}")
    encodedreads = torch.stack(batch, dim=0).to(DEVICE).float()
    seq_preds = model(encodedreads)
    calledvars = []
    for b in range(seq_preds.shape[0]):
        hap0_t, hap1_t = seq_preds[b, 0, :, :], seq_preds[b, 1, :, :]
        offset = batch_pos_offsets[b]
        hap0 = util.readstr(hap0_t)
        hap1 = util.readstr(hap1_t)
        hap0_probs = hap0_t.detach().cpu().numpy().max(axis=-1)
        hap1_probs = hap1_t.detach().cpu().numpy().max(axis=-1)

        refseq = reference.fetch(chrom, offset, offset + window_size)
        vars_hap0 = list(vcf.aln_to_vars(refseq, hap0, offset, hap0_probs))
        vars_hap1 = list(vcf.aln_to_vars(refseq, hap1, offset, hap1_probs))
        calledvars.append((vars_hap0, vars_hap1))
    return calledvars


def update_batchvars(allvars0, allvars1, batchvars, batch_offsets, step_count, window_retain_size=150):
    for window_start, (vars_hap0, vars_hap1) in zip(batch_offsets, batchvars):
        stepvars0 = {}
        for v in vars_hap0:
            if v.pos < (window_start + window_retain_size):
                v.hap_model = 0
                v.step = step_count
                stepvars0[(v.pos, v.ref, v.alt)] = v
        stepvars1 = {}
        for v in vars_hap1:
            if v.pos < (window_start + window_retain_size):
                v.hap_model = 1
                v.step = step_count
                stepvars1[(v.pos, v.ref, v.alt)] = v

        # swap haplotypes if supported by previous vars
        same_hap_var_count = sum(len(allvars0[v]) for v in stepvars0 if v in allvars0)
        same_hap_var_count += sum(len(allvars1[v]) for v in stepvars1 if v in allvars1)
        opposite_hap_var_count = sum(len(allvars1[v]) for v in stepvars0 if v in allvars1)
        opposite_hap_var_count += sum(len(allvars0[v]) for v in stepvars1 if v in allvars0)
        if opposite_hap_var_count > same_hap_var_count:  # swap haplotypes
            stepvars1, stepvars0 = stepvars0, stepvars1

        # add this step's vars to allvars
        [allvars0[key].append(v) for key, v in stepvars0.items()]
        [allvars1[key].append(v) for key, v in stepvars1.items()]
        step_count = step_count + 1

    return allvars0, allvars1


def add_ref_bases(encbases, reference, chrom, start, end, max_read_depth):
    """
    Add the reference sequence as read 0
    """
    refseq = reference.fetch(chrom, start, end)
    ref_encoded = bam.string_to_tensor(refseq)
    return torch.cat((ref_encoded.unsqueeze(1), encbases), dim=1)[:, 0:max_read_depth, :]


def _call_vars_region(aln, model, reference, chrom, start, end, max_read_depth, window_size=300, min_reads=5, window_step=50):
    """
    For the given region, identify variants by repeatedly calling the model over a sliding window,
    tallying all of the variants called, and passing back all call and repeat count info
    for further exploration
    Currently:
    - exclude all variants in the downstream half of the window
    - retain all remaining var calls noting how many time each one was called, qualities, etc.
    - call with no repeats are mostly false positives but they are retained
    - haplotype 0 and 1 for each step are set by comparing with repeat vars from previous steps

    TODO:
      - add prob info from alt sequence to vars?
      - add depth derived from tensor to vars?
      - create new prob from all duplicate calls?
    """
    var_retain_window_size = 150
    allvars0 = defaultdict(list)
    allvars1 = defaultdict(list)
    window_start = start - 2 * window_step # We start with regions a bit upstream of the focal / target region
    step_count = 0  # initialize
    batch_size = 32
    batch = []
    batch_offsets = []
    readwindow = bam.ReadWindow(aln, chrom, window_start, end + window_size)
    enctime_total = 0
    calltime_total = 0
    while window_start <= (end - window_step):
        logger.info(f"Window start..end: {window_start} - {window_start + window_size}")
        # Generate encoded reads and group them into batches for faster forward passes
        encstart = datetime.datetime.now()
        try:
            enc_reads = readwindow.get_window(window_start, window_start + window_size, max_reads=max_read_depth)
            encoded_with_ref = add_ref_bases(enc_reads, reference, chrom, window_start, window_start + window_size, max_read_depth=max_read_depth)
            batch.append(encoded_with_ref)
            batch_offsets.append(window_start)
        except LowReadCountException:
            logger.debug(
                f"Bam window {chrom}:{window_start}-{window_start + window_size} "
                f"had too few reads for variant calling (< {min_reads})"
            )
        enctime_total += (datetime.datetime.now() - encstart)
        callstart = datetime.datetime.now()
        if len(batch) > batch_size:
            batchvars = call_batch(batch, batch_offsets, model, reference, chrom, window_size)
            batch = []
            batch_offsets = []
            allvars0, allvars1 = update_batchvars(allvars0, allvars1, batchvars, batch_offsets, step_count - len(batch), var_retain_window_size)
        calltime_total += (datetime.datetime.now() - callstart)
        # continue
        window_start += window_step
        step_count += 1

    # Process any remaining items
    if batch:
        callstart = datetime.datetime.now()
        batchvars = call_batch(batch, batch_offsets, model, reference, chrom, window_size)
        allvars0, allvars1 = update_batchvars(allvars0, allvars1, batchvars, batch_offsets, step_count - len(batch),
                                              var_retain_window_size)
        calltime_total += (datetime.datetime.now() - callstart)
    # Only return variants that are actually in the window
    hap0_passing = {k: v for k, v in allvars0.items() if start <= v[0].pos <= end}
    hap1_passing = {k: v for k, v in allvars1.items() if start <= v[0].pos <= end}
    #logger.warning(f"Only returning variants in {start} - {end}")
    logger.info(f"Enc time total: {enctime_total.total_seconds()}  calltime total: {calltime_total.total_seconds()}")
    return hap0_passing, hap1_passing

