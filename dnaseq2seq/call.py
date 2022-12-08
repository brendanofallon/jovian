"""
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any
later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program.
If not, see <https://www.gnu.org/licenses/>.
"""


import os
import time

import datetime
import logging
import string
import random
from collections import defaultdict
from functools import partial
from tempfile import NamedTemporaryFile as NTFile
from concurrent.futures import ProcessPoolExecutor
from xgboost import XGBClassifier


import torch
import torch.multiprocessing as mp
import pysam
import pandas as pd
import numpy as np

from model import VarTransformer
import buildclf
import vcf
import util
import bam

logger = logging.getLogger(__name__)


DEVICE = torch.device("cpu")

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



def load_model(model_path):
    
    # 35M params
    encoder_attention_heads = 8 # was 4
    decoder_attention_heads = 4 # was 4
    dim_feedforward = 512
    encoder_layers = 8
    decoder_layers = 6 # was 2
    embed_dim_factor = 120 # was 100
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


def chroms_in_bed(bed):
    """
    Extract all unique chromosome names from a BED file.
    :param bed: the BED file
    :return: a list of unique chromosome names in the BED file.
    """
    df = pd.read_csv(bed, sep="\t", comment="#", usecols=[0], dtype=str, header=None)
    return list(df[df.columns[0]].unique())


def bed_for_chrom(chrom, bed, folder="."):
    """
    Extract all the regions defined in the given bed file that are on the
    given collection of chromosome; these regions are saved in a new BED
    file that is returned to the caller.

    :param chrom: a chromosome name
    :param bed: the path to a BED file.
    :param folder: the directory where the new BED file will be saved.

    :return: a BED file that contains all the regions on the given chromosome.
    """
    chrom_bed = NTFile(
        mode="w+t", delete=False, suffix=".bed", dir=folder, prefix=f"chrom_{chrom}."
    )
    with open(bed) as fh:
        for line in fh:
            if not line.startswith("#"):
                contig, start, end = line.split("\t")[:3]
                if contig == chrom:
                    chrom_bed.write(line)

    chrom_bed.close()
    return chrom_bed.name


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

    threads = kwargs.get('threads', 1)
    min_qual = kwargs.get('min_qual', 1e-4)
    logger.info(f"Using {threads} threads")

    var_freq_file = kwargs.get('freq_file')
    if var_freq_file:
        logger.info(f"Loading variant pop frequency file from {var_freq_file}")
    else:
        logger.info(f"No variant frequency file specified, this might not work depending on the classifier requirements")

    tmpdir = f".tmp.varcalls_{randchars()}"
    os.makedirs(tmpdir, exist_ok=False)

    chroms = chroms_in_bed(bed)
    func = partial(bed_for_chrom, bed=bed, folder=tmpdir)
    chrom_beds = [b for b in mp.Pool(threads).map(func, chroms)]

    logger.info(f"The model will be loaded from path {model_path}")

    vcf_file = vcf.init_vcf(
        vcf_out, sample_name="sample", lowcov=20, cmdline=kwargs.get('cmdline')
    )

    for chrom, chrom_bed in zip(chroms, chrom_beds):
        chrom_vcf = call_variants_on_chrom(
            chrom,
            bamfile=bam,
            bed=chrom_bed,
            reference_fasta=reference_fasta,
            model_path=model_path, 
            classifier_path=classifier_path,
            threads=threads,
            tmpdir=tmpdir,
            var_freq_file=var_freq_file,
        )
        for var in pysam.VariantFile(chrom_vcf):
            if var.qual > min_qual:
                vcf_file.write(var)
        os.unlink(chrom_vcf)
        os.unlink(chrom_bed)

    vcf_file.close()
    logger.info(f"All variants are saved to {vcf_out}")

    end_time = time.perf_counter()
    logger.info(f"Total running time of call subcommand is: {end_time - start_time}")


def split_even_chunks(n_item, n_chunk):
    """
    This function splits n_items into n_chunk chunks
    :param n_item : total number of items that will be splitted.
    :param n_chunk: the number of chunks
    :return chunks: the start index and end index of each chunk.
    """
    chunks = []
    chunk_size = n_item // n_chunk
    num_left = n_item - chunk_size * n_chunk
    for i in range(n_chunk):
        chunks.append(chunk_size)
    for i in range(num_left):
        chunks[i] += 1

    start_idx = 0
    end_idx = 0
    for i in range(n_chunk):
        start_idx = end_idx
        end_idx = start_idx + chunks[i]
        yield (start_idx, end_idx)


def call_variants_on_chrom(
    chrom, bamfile, bed, reference_fasta, model_path, classifier_path, threads, tmpdir, var_freq_file
):
    """
    Call variants on the given chromsome and write the variants to a temporary VCF.
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
    window_step = 25
    logger.info(f"Max read depth: {max_read_depth} window step: {window_step}") 
    # Iterate over the input BED file, splitting larger regions into chunks
    # of at most 'max_region_size'

    logger.info(f"Creating windows from {bed}")
    windows = [
        (chrom, window_idx, window_start, window_end) 
        for window_idx, (chrom, window_start, window_end) in enumerate(
            split_large_regions(read_bed_regions(bed), max_region_size=1000)
        )
    ]

    func = partial(
        cluster_positions_for_window,
        bamfile=bamfile,
        reference_fasta=reference_fasta,
        maxdist=100,
    )

    logger.info(f"Generating regions from windows on chromosome {chrom}...")
    region_file = f"{tmpdir}/chrom_{chrom}.regions.txt"
    n_regions = 0
    with open(region_file, "w") as rfh:
        for regions in mp.Pool(threads).map(func, windows):
            for r in regions:
                print(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]}", file=rfh)
                n_regions += 1
    logger.info(f"Generated a total of {n_regions} regions on chromosome {chrom}")

    chunks = [(chrom, si, ei) for si, ei in split_even_chunks(n_regions, threads)]

    process_chunk_func = partial(
        process_chunk_regions,
        region_file=region_file,
        bamfile=bamfile,
        model_path=model_path,
        reference_fasta=reference_fasta,
        classifier_path=classifier_path,
        tmpdir=tmpdir,
        max_read_depth=max_read_depth,
        window_size=150,
        window_step=window_step,
        var_freq_file=var_freq_file
    )

    chrom_vcf = f"{tmpdir}/chrom_{chrom}_raw.vcf"
    vcf_file = vcf.init_vcf(chrom_vcf, sample_name="sample", lowcov=20)
    vcf_file.close()


    chrom_nvar = 0
    with open(chrom_vcf, "a") as chrom_vfh:
        if threads == 1:
            for chunk in chunks:
                chunk_nvar, chunk_vcf = process_chunk_func(chunk)
                for var in pysam.VariantFile(chunk_vcf):
                    chrom_vfh.write(str(var))
                os.unlink(chunk_vcf)
                chrom_nvar += chunk_nvar
        else:
            futs = []
            with ProcessPoolExecutor(max_workers=2) as pool:
                for chunk in chunks:
                    fut = pool.submit(process_chunk_func, chunk)
                    futs.append(fut)
                logger.info(f"Submitted {len(futs)} jobs for variant calling")
                for i, fut in enumerate(futs):
                    try:
                        nvars, chunk_vcf = fut.result(timeout= 10 * 60 * 60)
                        logger.debug(f"Got result {i} of {len(futs)} results for chunked variant calling with {nvars} vars")
                        chrom_nvar += nvars
                        for var in pysam.VariantFile(chunk_vcf):
                            chrom_vfh.write(str(var))
                            chrom_vfh.flush()
                    except Exception as ex:
                        logger.error(f"Exception processing chunk {i} ({chunk}): {ex}")
                        raise ex

                os.unlink(chunk_vcf)

    logger.info(f"Done calling variants on {len(chunks)} chunks, now sorting & deduping raw VCF")
    chrom_vcf_sorted = f"{tmpdir}/chrom_{chrom}_sorted.vcf"
    util.sort_chrom_vcf(chrom_vcf, chrom_vcf_sorted)
    logger.info(f"Done sorting, now deduping..")
    chrom_vcf_dedup = f"{tmpdir}/chrom_{chrom}_dedup.vcf"
    util.dedup_vcf(chrom_vcf_sorted, chrom_vcf_dedup)

    logger.info(
        f"A total of {chrom_nvar} variants called on chromosome {chrom} and saved "
        f"to {chrom_vcf_dedup}"
    )
    os.unlink(region_file)
    os.unlink(chrom_vcf)
    os.unlink(chrom_vcf_sorted)
    
    return chrom_vcf_dedup


def process_chunk_regions(
    chunk,
    region_file,
    bamfile,
    model_path,
    reference_fasta,
    classifier_path,
    max_read_depth,
    tmpdir,
    var_freq_file,
    window_size=300,
    min_reads=5,
    window_step=50,
):
    """
    Call variants on the given region and write variants to a temporary VCF file.
    """
    chrom, start_idx, end_idx = chunk
    chunk_vcf =f"{tmpdir}/chrom_{chrom}.chunk_{start_idx}-{end_idx}.vcf"
    chunk_vfh = vcf.init_vcf(chunk_vcf, sample_name="sample", lowcov=20)
    chunk_vfh.close()
    
    cpname = mp.current_process().name
    logger.debug(
        f"{cpname}: Writing variants to tmp VCF {chunk_vcf}"
    )


    aln = pysam.AlignmentFile(bamfile)
    reference = pysam.FastaFile(reference_fasta)
    model = load_model(model_path)

    if var_freq_file:
        var_freq_file = pysam.VariantFile(var_freq_file)

    # if classifier model available then use classifier quality
    classifier_model = buildclf.load_model(classifier_path) if classifier_path else None

    chunk_nvar = 0
    with open(region_file) as fh, open(chunk_vcf, "a") as chunk_vfh:
        for idx, line in enumerate(fh):
            if idx < start_idx or idx >= end_idx:
                continue
            chrom, window_idx, start, end = line.strip().split("\t")[0:4]
            window_idx, start, end = int(window_idx), int(start), int(end)
            
            logger.debug(f"{cpname}: calling variants in region {chrom}:{start}-{end} (window {window_idx})")
            try:
                chrom, window_idx, vars_hap0, vars_hap1 = _call_vars_region(
                    chrom,
                    window_idx,
                    start,
                    end,
                    aln=aln,
                    model=model,
                    reference=reference,
                    max_read_depth=max_read_depth,
                    window_size=window_size,
                    min_reads=min_reads,
                    window_step=window_step
                )
            except Exception as ex:
                print(f"Error calling variants in {chrom}:{start}-{end} (window {window_idx}) : {ex}")
                raise ex

            try:
                logger.debug(f"{cpname}: Creating variant records for region {chrom}:{start}-{end} (window {window_idx})")
                nvar, vcfname = vars_hap_to_records(
                    chrom,
                    window_idx=window_idx,
                    vars_hap0=vars_hap0,
                    vars_hap1=vars_hap1,
                    aln=aln,
                    reference=reference,
                    classifier_model=classifier_model,
                    tmpdir=tmpdir,
                    var_freq_file=var_freq_file
                )
            except Exception as ex:
                logger.error(f"Error converting variants to VCF records in window {window_idx}: {ex}")
                raise ex

            logger.debug(f"Writing {nvar} variants from chunk {chrom}:{start}-{end} to {chunk_vcf}")
            for var in pysam.VariantFile(vcfname):
                chunk_vfh.write(str(var))
            os.unlink(vcfname)
            chunk_nvar += nvar

    return chunk_nvar, chunk_vcf


def vars_hap_to_records(
    chrom, window_idx, vars_hap0, vars_hap1, aln, reference, classifier_model, tmpdir, var_freq_file
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

    vcfh = NTFile(
        mode="w+t", delete=False, suffix=".vcf", dir=tmpdir, prefix=f"chrom_{chrom}_w{window_idx}."
    )
    vcfh.close()
    vcf_file = vcf.init_vcf(vcfh.name, sample_name="sample", lowcov=20)

    # covert variants to pysam vcf records
    vcf_records = [
        vcf.create_vcf_rec(var, vcf_file)
        for var in sorted(vcf_vars, key=lambda x: x.pos)
    ]

    # write to a chunk VCF
    for rec in vcf_records:
        if classifier_model:
            rec.info["RAW_QUAL"] = rec.qual
            rec.qual = buildclf.predict_one_record(classifier_model, rec, aln, var_freq_file)
        vcf_file.write(rec)

    vcf_file.close()

    return len(vcf_records), vcfh.name


def call_batch(encoded_reads, batch_pos_offsets, model, reference, chrom, window_size, var_retain_window_size):
    """
    Call variants in a batch (list) of regions, by running a forward pass of the model and
    then aligning the predicted sequences to the reference genome and picking out any
    mismatching parts
    :returns : List of variants called in both haplotypes for every item in the batch as a list of 2-tuples
    """
    n_output_toks = 37 # OK, this should be made a little more general...
    seq_preds, probs = util.predict_sequence(encoded_reads.to(DEVICE), model, n_output_toks=n_output_toks, device=DEVICE)
    probs = probs.detach().cpu().numpy()
    calledvars = []
    for b in range(seq_preds.shape[0]):
        hap0_t, hap1_t = seq_preds[b, 0, :, :], seq_preds[b, 1, :, :]
        offset = batch_pos_offsets[b]
        hap0 = util.kmer_preds_to_seq(hap0_t, util.i2s)
        hap1 = util.kmer_preds_to_seq(hap1_t, util.i2s)
        probs0 = np.exp(util.expand_to_bases(probs[b, 0, :]))
        probs1 = np.exp(util.expand_to_bases(probs[b, 1, :]))

        refseq = reference.fetch(chrom, offset, offset + window_size)
        vars_hap0 = list(v for v in vcf.aln_to_vars(refseq, hap0, offset, probs=probs0) if v.pos < offset + var_retain_window_size)
        vars_hap1 = list(v for v in vcf.aln_to_vars(refseq, hap1, offset, probs=probs1) if v.pos < offset + var_retain_window_size)
        splitvars0, splitvars1 = split_overlaps(vars_hap0, vars_hap1)
        calledvars.append((splitvars0, splitvars1))
    return calledvars


def add_ref_bases(encbases, reference, chrom, start, end, max_read_depth):
    """
    Add the reference sequence as read 0
    """
    refseq = reference.fetch(chrom, start, end)
    ref_encoded = bam.string_to_tensor(refseq)
    return torch.cat((ref_encoded.unsqueeze(1), encbases), dim=1)[:, 0:max_read_depth, :]


def _encode_region(aln, reference, chrom, start, end, max_read_depth, window_size=300, min_reads=5, window_step=50, batch_size=64):
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
    readwindow = bam.ReadWindow(aln, chrom, window_start, end + window_size)
    logger.debug(f"Encoding region {chrom}:{start}-{end}")
    while window_start <= (end - 0.1 * window_size):
        logger.debug(f"Window start: {window_start}")
        try:
            enc_reads = readwindow.get_window(window_start, window_start + window_size, max_reads=max_read_depth)
            encoded_with_ref = add_ref_bases(enc_reads, reference, chrom, window_start, window_start + window_size,
                                             max_read_depth=max_read_depth)
            batch.append(encoded_with_ref)
            batch_offsets.append(window_start)
        except bam.LowReadCountException:
            logger.debug(
                f"Bam window {chrom}:{window_start}-{window_start + window_size} "
                f"had too few reads for variant calling (< {min_reads})"
            )

        window_start += window_step

        if len(batch) >= batch_size:
            encodedreads = torch.stack(batch, dim=0).cpu().float()
            yield encodedreads, batch_offsets
            batch = []
            batch_offsets = []

    # Last few
    if batch:
        encodedreads = torch.stack(batch, dim=0).cpu().float() # Keep encoded tensors on cpu for now
        yield encodedreads, batch_offsets


def _call_vars_region(
    chrom, window_idx, start, end, aln, model, reference, max_read_depth, window_size=300, min_reads=5, window_step=50
):
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
    cpname = mp.current_process().name
    logger.debug(
        f"{cpname}: Processing region: {chrom}:{start}-{end} on window {window_idx}"
    )

    step_count = 0  # initialize
    var_retain_window_size = 145
    batch_size = 128

    enctime_total = datetime.timedelta(0)
    calltime_total = datetime.timedelta(0)
    encstart = datetime.datetime.now()
    hap0 = defaultdict(list)
    hap1 = defaultdict(list)
    for batch, batch_offsets in _encode_region(aln, reference, chrom, start, end,
                                               max_read_depth,
                                               window_size=window_size,
                                               min_reads=min_reads,
                                               window_step=window_step,
                                               batch_size=batch_size):
        logger.debug(f"{cpname}: Forward pass for batch with starts {min(batch_offsets)} - {max(batch_offsets)}")
        enctime_total += (datetime.datetime.now() - encstart)
        callstart = datetime.datetime.now()
        batchvars = call_batch(batch, batch_offsets, model, reference, chrom, window_size, var_retain_window_size)

        h0, h1 = merge_genotypes(batchvars)

        for k, v in h0.items():
            hap0[k].extend(v)
        for k, v in h1.items():
            hap1[k].extend(v)
        calltime_total += (datetime.datetime.now() - callstart)

        step_count += batch.shape[0]

    # Only return variants that are actually in the window
    hap0_passing = {k: v for k, v in hap0.items() if start <= v[0].pos <= end}
    hap1_passing = {k: v for k, v in hap1.items() if start <= v[0].pos <= end}

    logger.debug(f"{cpname}: Enc time total: {enctime_total.total_seconds()}  calltime total: {calltime_total.total_seconds()}")
    return chrom, window_idx, hap0_passing, hap1_passing


def vars_dont_overlap(v0, v1):
    """
    True if the two variant do not share any reference bases
    """
    return v0.pos + len(v0.ref) <= v1.pos or v1.pos + len(v1.ref) <= v0.pos


def all_overlaps(vars0, vars1):
    """
    Just do an n^2 search for all variants that overlap, aren't exactly equal, have length > 1, and share
    at least one alt base.
    This is meant to be used within a single calling window, which never really
    more than about 5 total variants, so the n^2 isn't bad
    """
    for v0 in vars0:
        for v1 in vars1:
            if (not vars_dont_overlap(v0, v1)
                    and v1 != v0
                    and (len(v0.ref) > 1 or len(v1.ref) > 1)
                    and any_alt_match(v0, v1)):
                yield v0, v1


def any_alt_match(v0, v1):
    """
    Returns true if any bases in the alt sequence match when aligned
    """
    v0_offset = max(v0.pos, v1.pos) - v0.pos
    v1_offset = max(v0.pos, v1.pos) - v1.pos
    for a, b in zip(v0.alt[v0_offset:], v1.alt[v1_offset:]):
        if a == b:
            return True
    return False




def splitvar(v, pos):
    """
    Split a single vcf.Variant object into two Variants at the given position
    Only pos, ref, and alt are adjusted other fields are copied
    """
    assert v.pos < pos < (v.pos + max(len(v.ref), len(v.alt)))
    a = vcf.Variant(
        pos=v.pos,
        ref=v.ref[0:pos - v.pos],
        alt=v.alt[0:pos - v.pos],
        qual=v.qual,
        hap_model=v.hap_model,
        step=v.step,
        window_offset=v.window_offset,
        var_index=v.var_index,
    )
    b = vcf.Variant(
        pos=pos,
        ref=v.ref[pos - v.pos:],
        alt=v.alt[pos - v.pos:],
        qual=v.qual,
        hap_model=v.hap_model,
        step=v.step,
        window_offset=v.window_offset,
        var_index=v.var_index,
    )
    return a, b


def splitvars(v0, v1):
    """

    v0    |---|
    v1     |---|

    """
    assert not vars_dont_overlap(v0, v1), f"Variants {v0} and {v1} do not overlap, can't split"
    interior_start = max(v0.pos, v1.pos)
    h0vars = []
    h1vars = []
    if v0.pos < interior_start:
        h0a, h0b = splitvar(v0, interior_start)
        h0vars.extend([h0a, h0b])
        if v1.end < v0.end:
            h0vars.remove(h0b)
            h0c, h0d = splitvar(h0b, v1.end)
            h0vars.extend([h0c, h0d])
            h1vars.append(v1)
        elif v1.end == v0.end:
            h1vars.append(v1)
        elif v1.end > v0.end:
            h1a, h1b = splitvar(v1, v0.end)
            h1vars.extend([h1a, h1b])

    elif v1.pos < interior_start:
        h1a, h1b = splitvar(v1, interior_start)
        h1vars.extend([h1a, h1b])
        if v0.end < v1.end:
            h1vars.remove(h1b)
            h1c, h1d = splitvar(h1b, v0.end)
            h1vars.extend([h1c, h1d])
            h0vars.append(v0)
        elif v1.end == v0.end:
            h0vars.append(v0)
        elif v0.end > v1.end:
            h0a, h0b = splitvar(v0, v1.end)
            h0vars.extend([h0a, h0b])

    return h0vars, h1vars


def split_overlaps(h0vars, h1vars):
    """
    Assume h0vars and h1vars are lists of variants on each haplotype
    Find any variants that overlap, and split any that do into chunks
    perfectly aligned with the variants from the other hap
    """
    unused_h0 = [h for h in h0vars]
    unused_h1 = [h for h in h1vars]
    for v0, v1 in all_overlaps(h0vars, h1vars):
        if v0 in unused_h0:
            unused_h0.remove(v0)
        if v1 in unused_h1:
            unused_h1.remove(v1)
        new_h0, new_h1 = splitvars(v0, v1)
        unused_h0.extend(new_h0)
        unused_h1.extend(new_h1)
    return sorted(unused_h0), sorted(unused_h1)


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


