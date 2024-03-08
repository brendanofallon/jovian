import itertools

import pysam
import torch.multiprocessing as mp
import os
import torch
import datetime
from string import ascii_letters, ascii_uppercase
from call import _encode_region, call_and_merge, vars_hap_to_records, load_model, cluster_positions_for_window

import logging

from dnaseq2seq import buildclf, vcf, util
from call import read_bed_regions, split_large_regions

logger = logging.getLogger(__name__)

logging.basicConfig(format='[%(asctime)s] %(process)d  %(name)s  %(levelname)s  %(message)s',
                    datefmt='%m-%d %H:%M:%S',
                    level=os.environ.get('JV_LOGLEVEL', logging.INFO),
                    handlers=[
                        logging.StreamHandler(),  # Output logs to stdout
                    ])

REGION_STOP_TOKEN = "stop"

def encode_region(bampath, refpath, region, max_read_depth, window_size, min_reads, batch_size, window_step):
    """
    Encode the reads in the given region and save the data along with the region and start offsets to a file
    and return the absolute path of the file

    Somewhat confusingly, the 'region' argument must be a tuple of  (chrom, index, start, end)
    """
    chrom, idx, start, end = region
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


def generate_tensors(region_queue: mp.Queue, output_queue: mp.Queue, bampath, refpath):
    max_read_depth = 150
    window_size = 150
    min_reads = 5
    batch_size = 4
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
            encoded_region_count += 1
            if data is not None:
                output_queue.put(data)
            else:
                logger.warning(f"Whoa, got back None from encode_region")

    logger.info(f"Region worker {os.getpid()} is shutting down after generating {encoded_region_count} encoded regions")

@torch.no_grad()
def call_multi_paths(datas, model, reference, aln, classifier_model, vcf_template, max_batch_size):
    """
    Call variants from the encoded regions and return them as a list of variant records suitable for writing to a VCF file
    No more than max_batch_size are processed in a single batch
    """
    # Accumulate regions until we have at least this many
    min_samples_callbatch = 256

    batch_encoded = []
    batch_start_pos = []
    batch_regions = []
    batch_count = 0
    call_start = datetime.datetime.now()
    window_count = 0
    var_records = []  # Stores all variant records so we can sort before writing

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
            logger.info(f"Calling for block of size {allencoded.shape[0]}")
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
        logger.info(f"Calling for block of size {allencoded.shape[0]}")
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
                                max_batch_size: int):

    model = load_model(modelpath)
    model.eval()
    classifier = buildclf.load_model(classifier_path)

    vcf_header = vcf.create_vcf_header(sample_name="sample", lowcov=20, cmdline="")
    vcf_template = pysam.VariantFile("/dev/null", mode='w', header=vcf_header)

    reference = pysam.FastaFile(refpath)
    aln = pysam.AlignmentFile(bampath, reference_filename=refpath)

    datas = []
    max_datas = 24
    while True:
        data = inputq.get()
        if data is not None:
            logger.debug(f"Model proc found a non-empty item with region {data['region']}")
            datas.append(data)

        if data is None or len(datas) > max_datas:
            records = call_multi_paths(datas, model, reference, aln, classifier, vcf_template, max_batch_size=max_batch_size)
            for record in records:
                print(record)

            datas = []

        if data is None:
            logger.info("Found data stop token, no submitting any more region tensors for calling")
            break

    logger.info("Calling worker is exiting")


def find_regions(regionq, inputbed, bampath, refpath, n_signals):
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

        if idx % 100 == 0:
            logger.info(f"Read {region_count} raw regions with suspect regions: {sus_region_count} tot bp: {sus_region_bp} ")

    logger.info("Done generating sus regions")
    for i in range(n_signals):
        regionq.put(REGION_STOP_TOKEN)

def main():
    mp.set_start_method('spawn')


    modelpath = "models/100M_s28_cont_mapsus_lolr2_epoch2.model"
    classifierpath = "models/s28ce40_bamfix.model"

    #bampath = "/Users/brendan/data/WGS/99702111878_NA12878_1ug.cram"
    bampath = "/data1/brendan/bams/99702111878_NA12878_S89.cram"
    #refpath = "/Users/brendan/data/ref_genome/human_g1k_v37_decoy_phiXAdaptr.fasta.gz"
    refpath = "/data1/brendan/ref/human_g1k_v37_decoy_phiXAdaptr.fasta.gz"

    max_batch_size = 128

    inputbed = "test.bed"


    regions_queue = mp.Queue(maxsize=1024) # Hold BED file regions, generated in main process and sent to 'generate_tensors' workers
    tensors_queue = mp.Queue(maxsize=16) # Holds tensors generated in 'generate tensors' workers, consumed by accumulate_regions_and_call
    #vcfrecs_queue = mp.Queue() # Holds VCF records generated in accumulate_regions_and_call, consumed by main process


    n_region_workers = 16

    # This one processes the input BED file and find 'suspect regions', and puts them in the regions_queue
    region_finder = mp.Process(target=find_regions, args=(regions_queue, inputbed, bampath, refpath, n_region_workers))
    region_finder.start()


    region_workers = [mp.Process(target=generate_tensors,
                        args=(regions_queue, tensors_queue, bampath, refpath))
             for _ in range(n_region_workers)]

    for p in region_workers:
        p.start()

    model_proc = mp.Process(target=accumulate_regions_and_call,
                            args=(modelpath, tensors_queue, refpath, bampath, classifierpath, max_batch_size))
    model_proc.start()



    region_finder.join()
    logger.info("Done loading regions")

    for p in region_workers:
        p.join()
    logger.info("Done generating input tensors")

    model_proc.join()
    logger.info("Done calling variants")

if __name__=="__main__":
    main()
