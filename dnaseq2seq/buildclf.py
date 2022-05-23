#!/usr/bin/env python


from enum import Enum
import yaml
import numpy as np
import bisect
import pysam
import pickle
import sklearn
from sklearn.ensemble import RandomForestClassifier

from functools import lru_cache
import logging
import argparse

logger = logging.getLogger(__name__)
logging.basicConfig(format='[%(asctime)s]  %(name)s  %(levelname)s  %(message)s',
                    datefmt='%m-%d %H:%M:%S',
                    level=logging.INFO)


SUPPORTS_REF=0
SUPPORTS_ALT=1
SUPPORTS_OTHER=2


class CigarOperator(Enum):
    BAM_CMATCH = 0  # 'M'
    BAM_CINS = 1  # 'I'
    BAM_CDEL = 2  # 'D'
    BAM_CREF_SKIP = 3  # 'N'
    BAM_CSOFT_CLIP = 4  # 'S'
    BAM_CHARD_CLIP = 5  # 'H'
    BAM_CPAD = 6  # 'P'
    BAM_CEQUAL = 7  # '='
    BAM_CDIFF = 8  # 'X'
    BAM_CBACK = 9  # 'B'

class OffsetOutOfBoundsException(Exception):
    """
    Raised when computing read position offset if we end up with an invalid value
    """
    pass

def trim_suffix(ref, alt):
    r = ref
    a = alt
    while r[-1] == a[-1] and min(len(r), len(a)) > 1:
        r = r[0:-1]
        a = a[0:-1]
    return r, a


def trim_prefix(ref, alt):
    offset = 0
    if len(ref) == 1 or len(alt) == 1:
        return 0, ref, alt
    for r, a in zip(ref[:-1], alt[:-1]):
        if r != a:
            return offset, ref[offset:], alt[offset:]
        offset += 1
    return offset, ref[offset:], alt[offset:]


def full_prefix_trim(ref, alt):
    """
    Trim prefix, leaving empty alleles if necessary
    """
    for i, (r, a) in enumerate(zip(ref + "$", alt + "$")):
        if r != a:
            break
    return i, ref[i:], alt[i:]


def find_var(varfile, chrom, pos, ref, alt):
    """
    Search varfile for VCF record with matching chrom, pos, ref, alt
    If chrom/pos/ref/alt match is found, return that record and allele index of matching alt
    """
    ref, alt = trim_suffix(ref, alt)
    poffset, trimref, trimalt = trim_prefix(ref, alt)
    pos = pos + poffset
    for var in varfile.fetch(chrom, pos-5, pos+3):
        for i, varalt in enumerate(var.alts):
            varref, varalt = trim_suffix(var.ref, varalt)
            offset, r, a = trim_prefix(varref, varalt)
            if var.pos + offset == pos and r == trimref and a == trimalt:
                return var, i



@lru_cache(maxsize=1000000)
def var_af(varfile, chrom, pos, ref, alt):
    result = find_var(varfile, chrom, pos, ref, alt)
    af = 0.0
    if result is not None:
        var, alt_index = result
        af = var.info['AF'][alt_index]
        if af is None or "PASS" not in var.filter:
            af = 0.0
        else:
            af = float(af)
    return af

def get_query_pos(read, ref_pos):
    """
    Returns the offset in the read sequence corresponding to the given reference position. Follows the
    rules given by bisect.bisect_left, so if multiple read reference positions correspond to the ref_pos,
    then the index of the leftmost is returned.
    :param read: pysam.AlignedSegment
    :param ref_pos: Reference genomic coordinate
    :return: Index in read.get_reference_positions() that corresponds to ref_pos
    """
    ref_mapping = read.get_reference_positions()
    if not ref_mapping:
        raise OffsetOutOfBoundsException('ref_mapping is empty for read {}'.format(read))
    offset = bisect.bisect_left(ref_mapping, ref_pos)
    if offset > len(ref_mapping):
        raise OffsetOutOfBoundsException('offset {} > len(ref_mapping): {}'.format(offset, len(ref_mapping)))
    elif ref_pos < ref_mapping[0]:
        raise OffsetOutOfBoundsException('ref_pos {} < ref_mapping[0] {}'.format(ref_pos, ref_mapping[0]))

    return offset

def cigop_at_offset(read, offset):
    """
    Return the cigar operation / length at the given offset in the read
    """
    if read.cigartuples is None:
        raise ValueError("Yikes, cigartuples is none!")
    for cigop, length in read.cigartuples:
        if offset <= length:
            return cigop, length
        else:
            offset -= length
    return -1, -1


def read_support(read, offset, ref, alt):
    """
    Returns True if the given read appears to support the given alt allele
    Expects TRIMMED ref + alt variants as input
    :return: 0 if read supports ref allele, 1 if read supports alt, 2 oth
    """
    if len(ref) == 1 and len(alt) == 1:
        base = read.query_sequence[offset]
        if base == ref:
            return SUPPORTS_REF
        elif base == alt:
            return SUPPORTS_ALT
        else:
            return SUPPORTS_OTHER
    elif len(ref) > 0 and len(alt) == 0:
        # Deletion...
        cigop, length = cigop_at_offset(read, offset+1)
        if cigop == CigarOperator.BAM_CDEL.value and length == len(ref):
            return SUPPORTS_ALT
        elif cigop == CigarOperator.BAM_CDEL.value and length != len(ref):
            return SUPPORTS_OTHER
        else:
            return SUPPORTS_REF
    elif len(ref) == 0 and len(alt) > 0:
        # Insertion...
        cigop, length = cigop_at_offset(read, offset + 1)
        if cigop == CigarOperator.BAM_CINS.value and length == len(alt):
            return SUPPORTS_ALT
        elif cigop == CigarOperator.BAM_CINS.value and length != len(alt):
            return SUPPORTS_OTHER
        else:
            return SUPPORTS_REF
    else:
        raise ValueError("Cant handle multinucleotide vars yet (ref={}, alt={})".format(ref, alt))


def bamfeats(var, aln):
    tot_reads = 0
    pos_ref = 0
    pos_alt = 0
    neg_ref = 0
    neg_alt = 0
    min_mq = 10
    highmq_ref = 0
    highmq_alt = 0
    pos_offset, trimref, trimalt = full_prefix_trim(var.ref, var.alts[0])
    if len(trimref) == 0 and len(trimalt) == 0:
        return 0, 0, 0, 0, 0, 0, 0

    varstart = var.start + pos_offset
    for read in aln.fetch(var.chrom, var.start-1, var.start+1):
        try:
            offset = get_query_pos(read, varstart)
            support_val = read_support(read, offset, trimref, trimalt)
            tot_reads += 1
        except:
            continue

        if read.is_forward:
            if support_val == SUPPORTS_REF:
                pos_ref += 1
            elif support_val == SUPPORTS_ALT:
                pos_alt += 1
        else:
            if support_val == SUPPORTS_REF:
                neg_ref += 1
            elif support_val == SUPPORTS_ALT:
                neg_alt += 1
        if read.mapping_quality > min_mq:
            if support_val == SUPPORTS_REF:
                highmq_ref += 1
            if support_val == SUPPORTS_ALT:
                highmq_alt += 1

    return tot_reads, pos_ref, pos_alt, neg_ref, neg_alt, highmq_ref, highmq_alt



def var_feats(var, aln, var_freq_file):
    feats = []
    bamreads, pos_ref, pos_alt, neg_ref, neg_alt, highmq_ref, highmq_alt = bamfeats(var, aln)
    if bamreads > 0:
        vaf = (pos_alt + neg_alt) / (pos_ref + neg_ref + pos_alt + neg_alt)
    else:
        vaf = 0.0
    if (highmq_ref + highmq_alt) > 0:
        mq_vaf = (highmq_alt) / (highmq_alt + highmq_ref)
    else:
        mq_vaf = 0.0
    feats.append(var.qual)
    feats.append(len(var.ref))
    feats.append(max(len(a) for a in var.alts))
    feats.append(min(var.info['QUALS']))
    feats.append(max(var.info['QUALS']))
    feats.append(var.info['WIN_VAR_COUNT'][0])
    feats.append(var.info['WIN_CIS_COUNT'][0])
    feats.append(var.info['WIN_TRANS_COUNT'][0])
    feats.append(var.info['STEP_COUNT'][0])
    feats.append(var.info['CALL_COUNT'][0])
    feats.append(min(var.info['WIN_OFFSETS']))
    feats.append(max(var.info['WIN_OFFSETS']))
    feats.append(var.samples[0]['DP'])
    feats.append(var_af(var_freq_file, var.chrom, var.pos, var.ref, var.alts[0]))
    feats.append(1 if 0 in var.samples[0]['GT'] else 0)
    feats.append(vaf)
    feats.append(mq_vaf)
    feats.append(pos_alt + neg_alt)
    return np.array(feats)


def feat_names():
    return [
            "qual",
            "ref_len",
            "alt_len",
            "min_qual",
            "max_qual",
            "var_count",
            "cis_count",
            "trans_count",
            "step_count",
            "call_count",
            "min_win_offset",
            "max_win_offset",
            "dp",
            "af",
            "het",
            "vaf",
            "highmq_vaf",
            "altreads",
        ]


def varstr(var):
    return f"{var.chrom}:{var.pos}-{var.ref}-{var.alts[0]}"


def extract_feats(vcf, aln, var_freq_file, fh=None):
    allfeats = []
    for var in pysam.VariantFile(vcf, ignore_truncation=True):
        feats = var_feats(var, aln, var_freq_file)
        if fh:
            fstr = ",".join(str(x) for x in feats)
            fh.write(f"{varstr(var)},{fstr}\n")
        allfeats.append(feats)
    return allfeats


def save_model(mdl, path):
    logger.info(f"Saving model to {path}")
    with open(path, 'wb') as fh:
        pickle.dump(mdl, fh)
        

def load_model(path):
    logger.info(f"Loading model from {path}")
    with open(path, 'rb') as fh:
        return pickle.load(fh)


def train_model(conf, threads, var_freq_file, feat_csv=None, labels_csv=None, reference_filename=None):
    alltps = []
    allfps = []
    var_freq_file = pysam.VariantFile(var_freq_file)
    if feat_csv:
        logger.info("Writing feature dump to {feat_csv}")
        feat_fh = open(feat_csv, "w")
        feat_fh.write("varstr," + ",".join(feat_names()) + "\n")
    else:
        feat_fh = None

    for sample in conf.keys():
        aln = pysam.AlignmentFile(sample['bam'], reference_filename=reference_filename)
        if 'tps' in sample:
            alltps.extend(extract_feats(sample['tps'], aln, var_freq_file, feat_fh))
        if 'fps' in sample:
            allfps.extend(extract_feats(sample['fps'], aln, var_freq_file, feat_fh))

    if feat_fh:
        feat_fh.close()

    logger.info(f"Loaded {len(alltps)} TP and {len(allfps)} FPs")
    feats = alltps + allfps
    y = np.array([1.0 for _ in range(len(alltps))] + [0.0 for _ in range(len(allfps))])
    if labels_csv:
        with open(labels_csv, "w") as fh:
            fh.write("label\n") 
            for val in y:
                fh.write(str(val) + "\n")

    clf = RandomForestClassifier(n_estimators=100, max_depth=25, random_state=0, max_features=None, class_weight="balanced", n_jobs=threads)
    clf.fit(feats, y)
    return clf


def predict(model, vcf, **kwargs):
    model = load_model(model)
    vcf = pysam.VariantFile(vcf, ignore_truncation=True)
    if kwargs.get('freq_file'):
        var_freq_file = pysam.VariantFile(kwargs.get('freq_file'))
    else:
        var_freq_file = None
    print(vcf.header, end='')
    for var in vcf:
        proba = predict_one_record(model, var, var_freq_file)
        var.qual = proba
        print(var, end='')


def predict_one_record(loaded_model, var_rec, aln, var_freq_file, **kwargs):
    """
    given a loaded model object and a pysam variant record, return classifier quality
    :param loaded_model: loaded model object for classifier
    :param var_rec: single pysam vcf record
    :param kwargs:
    :return: classifier quality
    """
    feats = var_feats(var_rec, aln, var_freq_file)
    prediction = loaded_model.predict_proba(feats[np.newaxis, ...])
    return prediction[0, 1]


def train(conf, output, **kwargs):
    logger.info("Loading configuration from {conf_file}")
    conf = yaml.safe_load(open(conf).read())
    model = train_model(conf, 
            threads=kwargs.get('threads'), 
            var_freq_file=kwargs.get('freq_file'),
            feat_csv=kwargs.get('feat_csv'),
            labels_csv=kwargs.get('labels_csv'),
            reference_filename=kwargs.get('reference'))
    save_model(model, output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--threads", help="Number of threads to use", type=int, default=-1) # -1 means all threads
    
    subparser = parser.add_subparsers()

    trainparser = subparser.add_parser("train", help="Train a new model")
    trainparser.add_argument("-c", "--conf", help="Configuration file")
    trainparser.add_argument("-t", "--threads", help="thread count", default=24, type=int)
    trainparser.add_argument("-o", "--output", help="Output path")
    trainparser.add_argument("-r", "--reference", help="Reference genome fasta")
    trainparser.add_argument("-f", "--freq-file", help="Variant frequency file (Gnomad or similar)")
    trainparser.add_argument("--feat-csv", help="Feature dump CSV")
    trainparser.add_argument("--labels-csv", help="Label dump CSV")
    trainparser.set_defaults(func=train)

    predictparser = subparser.add_parser("predict", help="Predict")
    predictparser.add_argument("-m", "--model", help="Model file")
    predictparser.add_argument("-f", "--freq-file", help="Variant frequency file (Gnomad or similar)")
    predictparser.add_argument("-v", "--vcf", help="Input VCF")
    predictparser.set_defaults(func=predict)

    args = parser.parse_args()
    args.func(**vars(args))


if __name__ == "__main__":
    main()
    # for a,b in zip("ABC", "ABCDEF"):
    #     print(f"{a},{b}")
    # print(pysam.__version__)
    # aln = pysam.AlignmentFile("/Volumes/Share/genomics/WGS/99702111878_NA12878_1ug_chr22.bam")
    # vcf = pysam.VariantFile("/Volumes/Share/genomics/variantcalling/test.vcf")
    # var = next(vcf)
    # x = bamfeats(var, aln)
    # print(x)
