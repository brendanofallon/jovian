#!/usr/bin/env python


import sys
import yaml
import numpy as np
import scipy.stats as stats
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


def find_var(varfile, chrom, pos, ref, alt):
    """
    Search varfile for VCF record with matching chrom, pos, ref, alt
    If chrom/pos/ref/alt match is found, return that record and allele index of matching alt
    """
    ref, alt = trim_suffix(ref, alt)
    poffset, trimref, trimalt = trim_prefix(ref, alt)
    #print(f"Trimmed query: {pos + poffset} {trimref}  {trimalt}")
    pos = pos + poffset
    for var in varfile.fetch(chrom, pos-5, pos+3):
        #print(f"Comparing to : {var}")
        for i, varalt in enumerate(var.alts):
            varref, varalt = trim_suffix(var.ref, varalt)
            offset, r, a = trim_prefix(varref, varalt)
            #print(f"Comparing to {var.pos + offset}  {r}  {a}")
            if var.pos + offset == pos and r == trimref and a == trimalt:
                return var, i



@lru_cache(maxsize=1000000)
def var_af(varfile, chrom, pos, ref, alt):
    result = find_var(varfile, chrom, pos, ref, alt)
    print(f"Find var result: {result}")
    af = 0.0
    if result is not None:
        var, alt_index = result
        af = var.info['AF'][alt_index]
        print(f"Got AF: {af}")
        if af is None or "PASS" not in var.filter:
            af = 0.0
        else:
            af = float(af)
    return af


def var_feats(var, var_freq_file):
    feats = []
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
            "het"
        ]


def varstr(var):
    return f"{var.chrom}:{var.pos}-{var.ref}-{var.alts[0]}"


def extract_feats(vcf, var_freq_file, fh=None):
    allfeats = []
    for var in pysam.VariantFile(vcf, ignore_truncation=True):
        feats = var_feats(var, var_freq_file)
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

def train_model(conf, threads, var_freq_file, feat_csv=None, labels_csv=None):
    alltps = []
    allfps = []
    var_freq_file = pysam.VariantFile(var_freq_file)
    if feat_csv:
        logger.info("Writing feature dump to {feat_csv}")
        feat_fh = open(feat_csv, "w")
        feat_fh.write("varstr," + ",".join(feat_names()) + "\n")
    else:
        feat_fh = None

    for tpf in conf['tps']:
        alltps.extend(extract_feats(tpf, var_freq_file, feat_fh))
    for fpf in conf['fps']:
        allfps.extend(extract_feats(fpf, var_freq_file, feat_fh))

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


def predict_one_record(loaded_model, var_rec, var_freq_file, **kwargs):
    """
    given a loaded model object and a pysam variant record, return classifier quality
    :param loaded_model: loaded model object for classifier
    :param var_rec: single pysam vcf record
    :param kwargs:
    :return: classifier quality
    """
    feats = var_feats(var_rec, var_freq_file)
    prediction = loaded_model.predict_proba(feats[np.newaxis, ...])
    return prediction[0, 1]


def train(conf, output, **kwargs):
    logger.info("Loading configuration from {conf_file}")
    conf = yaml.safe_load(open(conf).read())
    model = train_model(conf, 
            threads=kwargs.get('threads'), 
            var_freq_file=kwargs.get('freq_file'),
            feat_csv=kwargs.get('feat_csv'),
            labels_csv=kwargs.get('labels_csv'))
    save_model(model, output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--threads", help="Number of threads to use", type=int, default=-1) # -1 means all threads
    
    subparser = parser.add_subparsers()

    trainparser = subparser.add_parser("train", help="Train a new model")
    trainparser.add_argument("-c", "--conf", help="Configuration file")
    trainparser.add_argument("-t", "--threads", help="thread count", default=24, type=int)
    trainparser.add_argument("-o", "--output", help="Output path")
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
    #print(trim_suffix("T", "A"))
    varfile = pysam.VariantFile("test.vcf.gz")
    print(var_af(varfile, "2", 100, "TTG", "TCT"))
    #main()

