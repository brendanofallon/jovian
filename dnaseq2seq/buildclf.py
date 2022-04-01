#!/usr/bin/env python


import sys
import yaml
import numpy as np
import scipy.stats as stats
import pysam
import pickle
import sklearn
from sklearn.ensemble import RandomForestClassifier

import logging
import argparse

logger = logging.getLogger(__name__)
logging.basicConfig(format='[%(asctime)s]  %(name)s  %(levelname)s  %(message)s',
                    datefmt='%m-%d %H:%M:%S',
                    level=logging.INFO)

def var_feats(var):
    feats = []
    feats.append(var.qual)
    feats.append(1 if "PASS" in var.filter else 0)
    feats.append(1 if "LowCov" in var.filter else 0)
    feats.append(1 if "SingleCallHet" in var.filter else 0)
    feats.append(1 if "SingleCallHom" in var.filter else 0)
    feats.append(len(var.ref))
    feats.append(max(len(a) for a in var.alts))
    feats.append(min(var.info['QUALS']))
    feats.append(max(var.info['QUALS']))
    feats.append(sum(var.info['QUALS']))
    feats.append(var.info['WIN_VAR_COUNT'][0])
    feats.append(var.info['WIN_CIS_COUNT'][0])
    feats.append(var.info['WIN_TRANS_COUNT'][0])
    feats.append(var.info['STEP_COUNT'][0])
    feats.append(var.info['CALL_COUNT'][0])
    feats.append(min(var.info['VAR_INDEX']))
    feats.append(max(var.info['VAR_INDEX']))
    feats.append(min(var.info['WIN_OFFSETS']))
    feats.append(max(var.info['WIN_OFFSETS']))
    feats.append(var.samples[0]['DP'])
    return np.array(feats)

def extract_feats(vcf):
    allfeats = []
    for var in pysam.VariantFile(vcf, ignore_truncation=True):
        allfeats.append(var_feats(var))
    return allfeats

def save_model(mdl, path):
    logger.info(f"Saving model to {path}")
    with open(path, 'wb') as fh:
        pickle.dump(mdl, fh)
        
def load_model(path):
    logger.info(f"Loading model from {path}")
    with open(path, 'rb') as fh:
        return pickle.load(fh)

def train_model(conf, threads):
    alltps = []
    allfps = []
    for tpf in conf['tps']:
        alltps.extend(extract_feats(tpf))
    for fpf in conf['fps']:
        allfps.extend(extract_feats(fpf))

    logger.info(f"Loaded {len(alltps)} TP and {len(allfps)} FPs")
    feats = alltps + allfps
    y = np.array([1.0 for _ in range(len(alltps))] + [0.0 for _ in range(len(allfps))])
    clf = RandomForestClassifier(n_estimators=100, max_depth=25, random_state=0, max_features=16, class_weight="balanced", n_jobs=threads)
    clf.fit(feats, y)
    return clf


def predict(model, vcf, **kwargs):
    model = load_model(model)
    vcf = pysam.VariantFile(vcf, ignore_truncation=True)
    print(vcf.header, end='')
    for var in vcf:
        proba = predict_one_record(model, var)
        var.qual = proba
        print(var, end='')


def predict_one_record(loaded_model, var_rec, **kwargs):
    """
    given a loaded model object and a pysam variant record, return classifier quality
    :param loaded_model: loaded model object for classifier
    :param var_rec: single pysam vcf record
    :param kwargs:
    :return: classifier quality
    """
    feats = var_feats(var_rec)
    prediction = loaded_model.predict_proba(feats[np.newaxis, ...])
    return prediction[0, 1]


def train(conf, output, **kwargs):
    logger.info("Loading configuration from {conf_file}")
    conf = yaml.safe_load(open(conf).read())
    model = train_model(conf, threads=kwargs.get('threads'))
    save_model(model, output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--threads", help="Number of threads to use", type=int, default=-1) # -1 means all threads
    
    subparser = parser.add_subparsers()

    trainparser = subparser.add_parser("train", help="Train a new model")
    trainparser.add_argument("-c", "--conf", help="Configuration file")
    trainparser.add_argument("-o", "--output", help="Output path")
    trainparser.set_defaults(func=train)

    predictparser = subparser.add_parser("predict", help="Predict")
    predictparser.add_argument("-m", "--model", help="Model file")
    predictparser.add_argument("-v", "--vcf", help="Input VCF")
    predictparser.set_defaults(func=predict)

    args = parser.parse_args()
    args.func(**vars(args))


if __name__ == "__main__":
    main()

