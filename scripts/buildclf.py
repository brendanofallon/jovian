
import sys
import yaml
import numpy as np
import scipy.stats as stats
import pysam
import pickle
import sklearn
from sklearn.ensemble import RandomForestClassifier

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format='[%(asctime)s]  %(name)s  %(levelname)s  %(message)s',
                    datefmt='%m-%d %H:%M:%S',
                    level=logging.INFO)

def extract_feats(vcf):
    allfeats = []
    for var in pysam.VariantFile(vcf, ignore_truncation=True):
        feats = []
        feats.append(var.qual)
        feats.append(1 if "PASS" in var.filter else 0)
        feats.append(1 if "LowCov" in var.filter else 0)
        feats.append(1 if "SingleCallHet" in var.filter else 0)
        feats.append(1 if "SingleCallHom" in var.filter else 0)
        feats.append(min(var.info['QUALS']))
        feats.append(max(var.info['QUALS']))
        feats.append(var.info['WIN_VAR_COUNT'][0])
        feats.append(var.info['WIN_CIS_COUNT'][0])
        feats.append(var.info['WIN_TRANS_COUNT'][0])
        feats.append(var.info['STEP_COUNT'][0])
        feats.append(min(var.info['WIN_OFFSETS']))
        feats.append(max(var.info['WIN_OFFSETS']))
        allfeats.append(np.array(feats))
    return allfeats

def save_model(mdl, path):
    logger.info(f"Saving model to {path}")
    with open(path, 'wb') as fh:
        pickle.dump(mdl, fh)
        
def load_model(path):
    logger.info(f"Loading model from {path}")
    with open(path, 'rb') as fh:
        return pickle.load(fh)

def train_model(conf):
    alltps = []
    for tpf in conf['tps']:
        tps = extract_feats(tpf)
    for fpf in conf['fps']:
        fps = extract_feats(fpf)

    logger.info(f"Loaded {len(tps)} TP and {len(fps)} FPs")
    feats = tps + fps
    y = np.array([1.0 for _ in range(len(tps))] + [0.0 for _ in range(len(fps))])
    clf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=0, max_features=None)
    clf.fit(feats, y)
    return clf


def main(conf_file, output):
    logger.infof("Loading configuration from {conf_file}")
    conf = yaml.safe_load(open(conf_file).read())
    model = train_model(conf)
    save_model(model, output)

if __name__=="__main__":
    main(sys.argv[1], sys.argv[2])