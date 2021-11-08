import argparse
import logging
import numpy as np
import pandas as pd
import pyranges as pr
import pysam
import sys

logging.basicConfig(format='[%(asctime)s]  %(name)s  %(levelname)s  %(message)s',
                    datefmt='%m-%d %H:%M:%S',
                    level=logging.INFO) # handlers=[RichHandler()])
logger = logging.getLogger(__name__)

DEFAULT_HUMAN_REF = "/Reference/aws/prd/Data/B37/GATKBundle/2.8_subset_arup_v0.1/human_g1k_v37_decoy_phiXAdaptr.fasta"


def get_vtype(row):
    len_ref = len(row.ref)
    alt_count = len(row.alt.split(";"))
    len_alt_max = max(len(alt) for alt in row.alt.split(";"))
    len_alt_min = min(len(alt) for alt in row.alt.split(";"))
    if len_ref == len_alt_max == 1:  # snv
        return "snv"
    elif len_ref == 1 and len_alt_max > 1:  # ins
        return "small_ins_1-24"
    elif len_ref > 1 and len_alt_max == 1:  # del
        return "small_del_1-24"
    else:  # len_ref > 1 and len_alt_max > 1  probably complex but treat like del?
        return "small_del_1-24"


def add_vtype_col(
        input_csv,
        output_csv,
):
    logger.info(f"adding vtype column to labels.csv {input_csv}")
    df_csv = pd.read_csv(input_csv, dtype=dict(chrom=str))


def split_csv(
        input_csv,
        output_csv_1,
        output_csv_2,
        chrs,
):
    logger.info(f"splitting off chrs {', '.join(chrs)} from labels.csv {input_csv}")
    df_csv = pd.read_csv(input_csv, dtype=dict(chrom=str))
    df_out1 = df_csv.query("~chrom.isin(@chrs)")
    df_out2 = df_csv.query("chrom.isin(@chrs)")
    logger.info(f"found {len(df_out1)} vars not in chrs {', '.join(chrs)}")
    logger.info(f"writing vars not in chrs {', '.join(chrs)} to file {output_csv_1}")
    df_out1.to_csv(output_csv_1, index=False)
    logger.info(f"found {len(df_out2)} vars in chrs {', '.join(chrs)}")
    logger.info(f"writing vars in chrs {', '.join(chrs)} to file {output_csv_2}")
    df_out2.to_csv(output_csv_2, index=False)

def get_start_end(row):
    len_ref = len(row.ref)
    len_alt = max([len(alt) for alt in row.alt.split(";")])
    if len_ref == len_alt == 1:  # snv
        return row.pos - 1, row.pos
    elif len_ref == 1 and len_alt > 1:  # ins
        return row.pos, row.pos
    elif len_alt == 1 and len_ref > 1:  # del
        return row.pos, row.pos + len_ref - 1
    elif len_alt > 1 and len_ref > 1:  # probably complex but treat like del
        return row.pos, row.pos + len_ref - 1


def get_ref_base(df, **kwargs):
    fasta = pysam.FastaFile(kwargs["human_ref"])
    return df.apply(lambda row: fasta.fetch(reference=row.Chromosome, start=row.pos - 1, end=row.pos), axis=1)


def build_intersect_pr_beds(bed_paths):
    if not bed_paths:
        raise ValueError("need at least one bed file path to place TN calls")

    pr_beds = [
        pr.PyRanges(
            pd.read_csv(
                path,
                names="Chromosome Start End".split(),
                dtype=dict(Chromosome=str),
                sep="\t",
                usecols=[0, 1, 2])
        )
        for path in bed_paths
    ]
    pr_bed = pr_beds.pop(0)
    for pr_ in pr_beds:
        pr_bed = pr_bed.set_intersect(pr_)
    return pr_bed


def limit_chrs(pr_in, include_chrs=[], exclude_chrs=[]):
    if not include_chrs and not exclude_chrs:
        return pr_in
    if include_chrs:
        logger.info(f"only adding TNs to chromosomes {','.join(include_chrs)}")
        pr_include_chrs = pr.from_dict(dict(
            Chromosome=include_chrs,
            Start=[0] * len(include_chrs),
            End=[1000000] * len(include_chrs)
        ))
        pr_out = pr_in.intersect(pr_include_chrs)
    if exclude_chrs:
        logger.info(f"not adding any TNs to chromosomes {','.join(exclude_chrs)}")
        pr_exclude_chrs = pr.from_dict(dict(
            Chromosome=exclude_chrs,
            Start=[0] * len(exclude_chrs),
            End=[1000000] * len(exclude_chrs)
        ))
        pr_out = pr_in.subtract(pr_exclude_chrs)
    return pr_out




def add_true_negatives(
        input_csv,
        output_csv,
        beds,
        include_chrs=[],
        exclude_chrs=[],
        variant_distance=200,
        max_tn_window=500,
        min_tn_window=200,
        tn_count=1000,
        human_ref=DEFAULT_HUMAN_REF,
        rand_seed=None,
        cpus=1,
        **kwargs,
):
    logger.info(f"cpus = {cpus}")

    # process csv and beds into PyRange objects
    logger.info(f"processing csv labels file {input_csv}")
    df_csv = pd.read_csv(input_csv, dtype=dict(chrom=str))
    df_csv["Chromosome"] = df_csv["chrom"]
    df_csv["Start End".split()] = df_csv.apply(get_start_end, result_type="expand", axis=1)
    pr_csv = pr.PyRanges(df_csv)
    logger.info(f"bed regions for TNs are intersection of the following beds:")
    for bed in beds:
        logger.info(bed)
    pr_bed = build_intersect_pr_beds(beds)
    # remove areas around existing variants from possible TN locations
    logger.info(f"TN min distance from existing calls is {variant_distance}")
    pr_windows = pr_bed.subtract(pr_csv.slack(variant_distance), nb_cpu=cpus)
    # slice up larger regions to max TN window size tiles
    logger.info(f"window tiling size for randomly chosen TN locations is {max_tn_window} bp")
    pr_windows = pr_windows.window(max_tn_window)
    # drop very small windows and excluded windows
    logger.info(f"window min size for randomly chosen TN locations is {min_tn_window} bp")
    pr_windows.interval_len = pr_windows.lengths()
    pr_windows = pr_windows.apply(lambda df, min_tn_window=min_tn_window, **kwargs: df.query(f"interval_len > @min_tn_window"), nb_cpu=cpus, min_tn_window=min_tn_window)
    if exclude_chrs or include_chrs:
        logger.info(f"excluding new TN calls from chromosomes {exclude_chrs}")
        logger.info(f"only including new TN calls from chromosomes {include_chrs}")
        pr_windows = limit_chrs(pr_in=pr_windows, include_chrs=include_chrs, exclude_chrs=exclude_chrs)
    logger.info(f"found {len(pr_windows)} windows for possible TN location")
    # randomly sample desired number of TN windows, one TN at center of each chosen window
    logger.info(f"adding {tn_count} TN calls to csv labels file")
    logger.info(f"using random seed = {rand_seed}")
    if rand_seed:
        np.random.seed(rand_seed)
    pr_windows = pr_windows.sample(tn_count, replace=False)
    # get TN info, pos and ref base
    pr_windows = pr_windows.assign("pos", lambda df: ((df.End + df.Start) / 2).astype(int), nb_cpu=cpus)
    logger.info(f"human ref fastq {human_ref}")
    pr_windows = pr_windows.assign("ref", get_ref_base, nb_cpu=cpus, human_ref=human_ref)
    # build TN entries
    df_fn = pr_windows.df["Chromosome pos ref".split()].rename(columns=dict(Chromosome="chrom"))
    df_fn.insert(0, "vtype", "true_neg")
    df_fn.insert(4, "alt", df_fn.ref)
    df_fn.insert(5, "exp_vaf", 1.0)
    df_fn.insert(6, "ngs_gt", "0/0")
    df_fn.insert(7, "status", "TN")
    df_csv = df_csv.drop(columns="Chromosome Start End".split())
    df_out = pd.concat([df_csv, df_fn])
    df_out.to_csv(output_csv, index=False)


def float_lt_4(input):
    try:
        assert 4 >= float(input) > 0
        return float(input)
    except argparse.ArgumentTypeError:
        print("value must be a float <= 4")


def main(cmd_ln):

    parser = argparse.ArgumentParser(
        description="mod the tp-fp-fn lables.csv files: add TNs, split by chr, add 'vtype' column"
    )
    subparser = parser.add_subparsers()
    tn_parser = subparser.add_parser("tns", help="add tns to lables.csv file")
    tn_parser.add_argument("-i", "--input-csv", required=True, help="input labels.csv file")
    tn_parser.add_argument("-o", "--output-csv", required=True, help="output labels.csv file")
    tn_parser.add_argument("-b", "--beds", nargs="*", default=[],
                        help="bed files to choose TN (reference matching) labels")
    tn_parser.add_argument("-r", "--human-ref", default=DEFAULT_HUMAN_REF, help="path to human ref fasta")
    tn_parser.add_argument("-c", "--tn-count", type=int, default=1000, help="TN rows added to variant label csv")
    tn_parser.add_argument("--include-chrs", nargs="*", default=[], help="only locate new TNs in these chromosomes")
    tn_parser.add_argument("--exclude-chrs", nargs="*", default=[], help="exclude new TNs from these chromosomes")
    tn_parser.add_argument("-vd", "--variant-distance", default=200, help="Buffer zone from tp-fp-fn variants")
    tn_parser.add_argument("-maxw", "--max-tn-window", default=500,
                        help="variant free bed regions broken into this max region size, default 500")
    tn_parser.add_argument("-minw", "--min-tn-window", default=200,
                        help="variant free bed regions discarded if below this size, default 200")
    tn_parser.add_argument("--rand-seed", default=None, type=int, help="optional rand seed for repeatable TN locations")
    tn_parser.add_argument("--cpus", default=1, help="may help for very large variant sets or hurt for small sets")
    tn_parser.set_defaults(func=add_true_negatives)

    split_parser = subparser.add_parser("split", help="split lables.csv file by chr")
    split_parser.add_argument("-i", "--input-csv", required=True, help="input labels.csv file")
    split_parser.add_argument("-o1", "--output-csv-1", required=True, help="output labels.csv without removed chrs")
    split_parser.add_argument("-o2", "--output-csv-2", required=True, help="output labels.csv with only removed chrs")
    split_parser.add_argument("-c", "--chrs", nargs="*", required=True, help="chrs to move to separate file")
    split_parser.set_defaults(func=split_csv)

    vtype_parser = subparser.add_parser("vtype", help="add 'vtype' column to lables.csv file")
    vtype_parser.add_argument("-i", "--input-csv", required=True, help="input labels.csv file")
    vtype_parser.add_argument("-o", "--output-csv", required=True, help="output labels.csv without vtype column")
    vtype_parser.set_defaults(func=add_vtype_col)

    args = parser.parse_args(cmd_ln)
    args.func(**vars(args))


if __name__ == "__main__":
    main(sys.argv[1:])
