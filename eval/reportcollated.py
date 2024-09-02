import sys
import pandas as pd
import numpy as np
from rich.table import Table
from rich.text import Text
from rich.console import Console

def get_val(t, metric):
    if len(t) == 0:
        return "?"
    elif len(t) > 1:
        return "X"
    else:
        return t[metric].values[0]

def fmt(x):
    if type(x) == str:
        return x
    elif (type(x) == float) or (type(x) == np.float_):
        return f"{x :.4f}"
    elif type(x) == int or (type(x) == np.int_):
        return str(x)
    else:
        return x

def generate_table(df, samples, tagA, tagB, vtype, subtype="*", subset="*", filter="ALL"):
    table = Table(title=f"{vtype} ({subtype}, {subset}, {filter})")
    table.add_column("sample")
    for metric in ["PPA", "PPV"]:
        table.add_column(f"{metric}-{tagA}")
        table.add_column(f"{metric}-{tagB}")
        table.add_column(f"{metric} diff")

    recalls_A = []
    precisions_A = []
    recalls_B = []
    precisions_B = []
    for sample in samples:
        cols = []
        s = df[(df['sample'] == sample) & (df[' Type'] == vtype) & (df['Subtype'] == subtype) & (df['Subset'] == subset) & (
                    df['Filter'] == filter)]
        cols.append(sample.replace("_tag", ""))
        for metric in ['METRIC.Recall', 'METRIC.Precision']:
            tA = s[s['caller'] == tagA]
            valA = get_val(tA, metric)
            try:
                if metric == 'METRIC.Recall':
                    recalls_A.append(float(valA))
                elif metric == 'METRIC.Precision':
                    precisions_A.append(float(valA))
            except:
                pass
            tB = s[s['caller'] == tagB]
            valB = get_val(tB, metric)
            try:
                if metric == 'METRIC.Recall':
                    recalls_B.append(float(valB))
                elif metric == 'METRIC.Precision':
                    precisions_B.append(float(valB))
            except:
                pass
            try:
                diff = valB - valA
            except:
                diff = np.nan
            if np.isclose(diff, 0.0):
                cols.append(Text(fmt(valA), style='green'))
                cols.append(Text(fmt(valB), style='green'))
                cols.append(Text(f"0.0", style='green'))
            else:
                cols.append(Text(fmt(valA), style='bold red'))
                cols.append(Text(fmt(valB), style='bold red'))
                cols.append(Text(fmt(diff), style='bold red'))


        table.add_row(*cols)
    recall_A = np.mean(recalls_A)
    precision_A = np.mean(precisions_A)
    recall_B = np.mean(recalls_B)
    precision_B = np.mean(precisions_B)
    table.add_row(Text("Avg", style="bold"), Text(fmt(recall_A), style="bold"), Text(fmt(recall_B), style="bold"), Text(fmt(recall_B - recall_A), style="bold"), Text(fmt(precision_A), style="bold"), Text(fmt(precision_B), style="bold"), Text(fmt(precision_B - precision_A), style="bold"))
    return table

def main(collated_csv):
    df = pd.read_csv(collated_csv)
    samples = df['sample'].unique()
    tags = df['caller'].unique()
    assert len(tags) == 2

    print(f"Comparinsg {tags[0]} to {tags[1]} across {len(samples)}")
    console = Console()
    t = generate_table(df, samples, tags[0], tags[1], "INDEL")
    console.print(t)

    t = generate_table(df, samples, tags[0], tags[1], "SNP")
    console.print(t)

if __name__=="__main__":
    main(sys.argv[1])