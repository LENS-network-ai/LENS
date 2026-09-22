"""
Build LENS survival label lists (train_list_surv.txt / test_list_surv.txt)
from a healnet-style TCGA omics/survival CSV (e.g. tcga_brca_all_clean.csv.zip),
restricted to whichever slides actually have a graph built by
build_survival_graphs.py.

Discretization follows the same convention as
healnet/healnet/etl/loaders.py: quantile bin edges are computed from the
*uncensored* patients' survival_months, then applied to bin everyone
(this is the standard MCAT/PORPOISE discretization).

Output line format (consumed by utils/survival_dataset.SurvivalGraphDataset):
    <slide_id>\\t<y_disc>\\t<censorship>\\t<survival_months>

Usage:
    python preprocessing/build_survival_labels.py \\
        --omic-csv   /path/to/tcga_brca_all_clean.csv.zip \\
        --graphs-dir data/TCGA-BRCA-SURV/simclr_files \\
        --output-dir data/TCGA-BRCA-SURV \\
        --n-bins 4 --subset uncensored
"""

import argparse
import os

import numpy as np
import pandas as pd


def compute_y_disc(df, n_bins, subset, eps=1e-6):
    label_col = 'survival_months'

    if subset == 'all':
        df['y_disc'] = pd.qcut(df[label_col], q=n_bins, labels=False)
        return df

    if subset == 'uncensored':
        subset_df = df[df['censorship'] == 0]
    elif subset == 'censored':
        subset_df = df[df['censorship'] == 1]
    else:
        raise ValueError(f"Unknown subset: {subset}")

    _, q_bins = pd.qcut(subset_df[label_col], q=n_bins, retbins=True, labels=False)
    q_bins[0] = df[label_col].min() - eps
    q_bins[-1] = df[label_col].max() + eps
    df['y_disc'] = pd.cut(df[label_col], bins=q_bins, labels=False, right=False, include_lowest=True)
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--omic-csv', required=True, help='tcga_<cohort>_all_clean.csv.zip')
    parser.add_argument('--graphs-dir', required=True, help='simclr_files dir produced by build_survival_graphs.py')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--n-bins', type=int, default=4)
    parser.add_argument('--subset', default='uncensored', choices=['all', 'uncensored', 'censored'])
    args = parser.parse_args()

    df = pd.read_csv(args.omic_csv, index_col=0, low_memory=False)
    df = df[['case_id', 'slide_id', 'survival_months', 'censorship', 'train']].copy()
    df['censorship'] = df['censorship'].astype(int)

    # slide_id in the CSV keeps the ".svs" extension; graph folders (built from
    # the CLAM patch_features/*.pt filenames) drop it.
    df['graph_id'] = df['slide_id'].str.replace('.svs', '', regex=False)

    available = {d for d in os.listdir(args.graphs_dir)
                 if os.path.isdir(os.path.join(args.graphs_dir, d))}
    before = len(df)
    df = df[df['graph_id'].isin(available)]
    print(f"{len(df)}/{before} CSV rows have a built graph under {args.graphs_dir}")
    if len(df) == 0:
        raise RuntimeError("No overlap between the CSV's slide_id and the built graphs -- "
                            "check that build_survival_graphs.py ran on this cohort.")

    df = compute_y_disc(df, args.n_bins, args.subset)
    df['y_disc'] = df['y_disc'].astype(int)

    os.makedirs(args.output_dir, exist_ok=True)
    train_df = df[df['train'] == 1]
    test_df = df[df['train'] == 0]

    for split_name, split_df in [('train_list_surv.txt', train_df), ('test_list_surv.txt', test_df)]:
        path = os.path.join(args.output_dir, split_name)
        with open(path, 'w') as f:
            for _, row in split_df.iterrows():
                f.write(f"{row['graph_id']}\t{row['y_disc']}\t{row['censorship']}\t{row['survival_months']}\n")
        print(f"Wrote {len(split_df)} rows to {path}")

    print("\nSummary:")
    print(f"  Total usable samples : {len(df)}")
    print(f"  Censored share       : {np.round((df['censorship'] == 1).mean(), 3)}")
    print(f"  Bin sizes (y_disc)   : {dict(df['y_disc'].value_counts().sort_index())}")
    print(f"  Train / test split   : {len(train_df)} / {len(test_df)}")


if __name__ == '__main__':
    main()
