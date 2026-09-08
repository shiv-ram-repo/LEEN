#!/usr/bin/env python3
"""Document the MMseqs2 train/benchmark similarity filter (answers Reviewer 2, part 1).

Reads the actual .m8 used by protstab_data and reports: threshold, number of rows removed,
and writes the retained/removed index lists so they can be released with the paper.

Note: the .m8 encodes the RESULT of an mmseqs search. The GENERATING command (createdb/search
params, --min-seq-id, --cov-mode, -c coverage) lives in the preprocessing pipeline that produced
mmseq_mut_search_0.25.m8 — locate it and quote it verbatim in the response (this script cannot
recover flags that aren't stored in the .m8 output).

  python document_mmseqs_filter.py --data_root /backup/stability/data_root --out_dir mmseqs_audit
"""
from __future__ import annotations
import argparse, os
import pandas as pd


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--data_root", required=True)
    pa.add_argument("--out_dir", default="mmseqs_audit")
    args = pa.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    csv = os.path.join(args.data_root, "data/dataset/megascale/Tsuboyama2023_Dataset2_Dataset3_20230416.csv")
    m8 = os.path.join(args.data_root, "data/dataset/megascale/mmseq_mut_search_0.25.m8")

    drop_idx = []
    hits = 0
    with open(m8) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                drop_idx.append(int(parts[1])); hits += 1
    drop_idx = sorted(set(drop_idx))

    report = {
        "m8_file": os.path.basename(m8),
        "identity_threshold_from_filename": 0.25,
        "n_hit_lines_in_m8": hits,
        "n_unique_rows_removed": len(drop_idx),
        "applied_to": "non-test splits (train/val); test split is NOT filtered against itself",
        "unit": "per training-set mutation ROW (row index into the megascale CSV), matched against benchmark sequences",
        "note": "The .m8 is standard MMseqs2 tabular output (query, target, pident, ... columns). "
                "Quote the exact generating command (mmseqs createdb/search, --min-seq-id, --cov-mode, -c) "
                "from the preprocessing pipeline in the response — it is not stored in the .m8 itself.",
    }
    pd.Series(drop_idx, name="removed_row_index").to_csv(
        os.path.join(args.out_dir, "mmseqs_removed_indices.csv"), index=False)

    try:
        df = pd.read_csv(csv)
        removed_names = sorted(set(df.iloc[drop_idx]["WT_name"].tolist())) if "WT_name" in df.columns else []
        report["n_removed_unique_WT_names"] = len(removed_names)
        pd.Series(removed_names, name="removed_WT_name").to_csv(
            os.path.join(args.out_dir, "mmseqs_removed_WT_names.csv"), index=False)
    except Exception as e:
        report["csv_note"] = f"could not load CSV for WT_name view: {e}"

    import json
    json.dump(report, open(os.path.join(args.out_dir, "mmseqs_filter_report.json"), "w"), indent=2)
    print(json.dumps(report, indent=2))
    print(f"\nretained/removed lists -> {args.out_dir}/")


if __name__ == "__main__":
    main()
