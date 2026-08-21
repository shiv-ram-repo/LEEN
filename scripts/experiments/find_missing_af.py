#!/usr/bin/env python3
"""Identify benchmark proteins still lacking an AlphaFold model, and write a clean re-fold FASTA.

Handles the two ColabFold failure causes seen in the log:
  - MSA server timeouts (just need to re-fold the missing ones, ideally in small batches)
  - 'Invalid character in the sequence: -'  (sequences with gap dashes) -> replaced with 'X'

  PYTHONPATH=/backup/new_work python find_missing_af.py \
    --data_root /backup/stability/data_root \
    --af_dir /backup/stability/data_root/data/dataset/af_benchmarks \
    --benchmarks s669 s461 --out_fasta missing_af.fasta
"""
import argparse, glob, os, re
from protstab_data import MegaScaleTestDatasets

def af_ids(af_dir):
    ids = set()
    for f in glob.glob(os.path.join(af_dir, "*.pdb")):
        base = os.path.basename(f)
        pid = re.split(r"_(?:un)?relaxed_rank_\d+", base)[0].lower()
        ids.add(pid)
    return ids

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--data_root", required=True)
    pa.add_argument("--af_dir", required=True)
    pa.add_argument("--benchmarks", nargs="+", default=["s669","s461"])
    pa.add_argument("--out_fasta", default="missing_af.fasta")
    args = pa.parse_args()

    have = af_ids(args.af_dir)
    coll = dict(MegaScaleTestDatasets(data_root=args.data_root).iter_named())
    missing, total, dash = {}, 0, 0
    for bn in args.benchmarks:
        ds = coll[bn]
        for i in range(len(ds)):
            try:
                p = ds[i]
                if not p or "name" not in p: continue
                name = str(p["name"]); total += 1
                if name.lower() in have:
                    continue
                seq = p.get("seq","")
                if "-" in seq:
                    dash += 1
                    seq = seq.replace("-", "X")
                if name not in missing and seq:
                    missing[name] = seq
            except Exception:
                break

    with open(args.out_fasta, "w") as f:
        for name, seq in missing.items():
            f.write(f">{name}\n{seq}\n")

    print(f"AF models present: {len(have)}")
    print(f"benchmark proteins total (both sets): {total}")
    print(f"still MISSING an AF model: {len(missing)}  (written to {args.out_fasta})")
    print(f"  of which had '-' chars fixed to 'X': {dash}")
    print(f"\nRe-fold in SMALL batches to avoid MSA-server timeouts, e.g.:")
    print(f"  colabfold_batch --num-recycle 3 {args.out_fasta} {args.af_dir}")
    print(f"(or split {args.out_fasta} into chunks of ~10 sequences and run sequentially)")

if __name__ == "__main__":
    main()
