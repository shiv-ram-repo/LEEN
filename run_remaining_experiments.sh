#!/bin/bash
# ============================================================
# LEEN — all remaining experiments for the revision, in order.
# Run from the clean repo root. Each training run ~1h on the A6000.
# ============================================================
set -e
cd /backup/new_work/LEEN_clean
PP="PYTHONPATH=/backup/new_work:."
DR="/backup/stability/data_root"
EC="esm_cache_full.pt"

echo "=================================================================="
echo "STEP 1 — scalar-head ablation, seeds 43 and 44 (42 already done)"
echo "=================================================================="
for s in 43 44; do
  eval $PP python scripts/train/train_leen.py \
    --data_root $DR --esm_cache $EC \
    --head_type scalar --output_dir runs/leen_v2_scalar --seed $s
done

echo "=================================================================="
echo "STEP 2 — ESM-dropout=0 ablation, seeds 42 43 44"
echo "=================================================================="
for s in 42 43 44; do
  eval $PP python scripts/train/train_leen.py \
    --data_root $DR --esm_cache $EC \
    --esm_dropout 0 --output_dir runs/leen_v2_noesmdrop --seed $s
done

echo "=================================================================="
echo "STEP 3 — evaluate all new checkpoints to per-sample CSVs"
echo "=================================================================="
# scalar: seed 42 already eval'd during training print; re-eval all 3 to CSVs for the table
for cfg in leen_v2_scalar leen_v2_noesmdrop; do
  for s in 42 43 44; do
    if [ -f runs/$cfg/seed_$s/best.pt ]; then
      LABEL=$([ "$cfg" = "leen_v2_scalar" ] && echo "SCALAR" || echo "NOESMDROP")
      eval $PP python scripts/eval/eval_persample.py \
        --ckpt runs/$cfg/seed_$s/best.pt --family v2 \
        --esm_cache $EC --data_root $DR \
        --out_dir runs/leen_preds/${LABEL}_seed$s
    fi
  done
done

echo "=================================================================="
echo "STEP 4 — build the R5 ablation comparison table"
echo "=================================================================="
eval $PP python scripts/eval/build_tables.py \
  --preds_root runs/leen_preds --out_dir runs/leen_tables_r5 \
  --configs E2 SCALAR NOESMDROP --seeds 42 43 44

echo "=================================================================="
echo "STEP 5 — measure parameters + throughput (fills tab:param_accounting)"
echo "=================================================================="
eval $PP python scripts/eval/measure_params.py \
  --ckpt runs/leen_v2_drop/seed_42/best.pt --data_root $DR --esm_cache $EC || \
  echo "(if measure_params needs different args, check scripts/eval/measure_params.py --help)"

echo ""
echo "DONE. Now:"
echo "  - read runs/leen_tables_r5/ for the scalar + esm-dropout 3-seed numbers"
echo "  - fill tab:ablations [fill 3-seed] rows in missing_tables.tex"
echo "  - fill tab:param_accounting timing/param numbers from STEP 5"
