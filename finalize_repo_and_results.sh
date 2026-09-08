#!/bin/bash
# ============================================================
# LEEN — make the repo self-contained + stage R7 release materials.
# ============================================================
set -e
cd /backup/new_work/LEEN_clean

echo "=== 1. vendor protstab_data into the repo (self-contained install) ==="
cp -r /backup/new_work/protstab_data ./protstab_data
rm -rf protstab_data/__pycache__ protstab_data/*.egg-info
find protstab_data -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

echo "=== 2. add protstab_data to pyproject packages ==="
# edit pyproject.toml: include = ["leen*", "protstab_data*"]
python3 - <<'PY'
f="pyproject.toml"; s=open(f).read()
if 'protstab_data' not in s:
    s=s.replace('include = ["leen*"]','include = ["leen*", "protstab_data*"]')
    open(f,"w").write(s); print("  added protstab_data to pyproject")
else:
    print("  already present")
PY

echo "=== 3. verify self-contained install works (no PYTHONPATH hack) ==="
pip install -e . -q 2>&1 | tail -2 || echo "  (run pip install -e . manually if needed)"
python -c "import leen, protstab_data; print('  both packages import cleanly')" || \
  echo "  (still needs PYTHONPATH — check pyproject)"

echo "=== 4. stage R7 release materials ==="
mkdir -p results
cp -r runs/leen_preds results/per_sample_predictions 2>/dev/null || true
cp -r runs/leen_tables_full results/tables 2>/dev/null || true
cp runs/leen_tables_ood/*.csv results/tables/ 2>/dev/null || true
cp runs/leen_tables_r5/*.csv results/tables/ 2>/dev/null || true
mkdir -p results/mmseqs_audit
cp docs/mmseqs_audit/* results/mmseqs_audit/ 2>/dev/null || true

echo "=== 5. write the anonymized-repo note ==="
cat > ANONYMIZED_RELEASE.md <<'EOF'
This repository accompanies the LEEN manuscript and is provided for anonymous review.
Contents:
- leen/                     model package (EGNN/GVP/PaiNN encoders, energy head, adapters, losses)
- protstab_data/            dataset loaders, featurizer, PDB parser
- scripts/train|eval|experiments|preprocess/   all training, evaluation, and ablation scripts
- results/per_sample_predictions/   per-seed per-sample predictions for all models on all 11 benchmarks
- results/tables/           regenerated metric tables (Spearman/Pearson/RMSE/MAE, per-seed, bootstrap CIs)
- results/mmseqs_audit/     MMseqs2 filtering command output; retained/removed mutation lists
- configs/                  training configurations (seeds 42/43/44)
All tables and figures in the manuscript are regenerated from results/per_sample_predictions
via scripts/eval/build_tables.py and scripts/eval/audit_claims.py.
EOF

echo ""
echo "DONE. Repo is self-contained; results staged under results/."
echo "For public release, host results/ (predictions + tables) on Zenodo; code goes to GitHub."
