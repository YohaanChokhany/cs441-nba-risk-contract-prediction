#!/usr/bin/env bash
# Rebuilds data/interim (not tracked), data/processed and the figures from the raw tables.
# Run from the repository root. Requires pandas, numpy and matplotlib (requirements.txt).
set -euo pipefail
RAW=data/raw/nba-data-1946-2026
mkdir -p data/interim data/processed reports/figures
python code/clean_nba_tables.py "$RAW" data/interim
python code/build_master.py data/interim "$RAW" data/raw/external data/processed
python code/make_feature_dictionary.py data/processed
python code/baseline_check.py data/processed/master_model_ready.csv
python code/make_figures.py "$RAW" data/interim data/processed reports/figures
