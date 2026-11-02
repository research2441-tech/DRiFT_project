#!/usr/bin/env bash
# Reproduces every result, table and figure of the manuscript (about 3-4 h on a 2-core CPU).
set -e
python -m pytest -q test_core.py
for g in X SX SQiSW; do python run_experiments.py e1 --gate $g; done
python run_experiments.py e2 --gate X
python run_experiments.py e2 --gate SQiSW
python run_experiments.py e3
python run_experiments.py e45 --gate X
python run_experiments.py e45 --gate SQiSW
python run_experiments.py e6
python run_experiments.py e10 --gate X
python run_experiments.py e10 --gate SQiSW
python fig_architecture.py
python make_figures.py
python make_tables.py
