#!/usr/bin/env bash
# Reproduces every Person A result from the raw data. Takes about 10 minutes on 2 CPU cores.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 person_a/run_fraud.py
python3 person_a/analysis_cost.py
python3 person_a/analysis_attack.py
python3 person_a/run_german.py
python3 person_a/make_figures.py
