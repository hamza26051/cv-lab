#!/usr/bin/env bash
set -euo pipefail
python3 -m pip install -r requirements.txt
python3 lab08.py all --seed 42 --n-train 150 --n-val 30 --n-test 30 --epochs 12
