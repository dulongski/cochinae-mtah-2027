#!/usr/bin/env bash
# Regenerate the Cochinae submission from the raw data bundle, from scratch.
#   MTAH_DATA_DIR=/path/to/bundle ./reproduce.sh
set -euo pipefail
cd "$(dirname "$0")"
DATA="${MTAH_DATA_DIR:-data}"
rm -rf "$DATA/cache"                      # no stale intermediate results
python -m cochinae.load                   # parse ATD + Smart Tagging into arrays
python -m cochinae.model                  # model -> evaluation/results/Cochinae.csv
python evaluation/validate_submission.py evaluation/results/Cochinae.csv
shasum -a 256 evaluation/results/Cochinae.csv
