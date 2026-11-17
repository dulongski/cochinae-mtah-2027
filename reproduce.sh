#!/usr/bin/env bash
# Regenerate the Cochinae submission from the raw data bundle, from scratch (~50 min).
#   MTAH_DATA_DIR=/path/to/bundle ./reproduce.sh
set -euo pipefail
cd "$(dirname "$0")"
DATA="${MTAH_DATA_DIR:-data}"
rm -rf "$DATA/cache"                      # no stale intermediate results
python -m cochinae.load                   # parse ATD + Smart Tagging into arrays
./cochinae/skillcorner_download.sh        # 20 SkillCorner open-data matches (~1.9 GB, kept if present)
python -m cochinae.skillcorner build      # their features + phase labels
python -m cochinae.skillcorner fit        # team and dead-ball classifiers (all 20 matches)
python -m cochinae.tune_skillcorner predict   # 4-fold held-out predictions on SkillCorner
python -m cochinae.tune_skillcorner grid      # decoder settings chosen on SkillCorner only
python -m cochinae.model                  # external-only pipeline -> evaluation/results/Cochinae.csv
python evaluation/validate_submission.py evaluation/results/Cochinae.csv
shasum -a 256 evaluation/results/Cochinae.csv
