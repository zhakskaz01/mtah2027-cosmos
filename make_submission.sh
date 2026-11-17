#!/usr/bin/env bash
set -euo pipefail

TEAM="${1:?usage: ./make_submission.sh <team_name>}"
SAMPLE_COMMIT=e706dd506b360d69d9d123d5b8026e7294b13996
cd "$(dirname "$0")"

if [ ! -d external/sample-data ]; then
  git clone https://github.com/metrica-sports/sample-data external/sample-data
fi
git -C external/sample-data checkout -q "$SAMPLE_COMMIT"

python train.py --sample-data external/sample-data/data --out models
python predict.py --team "$TEAM" --models models --out-dir submission
python evaluation/validate_submission.py "submission/$TEAM.csv"
sha256sum "submission/$TEAM.csv" 2>/dev/null || shasum -a 256 "submission/$TEAM.csv"
