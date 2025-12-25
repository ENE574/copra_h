#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-config/feature_extract.yml}
PYTHON_BIN=${PYTHON_BIN:-/workspace/envs/copra_h/bin/python}

exec "${PYTHON_BIN}" extract_features.py --config "${CONFIG}" --models rna_ernie
