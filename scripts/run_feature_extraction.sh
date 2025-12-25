#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-/workspace/copra_h/config/feature_extract.yml}"
PYTHON_BIN="${PYTHON_BIN:-/workspace/envs/copra_h/bin/python}"

export LD_LIBRARY_PATH=/workspace/envs/copra_h/lib:/workspace/envs/copra_h/lib/python3.10/site-packages/nvidia/cublas/lib:/workspace/envs/copra_h/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}

exec "$PYTHON_BIN" /workspace/copra_h/extract_features.py --config "$CONFIG_PATH"
