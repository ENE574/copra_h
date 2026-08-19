#!/usr/bin/env bash
# Extract offline esm2 (sequence) and esm_if1 (structure) embeddings for the
# protein-protein datasets P2P, S79, S90.
#
# Must be run from the project root (/home/csd/lrg/copra_h) in the conda env
# that has `esm` installed (the extractors import esm lazily).
#
# Usage:
#   bash scripts/run_feature_extraction_PPI.sh [P2P|S79|S90]   # optional subset
set -euo pipefail

cd "$(dirname "$0")/.."   # project root

DATASETS=("P2P" "S79" "S90")
if [ $# -ge 1 ]; then
    DATASETS=("$@")
fi

for DS in "${DATASETS[@]}"; do
    echo "===== Processing dataset: ${DS} ====="
    OUT="outputs/feature_extraction_${DS}"
    PDB_DIR="datasets/${DS}/PDBs"

    # 1) Build per-chain FASTAs from the PDBs (datasets have no sequences in CSV).
    IDS_ARG=""
    if [ -f "datasets/${DS}/${DS,,}_pdb_ids.txt" ]; then
        IDS_ARG="--ids-file datasets/${DS}/${DS,,}_pdb_ids.txt"
    fi
    python scripts/build_pdb_fastas.py \
        --pdb-dir "${PDB_DIR}" \
        --out "${OUT}" \
        ${IDS_ARG}

    # 2) Extract esm2 (sequence) + esm_if1 (structure) offline embeddings.
    python extract_features.py --config "config/feature_extract_${DS}.yml"

    echo "===== Done: ${DS} (outputs in ${OUT}) ====="
done
