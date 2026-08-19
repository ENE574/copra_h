#!/usr/bin/env bash
# Unified entity-entity training entrypoints (see config/task_profiles/).
set -euo pipefail
cd "$(dirname "$0")/.."

run() {
  echo ">>> $*"
  python run.py finetune ddG "$@"
}

case "${1:-help}" in
  skempi)
    run \
      --model_config ./config/models/unified_ddg_skempi.yml \
      --data_config ./config/datasets/SKEMPI.yml \
      --run_config ./config/runs/finetune_unified_skempi.yml
    ;;
  mpd)
    run \
      --model_config ./config/models/unified_ddg_mpd_protna.yml \
      --data_config ./config/datasets/MPD_pempni_tv_match.yml \
      --run_config ./config/runs/finetune_unified_mpd_protna.yml
    ;;
  mcsm)
    run \
      --model_config ./config/models/unified_ddg_mcsm.yml \
      --data_config ./config/datasets/mCSM.yml \
      --run_config ./config/runs/finetune_unified_mpd_protna.yml
    ;;
  multitask)
    run \
      --model_config ./config/models/unified_multitask_protna.yml \
      --data_config ./config/datasets/mCSM.yml \
      --run_config ./config/runs/finetune_multitask_protna.yml
    ;;
  skempi_tuneA)
    run \
      --model_config ./config/models/unified_ddg_skempi_tuneA.yml \
      --data_config ./config/datasets/SKEMPI.yml \
      --run_config ./config/runs/finetune_unified_skempi_tuneA.yml
    ;;
  skempi_tuneB)
    run \
      --model_config ./config/models/unified_ddg_skempi_tuneB.yml \
      --data_config ./config/datasets/SKEMPI.yml \
      --run_config ./config/runs/finetune_unified_skempi_tuneB.yml
    ;;
  *)
    echo "Usage: $0 {skempi|mpd|mcsm|multitask|skempi_tuneA|skempi_tuneB}"
    exit 1
    ;;
esac
