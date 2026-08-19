#!/usr/bin/env bash
# ============================================================================
# Unified dG Pre-training Pipeline
# ============================================================================
# This script orchestrates the complete Unified dG → ddG transfer pipeline:
#
#   Phase 0: Data Preparation
#   Phase 1: Feature Extraction for all three interaction types
#   Phase 2: Unified dG Pre-training (PRA310 + PPI_dG + DNA_dG)
#   Phase 3: Transfer → ddG fine-tuning (mCSM-RNA, MPD, SKEMPI)
#   Phase 4: Comparison vs from-scratch training
#
# Usage:
#   bash scripts/run_unified_dg_pipeline.sh [phase]
#
#   phase=0  : Prepare datasets only
#   phase=1  : Extract features only (requires data)
#   phase=2  : Run unified dG pre-training (requires features)
#   phase=3  : Run ddG transfer fine-tuning (requires checkpoint)
#   phase=all: Run everything
#
# Requirements:
#   - Conda environment 'copra_h' (from environment.yml)
#   - PRA310 dataset + features already prepared
#   - SKEMPI PDBs in /media/SSD0/csd/lrg/copra_h/datasets/SKEMPI/PDBs
#   - Hardware: GPU with ≥24 GB VRAM recommended
# ============================================================================

set -euo pipefail

# ---- Configuration ----------------------------------------------------------
PROJECT_ROOT="/home/csd/lrg/copra_h"
DATASETS_ROOT="/media/SSD0/csd/lrg/copra_h/datasets"
OUTPUTS_ROOT="/media/SSD0/csd/lrg/copra_h/outputs"
WEIGHTS_ROOT="/media/SSD0/csd/lrg/copra_h/weights"
CONDA_ENV="copra_h"
PYTHON_BIN="${CONDA_PREFIX:-${HOME}/miniconda3/envs/${CONDA_ENV}}/bin/python"

# GPU assignment (customize per phase)
GPU_FEATURE_EXTRACT=""    # Will use all available GPUs
GPU_PRETRAIN="0"
GPU_TRANSFER_MCSM="1"
GPU_TRANSFER_MPD="2"
GPU_TRANSFER_SKEMPI="3"

# SKEMPI v2 download URL (for PPI dG)
SKEMPI_V2_URL="https://life.bsc.es/pid/skempi2/download/skempi_v2.csv"

# ---- Helper functions -------------------------------------------------------
log() { echo -e "\n[$(date '+%Y-%m-%d %H:%M:%S')] $*\n"; }

activate_env() {
    if command -v conda &>/dev/null; then
        eval "$(conda shell.bash hook)"
        conda activate "${CONDA_ENV}"
    elif [ -f "${CONDA_PREFIX}/envs/${CONDA_ENV}/bin/python" ]; then
        export PATH="${CONDA_PREFIX}/envs/${CONDA_ENV}/bin:$PATH"
    fi
}

run_python() {
    log "RUNNING: python $*"
    ${PYTHON_BIN} "$@"
}

# ---- Phase 0: Data Preparation ----------------------------------------------
phase0_prepare_data() {
    log "=== PHASE 0: Data Preparation ==="

    # ---- Step 0a: PPI dG (from SKEMPI 2.0) ----
    log "[0a] Preparing PPI dG dataset from SKEMPI 2.0..."
    if [ ! -f "${DATASETS_ROOT}/PPI_dG/splits/PPI_dG.csv" ]; then
        run_python prepare_datasets/prepare_ppi_dg_from_skempi.py \
            --skempi_url "${SKEMPI_V2_URL}" \
            --pdb_dir "${DATASETS_ROOT}/SKEMPI/PDBs" \
            --out_csv "${DATASETS_ROOT}/PPI_dG/splits/PPI_dG.csv" \
            --num_folds 5 \
            --seed 2024
        log "[0a] PPI dG dataset created!"
    else
        log "[0a] PPI dG dataset already exists, skipping."
    fi

    # ---- Step 0b: DNA dG (from PDBbind or custom CSV) ----
    log "[0b] Preparing DNA dG dataset..."
    log "[0b] NOTE: This step requires PDBbind v2020 (http://www.pdbbind.org.cn/)."
    log "[0b] If you don't have it, provide a custom CSV with:"
    log "[0b]   PDB, Protein chains, DNA chains, △G(kcal/mol)"
    log "[0b] Or place PDBbind at: ${DATASETS_ROOT}/PDBbind/"

    if [ -f "${DATASETS_ROOT}/PDBbind/index/INDEX_general_PL_data.2020" ]; then
        run_python prepare_datasets/prepare_dna_dg_from_pdbbind.py \
            --pdbbind_dir "${DATASETS_ROOT}/PDBbind" \
            --out_csv "${DATASETS_ROOT}/DNA_dG/splits/DNA_dG.csv"
    elif [ -f "${DATASETS_ROOT}/DNA_dG/splits/DNA_dG.csv" ]; then
        log "[0b] DNA dG CSV already exists, skipping."
    else
        log "[0b] ⚠ PDBbind not found. Creating placeholder — you'll need to fill in data."
        log "[0b] For now, creating a minimal template."
        mkdir -p "${DATASETS_ROOT}/DNA_dG/splits" "${DATASETS_ROOT}/DNA_dG/PDBs"
        if [ ! -f "${DATASETS_ROOT}/DNA_dG/splits/DNA_dG.csv" ]; then
            echo "PDB,Protein chains,DNA chains,△G(kcal/mol),fold_0,fold_1,fold_2,fold_3,fold_4" \
                > "${DATASETS_ROOT}/DNA_dG/splits/DNA_dG.csv"
        fi
    fi

    # ---- Step 0c: Verify dataset CSVs ----
    log "[0c] Verifying dataset structure..."
    for csv_name in PRA310 PPI_dG DNA_dG; do
        csv_path="${DATASETS_ROOT}/${csv_name}/splits/${csv_name}.csv"
        if [ -f "${csv_path}" ]; then
            n_lines=$(wc -l < "${csv_path}")
            log "  ✅ ${csv_name}: ${n_lines} lines (${n_lines-1} data rows)"
        else
            log "  ❌ ${csv_name}: NOT FOUND at ${csv_path}"
        fi
    done

    log "=== PHASE 0 complete ==="
}


# ---- Phase 1: Feature Extraction --------------------------------------------
phase1_extract_features() {
    log "=== PHASE 1: Feature Extraction ==="

    # PRA310 (already done if data exists, else skip)
    if [ -d "${OUTPUTS_ROOT}/feature_extraction_PRA310/protein_sequence" ]; then
        log "[1a] PRA310 features already exist, skipping."
    else
        log "[1a] Extracting PRA310 features..."
        run_python extract_features.py --config config/feature_extract_PRA310.yml
    fi

    # PPI dG
    if [ -d "${OUTPUTS_ROOT}/feature_extraction_PPI_dG/protein_sequence" ]; then
        log "[1b] PPI dG features already exist, skipping."
    elif [ -f "${DATASETS_ROOT}/PPI_dG/splits/PPI_dG.csv" ]; then
        log "[1b] Extracting PPI dG features..."
        run_python extract_features.py --config config/feature_extract_PPI_dG.yml
    else
        log "[1b] ⚠ PPI dG CSV not found, skipping feature extraction."
    fi

    # DNA dG
    if [ -d "${OUTPUTS_ROOT}/feature_extraction_DNA_dG/protein_sequence" ]; then
        log "[1c] DNA dG features already exist, skipping."
    elif [ -f "${DATASETS_ROOT}/DNA_dG/splits/DNA_dG.csv" ]; then
        log "[1c] Extracting DNA dG features..."
        run_python extract_features.py --config config/feature_extract_DNA_dG.yml
    else
        log "[1c] ⚠ DNA dG CSV not found, skipping feature extraction."
    fi

    log "=== PHASE 1 complete ==="
}


# ---- Phase 2: Unified dG Pre-training ---------------------------------------
phase2_pretrain_unified() {
    log "=== PHASE 2: Unified dG Pre-training ==="

    # Determine feature availability
    HAS_PRA310=true
    HAS_PPI=false
    HAS_DNA=false

    [ -d "${OUTPUTS_ROOT}/feature_extraction_PRA310/protein_sequence" ] && HAS_PRA310=true || HAS_PRA310=false
    [ -d "${OUTPUTS_ROOT}/feature_extraction_PPI_dG/protein_sequence" ] && HAS_PPI=true || HAS_PPI=false
    [ -d "${OUTPUTS_ROOT}/feature_extraction_DNA_dG/protein_sequence" ] && HAS_DNA=true || HAS_DNA=false

    log "Feature availability: PRA310=${HAS_PRA310} PPI_dG=${HAS_PPI} DNA_dG=${HAS_DNA}"

    # Build run config dynamically based on what's available
    RUN_CONFIG="/tmp/unified_dg_run_$$.yml"

    cat > "${RUN_CONFIG}" << 'YAMLEOF'
# Generated by run_unified_dg_pipeline.sh
epochs: 50
patience: 60
output_dir: '/media/SSD0/csd/lrg/copra_h/outputs/unified_dg_pretrain'
gpus:
YAMLEOF
    echo "  - ${GPU_PRETRAIN}" >> "${RUN_CONFIG}"

    cat >> "${RUN_CONFIG}" << 'YAMLEOF'
ckpt: null
run_name: 'unified_dg_'
wandb: false
num_folds: 1
multitask_primary_task: pra310
multitask_sources:
YAMLEOF

    # PRA310 always included
    cat >> "${RUN_CONFIG}" << 'YAMLEOF'
  - name: pra310
    data_config: ./config/datasets/PRA310.yml
    batch_size: 8
YAMLEOF

    # PPI dG (if available)
    if [ "${HAS_PPI}" = true ]; then
        cat >> "${RUN_CONFIG}" << 'YAMLEOF'
  - name: ppi_dg
    data_config: ./config/datasets/PPI_dG.yml
    batch_size: 6
YAMLEOF
    fi

    # DNA dG (if available)
    if [ "${HAS_DNA}" = true ]; then
        cat >> "${RUN_CONFIG}" << 'YAMLEOF'
  - name: dna_dg
    data_config: ./config/datasets/DNA_dG.yml
    batch_size: 4
YAMLEOF
    fi

    cat >> "${RUN_CONFIG}" << 'YAMLEOF'
accumulate_grad_batches: 4
checkpoint_monitor: val/all_pearson
checkpoint_mode: max
early_stop_monitor: val/all_pearson
early_stop_mode: max
skip_test: true
YAMLEOF

    log "[2a] Starting unified dG pre-training..."
    log "Run config: ${RUN_CONFIG}"
    cat "${RUN_CONFIG}"

    run_python run.py finetune dG \
        --model_config ./config/models/unified_dg_pretrain.yml \
        --data_config ./config/datasets/PRA310.yml \
        --run_config "${RUN_CONFIG}"

    # Find the best checkpoint
    log "[2b] Finding best checkpoint..."
    BEST_CKPT=$(find "${OUTPUTS_ROOT}/unified_dg_pretrain" -name "best-*.ckpt" 2>/dev/null | head -1)
    if [ -n "${BEST_CKPT}" ]; then
        log "  ✅ Best checkpoint: ${BEST_CKPT}"
        echo "${BEST_CKPT}" > "${OUTPUTS_ROOT}/unified_dg_pretrain/best_checkpoint_path.txt"
    else
        log "  ⚠ No best checkpoint found. Check training output."
    fi

    rm -f "${RUN_CONFIG}"
    log "=== PHASE 2 complete ==="
}


# ---- Phase 3: Transfer → ddG Fine-tuning ------------------------------------
phase3_transfer_ddg() {
    log "=== PHASE 3: ddG Transfer Fine-tuning ==="

    # Find the best unified dG checkpoint
    CKPT_FILE="${OUTPUTS_ROOT}/unified_dg_pretrain/best_checkpoint_path.txt"
    if [ -f "${CKPT_FILE}" ]; then
        BEST_CKPT=$(cat "${CKPT_FILE}")
    else
        BEST_CKPT=$(find "${OUTPUTS_ROOT}/unified_dg_pretrain" -name "best-*.ckpt" 2>/dev/null | head -1)
    fi

    if [ -z "${BEST_CKPT}" ] || [ ! -f "${BEST_CKPT}" ]; then
        log "  ❌ No unified dG checkpoint found! Run Phase 2 first."
        return 1
    fi
    log "  Using checkpoint: ${BEST_CKPT}"

    # ---- Transfer to mCSM-RNA (prot-RNA ddG) ----
    log "[3a] Transfer → mCSM-RNA..."
    if [ -f "${DATASETS_ROOT}/mCSM_RNA/splits/crossvalidation.csv" ]; then
        # Use best.yml model config (has mutation GNN + local diff) with unified checkpoint initialization
        # The DDGModule has strict_loading=False so ddG-specific layers init randomly
        run_python run.py finetune ddG \
            --model_config ./config/models/best.yml \
            --data_config ./config/datasets/mCSM.yml \
            --run_config ./config/runs/finetune_mcsm.yml \
            --model_config.ckpt "${BEST_CKPT}"
    else
        log "  ⚠ mCSM-RNA data not found, skipping."
    fi

    # ---- Transfer to MPD (prot-DNA ddG) ----
    log "[3b] Transfer → MPD..."
    if [ -f "${DATASETS_ROOT}/MPD_merged/splits/MPD_merged_copra.csv" ]; then
        run_python run.py finetune ddG \
            --model_config ./config/models/best_mpd.yml \
            --data_config ./config/datasets/MPD_pempni.yml \
            --run_config ./config/runs/finetune_mpd_pempni.yml \
            --model_config.ckpt "${BEST_CKPT}"
    else
        log "  ⚠ MPD data not found, skipping."
    fi

    # ---- Transfer to SKEMPI (PPI ddG) ----
    log "[3c] Transfer → SKEMPI..."
    if [ -f "${DATASETS_ROOT}/SKEMPI/splits/skempi.csv" ]; then
        run_python run.py finetune ddG \
            --model_config ./config/models/best_skempi.yml \
            --data_config ./config/datasets/SKEMPI.yml \
            --run_config ./config/runs/finetune_skempi.yml \
            --model_config.ckpt "${BEST_CKPT}"
    else
        log "  ⚠ SKEMPI data not found, skipping."
    fi

    log "=== PHASE 3 complete ==="
}


# ---- Phase 4: Comparison with from-scratch training -------------------------
phase4_comparison() {
    log "=== PHASE 4: Comparison Runs (from scratch) ==="

    log "[4a] From-scratch: mCSM-RNA..."
    if [ -f "${DATASETS_ROOT}/mCSM_RNA/splits/crossvalidation.csv" ]; then
        run_python run.py finetune ddG \
            --model_config ./config/models/best.yml \
            --data_config ./config/datasets/mCSM.yml \
            --run_config ./config/runs/finetune_mcsm.yml
    fi

    log "[4b] From-scratch: MPD..."
    if [ -f "${DATASETS_ROOT}/MPD_merged/splits/MPD_merged_copra.csv" ]; then
        run_python run.py finetune ddG \
            --model_config ./config/models/best_mpd.yml \
            --data_config ./config/datasets/MPD_pempni.yml \
            --run_config ./config/runs/finetune_mpd_pempni.yml
    fi

    log "[4c] From-scratch: SKEMPI..."
    if [ -f "${DATASETS_ROOT}/SKEMPI/splits/skempi.csv" ]; then
        run_python run.py finetune ddG \
            --model_config ./config/models/best_skempi.yml \
            --data_config ./config/datasets/SKEMPI.yml \
            --run_config ./config/runs/finetune_skempi.yml
    fi

    log "=== PHASE 4 complete ==="
    log "Compare: outputs/transfer_*/ vs outputs/{mCSM,MPD_merged,SKEMPI_cv}/"
}


# ---- Main -------------------------------------------------------------------
main() {
    activate_env

    PHASE="${1:-all}"

    case "${PHASE}" in
        0|data|dataprep)
            phase0_prepare_data
            ;;
        1|feat|extract)
            phase1_extract_features
            ;;
        2|pretrain|unified)
            phase2_pretrain_unified
            ;;
        3|transfer|finetune)
            phase3_transfer_ddg
            ;;
        4|compare|scratch)
            phase4_comparison
            ;;
        all)
            phase0_prepare_data
            phase1_extract_features
            phase2_pretrain_unified
            phase3_transfer_ddg
            log "=== ALL PHASES COMPLETE ==="
            log "Next: Run Phase 4 if you want from-scratch baselines for comparison."
            ;;
        *)
            echo "Usage: $0 [phase]"
            echo "  phase=0|data     : Prepare datasets"
            echo "  phase=1|feat     : Extract features"
            echo "  phase=2|pretrain : Unified dG pre-training"
            echo "  phase=3|transfer : ddG transfer fine-tuning"
            echo "  phase=4|compare  : From-scratch training for comparison"
            echo "  phase=all        : Run everything"
            exit 1
            ;;
    esac
}

main "$@"
