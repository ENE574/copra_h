#!/usr/bin/env bash
# ===========================================================================
# DDG Pipeline: Pre-train → Fine-tune (SKEMPI / mCSM / MPD)
#
#   Step 1:  DDG pre-training on SKEMPI (backbone + DDG joint)
#   Step 2a: DDG fine-tuning on SKEMPI (5-fold CV, freeze backbone)
#   Step 2b: DDG fine-tuning on mCSM   (5-fold CV, PPI→RNA transfer)
#   Step 2c: DDG fine-tuning on MPD    (single fold, PPI→DNA transfer)
#
# Usage:
#   bash scripts/run_ddg_pipeline.sh [GPU_ID] [--skip-step1]
#
#   GPU_ID          which GPU to use (default 0)
#   --skip-step1    skip pre-training if already done (uses latest ckpt)
# ===========================================================================

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────
GPU="${1:-0}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# Use conda env python directly (bypass pyenv shim)
CONDA_PYTHON="$HOME/anaconda3/envs/copra_h/bin/python"
export CUDA_VISIBLE_DEVICES="$GPU"

RUN_SCRIPT="$PROJECT_ROOT/run.py"
OUTPUT_BASE="/media/SSD0/csd/lrg/copra_h/outputs"

# Backbone checkpoint (from unified_dg_pretrain)
BACKBONE_DIR="$OUTPUT_BASE/unified_dg_pretrain"

# Step configs
PRETRAIN_RUN_CFG="$PROJECT_ROOT/config/runs/pretrain_ddg_skempi.yml"
PRETRAIN_MODEL_CFG="$PROJECT_ROOT/config/models/pretrain_ddg_skempi.yml"
PRETRAIN_DATA_CFG="$PROJECT_ROOT/config/datasets/SKEMPI.yml"

SKEMPI_RUN_CFG="$PROJECT_ROOT/config/runs/finetune_ddg_skempi.yml"
SKEMPI_MODEL_CFG="$PROJECT_ROOT/config/models/finetune_ddg_skempi.yml"
SKEMPI_DATA_CFG="$PROJECT_ROOT/config/datasets/SKEMPI.yml"

MCSM_RUN_CFG="$PROJECT_ROOT/config/runs/finetune_ddg_mcsm.yml"
MCSM_MODEL_CFG="$PROJECT_ROOT/config/models/finetune_ddg_mcsm.yml"
MCSM_DATA_CFG="$PROJECT_ROOT/config/datasets/mCSM.yml"

MPD_RUN_CFG="$PROJECT_ROOT/config/runs/finetune_ddg_mpd.yml"
MPD_MODEL_CFG="$PROJECT_ROOT/config/models/finetune_ddg_mpd.yml"
MPD_DATA_CFG="$PROJECT_ROOT/config/datasets/MPD_merged.yml"

SKIP_STEP1=false
for arg in "$@"; do
    case "$arg" in
        --skip-step1) SKIP_STEP1=true ;;
    esac
done

# ── Helpers ───────────────────────────────────────────────────────────────
_blue()  { echo -e "\033[1;34m$*\033[0m"; }
_green() { echo -e "\033[1;32m$*\033[0m"; }
_red()   { echo -e "\033[1;31m$*\033[0m"; }
_sep()   { echo -e "\033[1;36m══════════════════════════════════════════════════════════════════\033[0m"; }

_log_step() {
    echo ""
    _sep
    _blue "[$(date '+%H:%M:%S')]  $1"
    _sep
}

_find_best_ckpt() {
    # Find the best checkpoint (fold_0, exclude last.ckpt).
    # Lightning stores: checkpoint/best-epoch=N-val/monitor=V.ckpt
    local run_dir="$1"
    find "$run_dir" -path "*/log_fold_0/checkpoint/*.ckpt" -not -name "last.ckpt" 2>/dev/null | sort -V | tail -1
}

_find_backbone_ckpt() {
    # Find the latest unified_dg_pretrain checkpoint
    local latest_run=$(find "$BACKBONE_DIR" -maxdepth 1 -type d -name "unified_dg_*" 2>/dev/null | sort | tail -1)
    if [[ -z "$latest_run" ]]; then
        echo ""
        return
    fi
    _find_best_ckpt "$latest_run"
}

_create_temp_run_cfg() {
    # Replace REPLACE_WITH_PRETRAIN_DDG_SKEMPI_CKPT in a run config
    local src_cfg="$1"
    local ckpt_path="$2"
    local tmp_cfg="${src_cfg%.yml}_tmp.yml"
    sed "s|REPLACE_WITH_PRETRAIN_DDG_SKEMPI_CKPT|$ckpt_path|" "$src_cfg" > "$tmp_cfg"
    echo "$tmp_cfg"
}

_run_step() {
    local name="$1"
    local run_cfg="$2"
    local model_cfg="$3"
    local data_cfg="$4"

    echo "  RUN:  $run_cfg"
    echo "  MODEL: $model_cfg"
    echo "  DATA: $data_cfg"

    "$CONDA_PYTHON" "$RUN_SCRIPT" \
        --run_config "$run_cfg" \
        --model_config "$model_cfg" \
        --data_config "$data_cfg" \
        finetune --stage ddG

    local ec=$?
    if [[ $ec -ne 0 ]]; then
        _red "✗ $name failed (exit code $ec)"
        return $ec
    fi
    _green "✓ $name done"
    return 0
}

# ── Sanity checks ─────────────────────────────────────────────────────────
for f in "$RUN_SCRIPT" "$PRETRAIN_MODEL_CFG" "$PRETRAIN_DATA_CFG" \
         "$SKEMPI_MODEL_CFG" "$SKEMPI_DATA_CFG" \
         "$MCSM_MODEL_CFG" "$MCSM_DATA_CFG" \
         "$MPD_MODEL_CFG" "$MPD_DATA_CFG"; do
    if [[ ! -f "$f" ]]; then
        _red "Missing: $f"
        exit 1
    fi
done

echo ""
_sep
echo "  DDG Pipeline — GPU $GPU"
echo "  $(date)"
echo ""
echo "  Step 1  (pretrain SKEMPI)       : $([ "$SKIP_STEP1" = true ] && echo 'SKIP' || echo 'RUN')"
echo "  Step 2a (finetune SKEMPI 5-fold): RUN"
echo "  Step 2b (finetune mCSM   1-fold): RUN"
echo "  Step 2c (finetune MPD    1-fold): RUN"
_sep

# ═══════════════════════════════════════════════════════════════════════════
# Step 1: DDG Pre-training on SKEMPI
# ═══════════════════════════════════════════════════════════════════════════
if [[ "$SKIP_STEP1" = true ]]; then
    DDG_CKPT="$(_find_best_ckpt "$OUTPUT_BASE/pretrain_ddg_skempi" 2>/dev/null)"
    if [[ -z "$DDG_CKPT" ]]; then
        _red "Cannot find existing pretrain_ddg_skempi checkpoint."
        _red "Run without --skip-step1 first, or provide a valid ckpt."
        exit 1
    fi
    _green "Skipping Step 1, using checkpoint: $DDG_CKPT"
else
    _log_step "Step 1: DDG Pre-training on SKEMPI (backbone + DDG joint)"

    BACKBONE_CKPT="$(_find_backbone_ckpt)"
    if [[ -z "$BACKBONE_CKPT" ]]; then
        _red "No unified_dg_pretrain checkpoint found in $BACKBONE_DIR"
        _red "Please run unified dG pretraining first."
        exit 1
    fi
    echo "  Backbone ckpt: $BACKBONE_CKPT"

    # Use pretrain run config as-is (backbone ckpt is hardcoded in it)
    _run_step "Step 1 (pretrain)" "$PRETRAIN_RUN_CFG" "$PRETRAIN_MODEL_CFG" "$PRETRAIN_DATA_CFG" || exit 1

    # Find the output checkpoint
    LATEST_RUN=$(find "$OUTPUT_BASE/pretrain_ddg_skempi" -maxdepth 1 -type d -name "pretrain_ddg_skempi_*" 2>/dev/null | sort | tail -1)
    DDG_CKPT="$(_find_best_ckpt "$LATEST_RUN")"
    if [[ -z "$DDG_CKPT" ]]; then
        _red "Could not find best checkpoint in $LATEST_RUN"
        exit 1
    fi
    _green "DDG checkpoint: $DDG_CKPT"
fi

echo ""
echo "DDG_CKPT=$DDG_CKPT"

# ═══════════════════════════════════════════════════════════════════════════
# Step 2a: Fine-tune DDG on SKEMPI (5-fold CV)
# ═══════════════════════════════════════════════════════════════════════════
_log_step "Step 2a: DDG Fine-tuning on SKEMPI (5-fold CV)"

SKEMPI_TMP="$(_create_temp_run_cfg "$SKEMPI_RUN_CFG" "$DDG_CKPT")"
trap "rm -f $SKEMPI_TMP" EXIT

_run_step "Step 2a (SKEMPI 5-fold)" "$SKEMPI_TMP" "$SKEMPI_MODEL_CFG" "$SKEMPI_DATA_CFG" || exit 1

# ═══════════════════════════════════════════════════════════════════════════
# Step 2b: Fine-tune DDG on mCSM (5-fold CV, PPI → RNA)
# ═══════════════════════════════════════════════════════════════════════════
_log_step "Step 2b: DDG Fine-tuning on mCSM (5-fold CV, PPI→RNA)"

MCSM_TMP="$(_create_temp_run_cfg "$MCSM_RUN_CFG" "$DDG_CKPT")"
trap "rm -f $SKEMPI_TMP $MCSM_TMP" EXIT

_run_step "Step 2b (mCSM 5-fold)" "$MCSM_TMP" "$MCSM_MODEL_CFG" "$MCSM_DATA_CFG" || exit 1

# ═══════════════════════════════════════════════════════════════════════════
# Step 2c: Fine-tune DDG on MPD (1-fold, PPI → DNA)
# ═══════════════════════════════════════════════════════════════════════════
_log_step "Step 2c: DDG Fine-tuning on MPD (single fold, PPI→DNA)"

MPD_TMP="$(_create_temp_run_cfg "$MPD_RUN_CFG" "$DDG_CKPT")"
trap "rm -f $SKEMPI_TMP $MCSM_TMP $MPD_TMP" EXIT

_run_step "Step 2c (MPD)" "$MPD_TMP" "$MPD_MODEL_CFG" "$MPD_DATA_CFG" || exit 1

# ═══════════════════════════════════════════════════════════════════════════
# Done
# ═══════════════════════════════════════════════════════════════════════════
_log_step "Pipeline Complete"
echo ""
echo "Outputs:"
echo "  SKEMPI: $OUTPUT_BASE/finetune_ddg_skempi/"
echo "  mCSM:   $OUTPUT_BASE/finetune_ddg_mcsm/"
echo "  MPD:    $OUTPUT_BASE/finetune_ddg_mpd/"
echo ""
_green "All steps finished successfully."
rm -f "$SKEMPI_TMP" "$MCSM_TMP" "$MPD_TMP"
