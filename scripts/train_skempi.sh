#!/usr/bin/env bash
# SKEMPI数据集训练脚本
# 使用已有的预提取特征进行训练

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

# 配置文件路径
MODEL_CONFIG="./config/models/best_skempi_mutfocus.yml"
DATA_CONFIG="./config/datasets/SKEMPI.yml"
RUN_CONFIG="./config/runs/finetune_skempi_mutfocus.yml"

echo "========== SKEMPI数据集训练开始 =========="
echo "时间: $(date -Iseconds)"
echo "模型配置: $MODEL_CONFIG"
echo "数据配置: $DATA_CONFIG"
echo "运行配置: $RUN_CONFIG"
echo "=========================================="

# 使用copra_h conda环境启动训练
conda run -n copra_h python run.py finetune ddG \
    --model_config "$MODEL_CONFIG" \
    --data_config "$DATA_CONFIG" \
    --run_config "$RUN_CONFIG"

echo ""
echo "训练完成: $(date -Iseconds)"