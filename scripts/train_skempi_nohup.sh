#!/usr/bin/env bash
# SKEMPI数据集训练脚本 - 使用nohup确保进程持续运行

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

# 配置文件路径
MODEL_CONFIG="./config/models/best_skempi_mutfocus.yml"
DATA_CONFIG="./config/datasets/SKEMPI.yml"
RUN_CONFIG="./config/runs/finetune_skempi_mutfocus.yml"

# 日志文件路径
LOG_FILE="/media/SSD0/csd/lrg/copra_h/outputs/train_skempi_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="/tmp/train_skempi.pid"

echo "========== SKEMPI数据集训练开始 ==========" | tee -a "$LOG_FILE"
echo "时间: $(date -Iseconds)" | tee -a "$LOG_FILE"
echo "模型配置: $MODEL_CONFIG" | tee -a "$LOG_FILE"
echo "数据配置: $DATA_CONFIG" | tee -a "$LOG_FILE"
echo "运行配置: $RUN_CONFIG" | tee -a "$LOG_FILE"
echo "日志文件: $LOG_FILE" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

# 使用nohup在后台运行训练，即使SSH断开也不会终止
nohup conda run -n copra_h python run.py finetune ddG \
    --model_config "$MODEL_CONFIG" \
    --data_config "$DATA_CONFIG" \
    --run_config "$RUN_CONFIG" \
    > "$LOG_FILE" 2>&1 &

# 保存进程PID
echo $! > "$PID_FILE"
echo "训练进程已启动，PID: $!" | tee -a "$LOG_FILE"
echo "日志文件: $LOG_FILE"
echo "PID文件: $PID_FILE"

# 等待几秒后检查进程状态
sleep 5
if ps -p $! > /dev/null; then
    echo "进程运行正常" | tee -a "$LOG_FILE"
else
    echo "进程异常退出，请检查日志: $LOG_FILE"
    tail -30 "$LOG_FILE"
    exit 1
fi