#!/usr/bin/env bash
# 批量启动SKEMPI之外的所有训练任务
# PRA310 (dG, GPU0), PRA201 (dG, GPU1), mCSM (ddG, GPU4), MPD (ddG, GPU5)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# 日志目录
LOG_DIR="/media/SSD0/csd/lrg/copra_h/outputs"
mkdir -p "$LOG_DIR"

echo "========== 批量训练启动 =========="
echo "时间: $(date -Iseconds)"
echo ""

# 定义所有训练任务
TASKS=(
  "PRA310|finetune dG|./config/models/best.yml|./config/datasets/PRA310.yml|./config/runs/finetune_pra310.yml|$LOG_DIR/train_PRA310.log"
  "PRA201|finetune dG|./config/models/best.yml|./config/datasets/PRA201.yml|./config/runs/finetune_pra201.yml|$LOG_DIR/train_PRA201.log"
  "mCSM|finetune ddG|./config/models/unified_ddg_mcsm.yml|./config/datasets/mCSM.yml|./config/runs/finetune_mcsm.yml|$LOG_DIR/train_mCSM.log"
  "MPD|finetune ddG|./config/models/unified_ddg_mpd_protna.yml|./config/datasets/MPD_merged.yml|./config/runs/finetune_mpd.yml|$LOG_DIR/train_MPD.log"
)

# 启动每个任务
for TASK_INFO in "${TASKS[@]}"; do
  IFS='|' read -r NAME MODE MODEL_CONFIG DATA_CONFIG RUN_CONFIG LOG_FILE <<< "$TASK_INFO"
  
  echo "启动 $NAME ($MODE) ..."
  echo "  模型配置: $MODEL_CONFIG"
  echo "  数据配置: $DATA_CONFIG"
  echo "  运行配置: $RUN_CONFIG"
  echo "  日志文件: $LOG_FILE"
  
  nohup conda run -n copra_h python run.py $MODE \
    --model_config "$MODEL_CONFIG" \
    --data_config "$DATA_CONFIG" \
    --run_config "$RUN_CONFIG" \
    > "$LOG_FILE" 2>&1 &
  
  PID=$!
  echo "  PID: $PID"
  echo ""
done

echo "所有训练任务已启动！"
echo "使用以下命令检查状态："
echo "  ps aux | grep 'run.py finetune' | grep -v grep"