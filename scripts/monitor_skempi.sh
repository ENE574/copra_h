#!/usr/bin/env bash
# SKEMPI训练监控脚本

LOG_FILE=$(ls -lt /media/SSD0/csd/lrg/copra_h/outputs/train_skempi_*.log 2>/dev/null | head -1 | awk '{print $9}')
PID_FILE="/tmp/train_skempi.pid"

echo "========== SKEMPI训练监控 =========="
echo "时间: $(date -Iseconds)"
echo ""

# 检查进程状态
echo "### 进程状态 ###"
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null; then
        echo "进程运行中 (PID: $PID)"
        ps aux | grep "$PID" | grep -v grep | awk '{printf "CPU: %s%% | MEM: %s%% | TIME: %s\n", $3, $4, $10}'
    else
        echo "进程已退出 (PID: $PID)"
    fi
else
    echo "未找到PID文件"
fi
echo ""

# 检查GPU状态
echo "### GPU状态 ###"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader | awk -F', ' '{printf "GPU %s: %s | 显存: %s/%s | 利用率: %s%% | 温度: %s°C\n", $1, $2, $3, $4, $5, $6}'
echo ""

# 检查日志文件
echo "### 最新日志 ###"
if [ -f "$LOG_FILE" ]; then
    echo "日志文件: $LOG_FILE"
    echo "日志行数: $(wc -l < "$LOG_FILE")"
    echo ""
    echo "最近20行:"
    tail -20 "$LOG_FILE"
else
    echo "未找到日志文件"
fi

# 检查训练输出目录
OUTPUT_DIR="/media/SSD0/csd/lrg/copra_h/outputs/SKEMPI_localdiff_cv"
if [ -d "$OUTPUT_DIR" ]; then
    LATEST_RUN=$(ls -lt "$OUTPUT_DIR" | grep -E "skempi_localdiff_2026" | head -1 | awk '{print $9}')
    if [ -n "$LATEST_RUN" ]; then
        echo ""
        echo "### 训练输出 ###"
        echo "最新运行: $LATEST_RUN"
        echo "文件数量: $(find "$OUTPUT_DIR/$LATEST_RUN" -type f 2>/dev/null | wc -l)"
        
        # 检查metrics.csv
        METRICS_FILE=$(find "$OUTPUT_DIR/$LATEST_RUN" -name "metrics.csv" -type f 2>/dev/null | head -1)
        if [ -n "$METRICS_FILE" ]; then
            echo "Metrics文件: $METRICS_FILE"
            echo "最近记录:"
            tail -5 "$METRICS_FILE"
        fi
    fi
else
    echo ""
    echo "输出目录尚未创建"
fi

echo ""
echo "=========================================="