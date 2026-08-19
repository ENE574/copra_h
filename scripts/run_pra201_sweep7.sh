#!/usr/bin/env bash
# PRA201 七套训练配置并行启动 (GPU 1-7)
# 目标: 在保持 pearson>0.6 且 spearman>0.6 的前提下, 折均 RMSE<1.3
set -u

cd /home/csd/lrg/copra_h
# 必须使用 copra_h conda 环境 (含 torch_cluster 等依赖)
PY=/home/csd/anaconda3/envs/copra_h/bin/python
LOG_DIR=/media/SSD0/csd/lrg/copra_h/outputs/PRA201_sweep7/logs
mkdir -p "$LOG_DIR"

# S1..S7 对应的 (model_config, run_config, gpu, log_tag)
declare -a MODELS=(
  config/models/pra201_sweep_s1.yml
  config/models/pra201_sweep_s2.yml
  config/models/pra201_sweep_s3.yml
  config/models/pra201_sweep_s4.yml
  config/models/pra201_sweep_s5.yml
  config/models/pra201_sweep_s6.yml
  config/models/pra201_sweep_s7.yml
)
declare -a RUNS=(
  config/runs/finetune_pra201_sweep_s1.yml
  config/runs/finetune_pra201_sweep_s2.yml
  config/runs/finetune_pra201_sweep_s3.yml
  config/runs/finetune_pra201_sweep_s4.yml
  config/runs/finetune_pra201_sweep_s5.yml
  config/runs/finetune_pra201_sweep_s6.yml
  config/runs/finetune_pra201_sweep_s7.yml
)
declare -a GPUS=(1 2 3 4 5 6 7)
DATA_CFG=config/datasets/PRA201.yml

PID_FILE="$LOG_DIR/pids.txt"
: > "$PID_FILE"

for i in 0 1 2 3 4 5 6; do
  s=$((i+1))
  echo "Launching S${s} on GPU ${GPUS[$i]}  (model=${MODELS[$i]}, run=${RUNS[$i]})"
  # setsid 使子进程成为独立 session leader, 完全脱离本 shell 进程组,
  # 避免父进程退出(如工具返回/kill进程组)时连带终止训练.
  setsid "$PY" run.py finetune dG \
    --model_config "${MODELS[$i]}" \
    --data_config "$DATA_CFG" \
    --run_config "${RUNS[$i]}" \
    > "$LOG_DIR/s${s}.log" 2>&1 &
  echo "  -> PID $!, log: $LOG_DIR/s${s}.log"
  echo "S${s} $!" >> "$PID_FILE"
done

echo
echo "All 7 sweeps launched. Monitor with: tail -f $LOG_DIR/s*.log"
echo "Results will land under outputs/PRA201_sweep7/pra201_sweep_sN_<timestamp>/"
