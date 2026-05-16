#!/usr/bin/env bash
# 循环执行 finetune，每轮结束后立刻开始下一轮。
#
# 退出条件（满足任一即退出，退出码 0）：
#   1) 本地时间到达 DEADLINE_STR
#   2) 停止文件存在（默认仓库根目录 .finetune_loop_stop）
#
# 停止文件：当前 python 跑完后检测到文件即退出；创建示例：
#   touch "$ROOT/.finetune_loop_stop"
#
# 用法：
#   ./scripts/finetune_loop_until.sh
#   FINETUNE_LOOP_DEADLINE="2026-05-14 01:30:00" ./scripts/finetune_loop_until.sh
#   FINETUNE_LOOP_STOP_FILE=/tmp/stop_ft ./scripts/finetune_loop_until.sh

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

DEADLINE_STR="${FINETUNE_LOOP_DEADLINE:-2026-05-15 12:59:59}"
if ! DEADLINE_TS="$(date -d "$DEADLINE_STR" +%s 2>/dev/null)"; then
  echo "错误: 需要 GNU date（Linux 常见）。无法解析截止时间: $DEADLINE_STR" >&2
  exit 1
fi

STOP_FILE="${FINETUNE_LOOP_STOP_FILE:-$ROOT/.finetune_loop_stop}"

echo "仓库根目录: $ROOT"
echo "停止文件: $STOP_FILE  （touch 后即在本轮结束后退出）"
echo "截止时间(本地时区): $DEADLINE_STR -> epoch $DEADLINE_TS"
echo "当前时间: $(date -Iseconds)"

should_stop() {
  if [ -f "$STOP_FILE" ]; then
    echo "$(date -Iseconds) 检测到停止文件，停止循环: $STOP_FILE"
    return 0
  fi
  local now_ts
  now_ts="$(date +%s)"
  if [ "$now_ts" -ge "$DEADLINE_TS" ]; then
    echo "$(date -Iseconds) 已到达截止时间，退出。"
    return 0
  fi
  return 1
}

while true; do
  if should_stop; then
    exit 0
  fi

  echo "$(date -Iseconds) 开始新一轮 finetune ..."
  python run.py finetune dG \
    --model_config ./config/models/best.yml \
    --data_config ./config/datasets/PRA201_foldx_csv.yml \
    --run_config ./config/runs/finetune_struct.yml
  ec=$?
  echo "$(date -Iseconds) 本轮结束，退出码: $ec"

  if should_stop; then
    exit 0
  fi
done
