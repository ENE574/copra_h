#!/usr/bin/env bash
# 依次执行多组 offline dG finetune（PRA310_foldx_csv → PRA201_foldx_csv）。
#
# 用法（在任意目录）:
#   bash scripts/finetune_dg_offline_sweep.sh
# 或先 chmod +x 后:
#   ./scripts/finetune_dg_offline_sweep.sh
#
# 任一步失败则立即退出（set -e）。若要「失败后仍继续」，可改为 set +e 并自行判断 $?。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

RUN_CFG="./config/runs/finetune_struct.yml"
DATA_PRA310="./config/datasets/PRA310_foldx_csv.yml"
DATA_PRA201="./config/datasets/PRA201_foldx_csv.yml"

run() {
  local n="$1"
  shift
  echo ""
  echo "========== [$n] $(date -Iseconds) =========="
  echo "$*"
  echo "=========================================="
  "$@"
}

n=0

# --- PRA310 ---
run $((++n)) python run.py finetune dG --model_config ./config/models/best_htf_off.yml       --data_config "$DATA_PRA310" --run_config "$RUN_CFG"
run $((++n)) python run.py finetune dG --model_config ./config/models/best_pia_htf_off.yml    --data_config "$DATA_PRA310" --run_config "$RUN_CFG"
run $((++n)) python run.py finetune dG --model_config ./config/models/best_pia_off.yml       --data_config "$DATA_PRA310" --run_config "$RUN_CFG"
run $((++n)) python run.py finetune dG --model_config ./config/models/best_seq_off.yml        --data_config "$DATA_PRA310" --run_config "$RUN_CFG"
run $((++n)) python run.py finetune dG --model_config ./config/models/best_struct_off.yml     --data_config "$DATA_PRA310" --run_config "$RUN_CFG"

# --- PRA201 ---
run $((++n)) python run.py finetune dG --model_config ./config/models/best_htf_off.yml       --data_config "$DATA_PRA201" --run_config "$RUN_CFG"
run $((++n)) python run.py finetune dG --model_config ./config/models/best_pia_htf_off.yml    --data_config "$DATA_PRA201" --run_config "$RUN_CFG"
run $((++n)) python run.py finetune dG --model_config ./config/models/best_pia_off.yml       --data_config "$DATA_PRA201" --run_config "$RUN_CFG"
run $((++n)) python run.py finetune dG --model_config ./config/models/best_seq_off.yml        --data_config "$DATA_PRA201" --run_config "$RUN_CFG"
run $((++n)) python run.py finetune dG --model_config ./config/models/best_struct_off.yml     --data_config "$DATA_PRA201" --run_config "$RUN_CFG"

echo ""
echo "全部 $n 步完成: $(date -Iseconds)"
