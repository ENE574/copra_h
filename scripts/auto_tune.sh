#!/bin/bash
# copra_h 全自动超参调优循环
# 每2分钟执行一次：检查B3→B4→... 自动分析→决策→训练
set -e

cd /home/csd/lrg/copra_h
PYTHON=/home/csd/anaconda3/envs/copra_h/bin/python
LOG_DIR=/home/csd/lrg/copra_h/logs
OUTPUT_BASE=/media/SSD0/csd/lrg/copra_h/outputs/MPD_merged_pempni_test2val
MODEL_CFG=config/models/best_mpd_nofix.yml
RUN_CFG=config/runs/finetune_mpd_pempni_test2val_nofix.yml
STATE_FILE=/home/csd/lrg/copra_h/.auto_state

# 如果训练正在运行，不打扰
if ps aux | grep -v grep | grep -q "run.py finetune.*best_mpd_nofix"; then
    exit 0
fi

# 初始化状态
if [ ! -f "$STATE_FILE" ]; then
    echo "LAST_ROUND=B2" > "$STATE_FILE"
    echo "BEST_ALLP=0.3361" >> "$STATE_FILE"
    echo "BEST_PCP=0.3076" >> "$STATE_FILE"
    echo "BEST_CONFIG=lr=5e-5,wd=1e-5,do=0.2,physics=1.0,ep=30,sched_pat=8" >> "$STATE_FILE"
    echo "ROUNDS_NO_IMPROV=0" >> "$STATE_FILE"
fi
source "$STATE_FILE"

# 找最新完成的 B 轮
LATEST_DIR=$(ls -dt ${OUTPUT_BASE}/mpd_pempni_test2val_B* 2>/dev/null | head -1)
if [ -z "$LATEST_DIR" ]; then
    echo "[$(date)] No B-series runs found yet, starting B3" >> ${LOG_DIR}/auto_tune.log
    NEXT_ROUND=3
else
    LATEST_NAME=$(basename "$LATEST_DIR")
    CUR_ROUND=$(echo "$LATEST_NAME" | grep -oP 'B\K[0-9]+' | head -1)
    CSV=$(ls -t ${LATEST_DIR}/log_fold_0/lightning_logs/version_*/metrics.csv 2>/dev/null | head -1)
    
    if [ -z "$CSV" ]; then
        echo "[$(date)] WARN: no metrics.csv in $LATEST_DIR" >> ${LOG_DIR}/auto_tune.log
        exit 0
    fi
    
    # ======== 分析结果 ========
    ANALYSIS=$($PYTHON -c "
import pandas as pd, numpy as np, sys

df = pd.read_csv('$CSV')
v = df.dropna(subset=['val/all_pearson']).copy()
if len(v) == 0:
    print('SKIP:0:0:0:0:0')
    sys.exit(0)
v['epoch'] = v['epoch'].astype(int)
vg = v.groupby('epoch', as_index=False).last()

ba = vg.loc[vg['val/all_pearson'].idxmax()]
bp = vg.loc[vg['val/pc_pearson'].idxmax()]

tail = vg.tail(8)
x = tail['epoch'].values.astype(float)
y = tail['val/all_pearson'].values.astype(float)
trend = np.polyfit(x, y, 1)[0] if len(x) >= 3 else 0
std = y.std()

tl = 999
if 'train_loss' in df.columns:
    t = df.dropna(subset=['train_loss']).copy()
    if len(t) > 0:
        t['epoch'] = t['epoch'].astype(int)
        tl_t = t[t['epoch'] >= vg['epoch'].max()-5]
        tl = tl_t.groupby('epoch')['train_loss'].last().mean()

print(f'{ba[\"val/all_pearson\"]:.4f}:{ba[\"val/pc_pearson\"]:.4f}:{bp[\"val/pc_pearson\"]:.4f}:{trend:.5f}:{tl:.4f}:{std:.4f}:{int(ba[\"epoch\"])}')
" 2>&1)
    
    IFS=':' read -r CURR_ALLP CURR_PCP CURR_PCP_MAX CURR_TREND CURR_TLOSS CURR_STD BEST_EPOCH <<< "$ANALYSIS"
    
    # ======== 读取当前配置 ========
    CURR_LR=$(grep -oP 'lr:\s*\K[0-9.e+-]+' "$MODEL_CFG" | head -1)
    CURR_WD=$(grep -oP 'weight_decay:\s*\K[0-9.e+-]+' "$MODEL_CFG" | head -1)
    CURR_PP=$(grep -oP 'physics_aux_prob:\s*\K[0-9.]+' "$MODEL_CFG" | head -1)
    CURR_DO=$(grep -oP 'attention_dropout:\s*\K[0-9.]+' "$MODEL_CFG" | head -1)
    CURR_EP=$(grep -oP '^epochs:\s*\K[0-9]+' "$RUN_CFG" | head -1)
    CURR_SP=$(grep -oP 'patience:\s*\K[0-9]+' "$MODEL_CFG" | tail -1)
    
    # 浮点比较用 python
    DECISION=$($PYTHON -c "
import sys
ca=float('$CURR_ALLP')
cp=float('$CURR_PCP')
ct=float('$CURR_TREND')
tl=float('$CURR_TLOSS')
cs=float('$CURR_STD')
cl=float('$CURR_LR')
cw=float('$CURR_WD')
cd=float('$CURR_DO')
pp=float('$CURR_PP')
ep=int('$CURR_EP')
sp=int('$CURR_SP')
ba=float('$BEST_ALLP')
bp=float('$BEST_PCP')
ni=int('$ROUNDS_NO_IMPROV')
last_round='$LAST_ROUND'

reason = ''
nl, nw, nd, npp, nep = cl, cw, cd, pp, ep

# 更新历史最佳
new_best_allp = ba
new_best_pcp = bp
new_best_cfg = '$BEST_CONFIG'

if ca > ba:
    new_best_allp = ca
    new_best_pcp = cp
    new_best_cfg = f'lr={cl},wd={cw},do={cd},physics={pp},ep={ep},sched_pat={sp}'
    reason = 'NEW_RECORD'
    ni = 0
    nep = ep + 10
elif ca > ba - 0.01:
    ni = 0  # near best, not degrading
else:
    ni += 1

# 决策树
if reason == 'NEW_RECORD':
    pass  # keep all params, just extend
elif ni >= 5:
    reason = '5轮无改善, 最终报告'
    nl, nw, nd, npp, nep = float('$BEST_CONFIG'.split('lr=')[1].split(',')[0]), float('$BEST_CONFIG'.split('wd=')[1].split(',')[0]), float('$BEST_CONFIG'.split('do=')[1].split(',')[0]), float('$BEST_CONFIG'.split('physics=')[1].split(',')[0]), 50
elif ni >= 3:
    reason = f'连续{ni}轮无改善, 回退最佳+延长'
    nl, nw, nd, npp = float('$BEST_CONFIG'.split('lr=')[1].split(',')[0]), float('$BEST_CONFIG'.split('wd=')[1].split(',')[0]), float('$BEST_CONFIG'.split('do=')[1].split(',')[0]), float('$BEST_CONFIG'.split('physics=')[1].split(',')[0])
    nep = 50
elif ct < -0.003 and tl < 0.3:
    if cd < 0.3:
        nd = round(cd + 0.05, 2)
        reason = f'过拟合(train_loss={tl:.3f}): dropout {cd}→{nd}'
    elif cw < 5e-4:
        nw = cw * 5
        reason = f'过拟合(train_loss={tl:.3f}): wd {cw:.0e}→{nw:.0e}'
    else:
        nl = cl * 0.5
        reason = f'过拟合(train_loss={tl:.3f}): lr {cl:.0e}→{nl:.0e}'
elif ct > 0.003 and ca < ba:
    reason = f'尾部上升但未创新高, 延长训练'
    nep = ep + 10
elif ca < ba - 0.03:
    if pp < 0.5:
        npp = 1.0
        reason = 'allP退步, 开启physics_aux'
    elif cl > 2e-5:
        nl = cl * 0.7
        reason = f'allP退步, 降lr: {cl:.0e}→{nl:.0e}'
    else:
        nd = min(cd + 0.05, 0.3)
        reason = f'allP退步, 增dropout: {cd}→{nd}'
elif cl > 2e-5:
    nl = max(cl * 0.7, 1.5e-5)
    reason = f'微调: lr {cl:.0e}→{nl:.0e}'
elif nep < 50:
    nep = ep + 10
    reason = '延长训练观察'
else:
    nep = 60
    reason = '最终延长'

nep = min(nep, 80)
nround = int('$CUR_ROUND') + 1

print(f'{nround}:{reason}:{nl}:{nw}:{nd}:{npp}:{nep}:{new_best_allp}:{new_best_pcp}:{new_best_cfg}:{ni}')
" 2>&1)
    
    IFS=':' read -r NEXT_ROUND DECISION_REASON NEW_LR NEW_WD NEW_DO NEW_PP NEW_EP NEW_BEST_ALLP2 NEW_BEST_PCP2 NEW_BEST_CFG NEW_NI <<< "$DECISION"
    
    # 记录日志
    echo "========================================" >> ${LOG_DIR}/auto_tune.log
    echo "[$(date)] Round B${CUR_ROUND} complete" >> ${LOG_DIR}/auto_tune.log
    echo "  Result: allP=${CURR_ALLP}, pcP=${CURR_PCP}, pcP_max=${CURR_PCP_MAX}, trend=${CURR_TREND}, tloss=${CURR_TLOSS}" >> ${LOG_DIR}/auto_tune.log
    echo "  Decision: ${DECISION_REASON}" >> ${LOG_DIR}/auto_tune.log
    
    # 更新状态文件
    cat > "$STATE_FILE" << STATEEOF
LAST_ROUND=B${CUR_ROUND}
BEST_ALLP=${NEW_BEST_ALLP2}
BEST_PCP=${NEW_BEST_PCP2}
BEST_CONFIG=${NEW_BEST_CFG}
ROUNDS_NO_IMPROV=${NEW_NI}
STATEEOF
fi

# 检查终止条件
if echo "$DECISION_REASON" | grep -q "最终报告"; then
    echo "[$(date)] ===== LOOP ENDED: $DECISION_REASON =====" >> ${LOG_DIR}/auto_tune.log
    echo "BEST_ALLP=${NEW_BEST_ALLP2} BEST_PCP=${NEW_BEST_PCP2} CONFIG=${NEW_BEST_CFG}" >> ${LOG_DIR}/auto_tune.log
    exit 0
fi

if [ "$NEXT_ROUND" -gt 15 ]; then
    echo "[$(date)] ===== LOOP ENDED: Reached max 15 rounds =====" >> ${LOG_DIR}/auto_tune.log
    exit 0
fi

# ======== 应用配置 ========
echo "[$(date)] Starting B${NEXT_ROUND}: ${DECISION_REASON}" >> ${LOG_DIR}/auto_tune.log
echo "  Config: lr=${NEW_LR}, wd=${NEW_WD}, do=${NEW_DO}, physics=${NEW_PP}, ep=${NEW_EP}" >> ${LOG_DIR}/auto_tune.log

# 修改 model config
# lr
if grep -q "lr:" "$MODEL_CFG"; then
    sed -i "s/lr: [0-9.e+-]\+/lr: ${NEW_LR}/" "$MODEL_CFG"
fi
# weight_decay  
if grep -q "weight_decay:" "$MODEL_CFG"; then
    sed -i "s/weight_decay: [0-9.e+-]\+/weight_decay: ${NEW_WD}/" "$MODEL_CFG"
fi
# physics_aux_prob
if grep -q "physics_aux_prob:" "$MODEL_CFG"; then
    sed -i "s/physics_aux_prob: [0-9.]\+/physics_aux_prob: ${NEW_PP}/" "$MODEL_CFG"
fi
# attention_dropout
if grep -q "attention_dropout:" "$MODEL_CFG"; then
    sed -i "s/attention_dropout: [0-9.]\+/attention_dropout: ${NEW_DO}/" "$MODEL_CFG"
fi

# 修改 run config
sed -i "s/^epochs: [0-9]\+/epochs: ${NEW_EP}/" "$RUN_CFG"
sed -i "s/run_name: 'mpd_pempni_test2val_B[0-9]\+_'/run_name: 'mpd_pempni_test2val_B${NEXT_ROUND}_'/" "$RUN_CFG"

# ======== 启动训练 ========
nohup ${PYTHON} -u run.py finetune \
    --model_config "$MODEL_CFG" \
    --data_config config/datasets/MPD_pempni_test2val.yml \
    --run_config "$RUN_CFG" \
    --stage ddG \
    > ${LOG_DIR}/mpd_test2val_B${NEXT_ROUND}_train.log 2>&1 &

echo "[$(date)] B${NEXT_ROUND} started, PID=$!" >> ${LOG_DIR}/auto_tune.log
